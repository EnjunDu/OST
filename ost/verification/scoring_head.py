"""The verifier scoring head.

The backbone stays frozen, so verification lives entirely in a small head over its pooled
hidden states. At inference the head is read from a checkpoint and evaluated; this module
defines its geometry so a checkpoint cannot load into a mismatched shape.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import torch
from torch import nn

from ost.models.backbone import Backbone, MediaWindow
from ost.types import MODALITIES, Interval, Modality

LOGGER = logging.getLogger(__name__)


class VerifierScoringHead(nn.Module):
    """Shared ``hidden_dim -> head_dim -> 1`` scoring head with a modality embedding.

    Implements the scoring function of Eq. (7)::

        s_i = sigmoid f_phi( T_theta[ x_i, Pi_{m_i} O_{I_i} ; emb(m_i) ] )

    ``T_theta[...]`` is the frozen backbone's pooled representation of the claim text
    together with the retained media of the declared modality; ``emb(m_i)`` is a learned
    modality embedding concatenated to it. The head is shared across modalities; what
    differs per modality is the verdict band the score is read against.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        head_dim: int = 512,
        modality_embedding_dim: int = 32,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or head_dim < 1:
            raise ValueError("hidden_dim and head_dim must be positive")
        if modality_embedding_dim < 1:
            raise ValueError("modality_embedding_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.head_dim = int(head_dim)
        self.modality_embedding_dim = int(modality_embedding_dim)

        self.modality_embedding = nn.Embedding(len(MODALITIES), self.modality_embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + self.modality_embedding_dim, self.head_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.head_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.modality_embedding.weight, std=0.02)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # -- forward -------------------------------------------------------------------

    def forward(
        self, pooled: torch.Tensor, modality_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return logits of shape ``(batch,)``.

        ``pooled`` is ``(batch, hidden_dim)``; ``modality_ids`` is ``(batch,)`` with
        indices into :data:`ost.types.MODALITIES`.
        """
        if pooled.dim() != 2:
            raise ValueError(f"pooled must be 2-D, got shape {tuple(pooled.shape)}")
        if pooled.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"pooled has width {pooled.shape[-1]} but the head expects "
                f"{self.hidden_dim}; check verifier.hidden_dim against the backbone"
            )
        if modality_ids.dim() != 1 or modality_ids.shape[0] != pooled.shape[0]:
            raise ValueError("modality_ids must be 1-D with one entry per row of pooled")

        embedded = self.modality_embedding(modality_ids.to(pooled.device))
        features = torch.cat([pooled, embedded.to(pooled.dtype)], dim=-1)
        return self.mlp(features).squeeze(-1)

    def score(self, pooled: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        """``s_i``, the sigmoid of the logit."""
        return torch.sigmoid(self.forward(pooled, modality_ids))

    # -- persistence ---------------------------------------------------------------

    def config_dict(self) -> Dict[str, int]:
        return {
            "hidden_dim": self.hidden_dim,
            "head_dim": self.head_dim,
            "modality_embedding_dim": self.modality_embedding_dim,
        }

    @classmethod
    def load(
        cls,
        directory: str | Path,
        map_location: str = "cpu",
        *,
        default_geometry: Optional[Mapping[str, int]] = None,
    ) -> "VerifierScoringHead":
        """Read the head, taking its geometry from the checkpoint's own metadata.

        The sidecar JSON is what keeps a checkpoint from loading into a differently-shaped
        head: without it a width mismatch would only surface as a wrong score.
        ``default_geometry`` covers a checkpoint that ships without the sidecar, in which
        case the geometry has to come from the configuration instead.
        """
        path = Path(directory).expanduser()
        meta_path = path / "verifier_head.json"
        weights_path = path / "verifier_head.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"no verifier head weights at {weights_path}")

        geometry: Dict[str, int] = dict(default_geometry or {})
        if meta_path.exists():
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            geometry.update(dict(payload.get("config", {})))
        else:
            LOGGER.warning(
                "%s has no verifier_head.json; taking the head geometry from the "
                "configuration instead",
                path,
            )

        head = cls(
            hidden_dim=int(geometry.get("hidden_dim", 2048)),
            head_dim=int(geometry.get("head_dim", 512)),
            modality_embedding_dim=int(geometry.get("modality_embedding_dim", 32)),
        )
        state = torch.load(weights_path, map_location=map_location)
        head.load_state_dict(state)
        head.eval()
        return head


def modality_index(modality: Modality) -> int:
    """Index of a modality in the shared embedding table."""
    try:
        return MODALITIES.index(modality)
    except ValueError as exc:
        raise ValueError(f"unknown modality {modality!r}") from exc


def modality_id_tensor(
    modalities: Sequence[Modality], device: Optional[torch.device] = None
) -> torch.Tensor:
    return torch.tensor(
        [modality_index(m) for m in modalities], dtype=torch.long, device=device
    )


class HeadScorer:
    """Adapts a loaded head to the verifier's scorer protocol.

    The head consumes a pooled representation, so this is where a claim and the evidence
    retained for its declared modality are turned into the one text-plus-media call the
    backbone pools.
    """

    def __init__(
        self,
        head: VerifierScoringHead,
        backbone: Backbone,
        *,
        describe: Optional[Callable[[str, Mapping[str, Any], Modality], str]] = None,
    ) -> None:
        self.head = head
        self.backbone = backbone
        self.describe = describe

    def score_claim(
        self,
        claim_text: str,
        evidence: Mapping[str, Sequence[Any]],
        modality: Modality,
        interval: Interval,
    ) -> float:
        text = (
            self.describe(claim_text, evidence, modality)
            if self.describe is not None
            else (
                "Decide whether the evidence in the stated interval supports the claim.\n"
                f"Claim: {claim_text}\n"
                f"Verifying modality: {modality}\n"
                f"Evidence interval: ({interval.start:.1f}, {interval.end:.1f}]"
            )
        )
        media = media_from_evidence(evidence, interval)
        pooled = self.backbone.pooled_representation(text, media, modality=modality)
        device = next(self.head.parameters()).device
        with torch.no_grad():
            score = self.head.score(
                pooled.unsqueeze(0).to(device=device, dtype=torch.float32),
                modality_id_tensor([modality], device=device),
            )
        return float(score.item())


def media_from_evidence(
    evidence: Mapping[str, Sequence[Any]], interval: Interval
) -> Optional[MediaWindow]:
    """Pack the retained slots of one due window into a media window."""
    visual = list(evidence.get("visual", []) or [])
    audio = list(evidence.get("audio", []) or [])
    if not visual and not audio:
        return None
    return MediaWindow(
        visual=visual, audio=audio, t_start=interval.start, t_end=interval.end
    )


__all__ = [
    "HeadScorer",
    "VerifierScoringHead",
    "media_from_evidence",
    "modality_id_tensor",
    "modality_index",
]
