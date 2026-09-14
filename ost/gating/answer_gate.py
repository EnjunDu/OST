"""The answer gate.

OST answers at the first time both checks pass: the represented evidence is sufficient,
and no answer-critical claim is still scheduled for review. The second check is a hard
registry rule rather than something the learned gate can override, which is what stops
the model answering from an interpretation it has itself scheduled a test for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol, Set

import torch
from torch import nn

from ost.config import GateConfig
from ost.models.backbone import Backbone
from ost.retraction.algebra import ancestors
from ost.retraction.registry import SpanRegistry
from ost.state.omni_state import OmniState
from ost.types import Sufficiency

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Answer support sets
# --------------------------------------------------------------------------------------


def answer_supporting_span_ids(registry: SpanRegistry) -> Set[int]:
    """Span identifiers declared to support the answer."""
    return {
        span_id
        for span_id, span in registry.spans.items()
        if span.scope == "supports_answer"
    }


def unresolved_answer_support(registry: SpanRegistry) -> Set[int]:
    """``U_s(q)``: unsettled claims that transitively support the answer.

    Implements App. A.6::

        U_s(q) = { i in Gamma_s^claim : settled_s(i) = 0,
                   i <= j for some j with sigma_j = supports_answer }

    ``i <= j`` holds when ``i == j`` or ``i`` is an ancestor of ``j``, so a claim counts
    as answer support even when it only supports the answer through a chain of other
    records.
    """
    targets = answer_supporting_span_ids(registry)
    if not targets:
        return set()

    parents = registry.parents
    supporting_spans: Set[int] = set(targets)
    for target in targets:
        supporting_spans |= ancestors(parents, target)

    out: Set[int] = set()
    for claim in registry.claims.values():
        # settled_s(i) = 1 after confirmation or refutation; expired claims are not
        # settled, so they remain part of the answer's unresolved support.
        if claim.is_settled:
            continue
        if claim.span_id in supporting_spans:
            out.add(claim.claim_id)
    return out


def reviewable_answer_support(registry: SpanRegistry) -> Set[int]:
    """``A_s(q) = Q_s ∩ U_s(q)``: unresolved answer support still awaiting review."""
    return {cid for cid in unresolved_answer_support(registry) if cid in registry.queue}


def expired_answer_support(registry: SpanRegistry) -> Set[int]:
    """``E_s^ans``: answer support that has expired by this chunk (App. A.6)."""
    unresolved = unresolved_answer_support(registry)
    return {
        cid
        for cid in unresolved
        if registry.claims[cid].is_expired
    }


# --------------------------------------------------------------------------------------
# Learned gate head
# --------------------------------------------------------------------------------------


class LearnedGateHead(nn.Module):
    """Scalar sufficiency head over the frozen backbone's pooled representation.

    ``F_gate`` of Eq. (16) is this head's probability read against
    :attr:`GateConfig.threshold`. It is the soft half of the gate; the hard registry
    check above it cannot be overridden by whatever this returns.
    """

    def __init__(self, hidden_dim: int = 2048, head_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim < 1 or head_dim < 1:
            raise ValueError("gate head dimensions must be positive")
        self.hidden_dim = int(hidden_dim)
        self.head_dim = int(head_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.head_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.head_dim, 1),
        )
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.dim() != 2:
            raise ValueError(f"pooled must be 2-D, got {tuple(pooled.shape)}")
        if pooled.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"pooled has width {pooled.shape[-1]} but the gate head expects "
                f"{self.hidden_dim}"
            )
        return self.mlp(pooled).squeeze(-1)

    def probability(self, pooled: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(pooled))

    @classmethod
    def load(cls, directory: str | Path, map_location: str = "cpu") -> "LearnedGateHead":
        """Read the head, taking its geometry from the checkpoint."""
        path = Path(directory).expanduser() / "gate_head.pt"
        if not path.exists():
            raise FileNotFoundError(f"no gate head at {path}")
        payload = torch.load(path, map_location=map_location)
        head = cls(
            hidden_dim=int(payload.get("hidden_dim", 2048)),
            head_dim=int(payload.get("head_dim", 256)),
        )
        head.load_state_dict(payload["state_dict"])
        head.eval()
        return head


class GateScorer(Protocol):
    """Produces ``Suff_s`` from the question, provisional state and effective ledger."""

    def score_sufficiency(
        self, question: str, state: OmniState, ledger_text: str
    ) -> float:
        ...


#: The instruction the gate lane is read under. It asks about the represented state rather
#: than about the world, because the gate decides whether what has been represented is
#: enough, not whether an answer happens to be guessable.
GATE_INSTRUCTION = (
    "Decide whether the represented state already contains enough to answer the question.\n"
    "Answer only if the facts needed are present in the state itself, not merely plausible."
)


class HeadGateScorer:
    """Adapts a loaded gate head to :class:`GateScorer`.

    The head scores the effective ledger, so a refuted claim reaches the gate already
    down-weighted rather than as a fact the gate has to discount itself.
    """

    def __init__(self, head: LearnedGateHead, backbone: Backbone) -> None:
        self.head = head
        self.backbone = backbone

    def score_sufficiency(
        self, question: str, state: OmniState, ledger_text: str
    ) -> float:
        body = state.body.as_dict() if hasattr(state, "body") else {}
        fields = "\n".join(f"{key}: {value}" for key, value in body.items())
        text = (
            f"{GATE_INSTRUCTION}\n\n"
            f"Question: {question}\n"
            f"Provisional Omni-State:\n{fields}\n"
            f"Effective ledger:\n{ledger_text}"
        )
        pooled = self.backbone.pooled_representation(text)
        device = next(self.head.parameters()).device
        with torch.no_grad():
            probability = self.head.probability(
                pooled.unsqueeze(0).to(device=device, dtype=torch.float32)
            )
        return float(probability.item())


# --------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------


@dataclass
class GateDecision:
    """One gate decision and why it came out that way."""

    action: Sufficiency
    #: True when the hard registry rule forced a wait.
    hard_wait: bool
    #: ``Suff_s``; ``None`` when the hard rule short-circuited the learned gate.
    sufficiency_score: Optional[float]
    reviewable_support: List[int] = field(default_factory=list)
    unresolved_support: List[int] = field(default_factory=list)
    expired_support: List[int] = field(default_factory=list)

    @property
    def is_answer(self) -> bool:
        return self.action is Sufficiency.ANSWER


class AnswerGate:
    """Implements Eq. (8) and Eq. (16)."""

    def __init__(
        self,
        config: GateConfig,
        scorer: Optional[GateScorer] = None,
    ) -> None:
        config.validate()
        self.config = config
        self.scorer = scorer

    def decide(
        self,
        registry: SpanRegistry,
        *,
        question: str,
        state: OmniState,
        ledger_text: str,
        is_terminal: bool,
    ) -> GateDecision:
        """Decide whether to answer at this chunk.

        Implements Eq. (16)::

            act_s = Wait                                if Q_s ∩ U_s(q) != empty
                    F_gate(q, z_bar_s, M_hat_s^eff)     otherwise

        and the terminal rule of Eq. (8): ``T = min(T*, T_end)``, so the terminal chunk
        answers regardless of the gate.
        """
        unresolved = sorted(unresolved_answer_support(registry))
        reviewable = sorted(reviewable_answer_support(registry))
        expired = sorted(expired_answer_support(registry))

        if is_terminal:
            # Eq. (8): OST answers at T_end if no admissible chunk was reached. Remaining
            # open claims have already been expired by the caller, so the answer carries
            # their identifiers with a low-confidence flag rather than waiting further.
            return GateDecision(
                action=Sufficiency.ANSWER,
                hard_wait=False,
                sufficiency_score=None,
                reviewable_support=reviewable,
                unresolved_support=unresolved,
                expired_support=expired,
            )

        if reviewable:
            # The hard support check. An expired claim has left the queue, so it no longer
            # forces a wait even though it stays in the answer's unresolved support.
            return GateDecision(
                action=Sufficiency.WAIT,
                hard_wait=True,
                sufficiency_score=None,
                reviewable_support=reviewable,
                unresolved_support=unresolved,
                expired_support=expired,
            )

        score: Optional[float] = None
        if self.config.learned_head_enabled and self.scorer is not None:
            score = float(self.scorer.score_sufficiency(question, state, ledger_text))
            action = (
                Sufficiency.ANSWER if score >= self.config.threshold else Sufficiency.WAIT
            )
        else:
            # Without a learned gate only the hard rule applies. This is the matched
            # control the paper labels a hard support gate, and it must be selected
            # explicitly rather than arrived at by a missing checkpoint.
            LOGGER.debug("learned gate unavailable; falling back to the hard rule only")
            action = Sufficiency.WAIT

        return GateDecision(
            action=action,
            hard_wait=False,
            sufficiency_score=score,
            reviewable_support=reviewable,
            unresolved_support=unresolved,
            expired_support=expired,
        )


__all__ = [
    "GATE_INSTRUCTION",
    "AnswerGate",
    "GateDecision",
    "GateScorer",
    "HeadGateScorer",
    "LearnedGateHead",
    "answer_supporting_span_ids",
    "expired_answer_support",
    "reviewable_answer_support",
    "unresolved_answer_support",
]
