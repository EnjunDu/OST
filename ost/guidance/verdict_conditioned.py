"""Verdict-conditioned continuation.

Retraction lowers the influence of a refuted premise; this module changes what gets
written next. Two matched contexts differ only in whether this chunk's refutations are in
force, and the logit contrast between them steers generation toward the tokens the
correction supports. The same rule runs twice per chunk: first to rewrite the current
interpretation, then to forecast from that rewritten interpretation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import torch

from ost.retraction.registry import RegistrySnapshot, SpanRegistry
from ost.types import ClaimStatus, GenerationStage

LOGGER = logging.getLogger(__name__)


def guidance_scale(
    lambda_0: float, margins: Sequence[float]
) -> float:
    """Chunk guidance scale ``lambda_s``.

    Implements Eq. (6): ``lambda_s = (lambda_0 / |X_s|) * sum_{i in X_s} gamma_i``, that
    is ``lambda_0`` times the mean contradiction margin over the claims refuted at this
    chunk, and zero when no claim is refuted.
    """
    if lambda_0 < 0:
        raise ValueError("lambda_0 must be non-negative")
    if not margins:
        return 0.0
    mean_margin = sum(float(m) for m in margins) / float(len(margins))
    return float(lambda_0) * mean_margin


def guided_logits(
    logits_positive: torch.Tensor,
    logits_negative: Optional[torch.Tensor],
    scale: float,
) -> torch.Tensor:
    """Apply the guidance contrast to one position's logits.

    Implements Eq. (6) / Eq. (13)::

        l_tilde = l_plus + lambda_s * (l_plus - l_minus)

    With ``scale == 0`` or no comparison branch this is the corrected branch unchanged,
    which is exactly the behaviour App. A.5 specifies when ``X_s`` is empty.
    """
    if logits_negative is None or scale == 0.0:
        return logits_positive
    if logits_negative.shape != logits_positive.shape:
        raise ValueError(
            "comparison logits shape "
            f"{tuple(logits_negative.shape)} does not match corrected logits shape "
            f"{tuple(logits_positive.shape)}"
        )
    return logits_positive + float(scale) * (logits_positive - logits_negative)


def exponential_tilt(
    logits_positive: torch.Tensor,
    logits_negative: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """The guided distribution written as an exponential tilt of ``p_plus``.

    Theorem 1: the softmax of Eq. (13) equals ``p_plus(v) * exp(lambda * Delta(v)) / Z``
    with ``Delta(v) = log[p_plus(v) / p_minus(v)]``. Computing it this way is numerically
    identical to softmaxing the guided logits of :func:`guided_logits`; it is written out
    separately because this is the form in which the tilt, and its monotonicity in
    ``lambda``, can be read directly.
    """
    log_p_plus = torch.log_softmax(logits_positive, dim=-1)
    log_p_minus = torch.log_softmax(logits_negative, dim=-1)
    delta = log_p_plus - log_p_minus
    return torch.softmax(log_p_plus + float(scale) * delta, dim=-1)


def correction_gain(
    logits_positive: torch.Tensor, logits_negative: torch.Tensor
) -> torch.Tensor:
    """``Delta(v) = log[p_plus(v) / p_minus(v)]``, the correction-induced gain."""
    return torch.log_softmax(logits_positive, dim=-1) - torch.log_softmax(
        logits_negative, dim=-1
    )


@dataclass
class BranchState:
    """One decoding branch's view of the registry."""

    #: Effective reliability per span, propagated for this branch alone.
    effective_reliability: Dict[int, float]
    #: Claim lifecycle states as this branch sees them.
    claim_status: Dict[int, ClaimStatus]
    #: Review-queue membership as this branch sees it.
    queue: set
    #: Verdicts this branch reports, i.e. with the refutations of ``X_s`` omitted for the
    #: comparison branch.
    omit_claim_ids: Tuple[int, ...] = ()

    @property
    def is_comparison(self) -> bool:
        return bool(self.omit_claim_ids)


@dataclass
class BranchPair:
    """The matched context pair ``(C_s^+, C_s^-)`` of App. A.5."""

    corrected: BranchState
    comparison: Optional[BranchState]
    #: lambda_s, fixed for both stages of this chunk.
    scale: float
    refuted_claim_ids: Tuple[int, ...] = ()

    @property
    def has_contrast(self) -> bool:
        return self.comparison is not None and self.scale != 0.0

    @classmethod
    def from_snapshot(
        cls,
        registry: SpanRegistry,
        snapshot: Optional[RegistrySnapshot],
        refuted_claim_ids: Sequence[int],
        margins: Sequence[float],
        lambda_0: float,
    ) -> "BranchPair":
        """Build both branches from the pre-verification snapshot.

        App. A.5: the corrected branch uses the post-verdict registry and propagated
        reliability scores. The comparison branch restores the affected records'
        pre-refutation scores, lifecycle states and queue membership, and omits the
        verdicts for ``X_s``. Both contexts keep the same observed audio-visual evidence
        and the same confirmations and unresolved verdicts, so the only difference between
        them is this chunk's refutations.
        """
        refuted = tuple(int(cid) for cid in refuted_claim_ids)
        corrected = BranchState(
            effective_reliability=dict(registry.effective_reliability),
            claim_status={cid: c.status for cid, c in registry.claims.items()},
            queue=set(registry.queue),
        )
        scale = guidance_scale(lambda_0, margins)

        if not refuted or snapshot is None or scale == 0.0:
            return cls(corrected=corrected, comparison=None, scale=0.0, refuted_claim_ids=refuted)

        comparison = BranchState(
            effective_reliability=registry.comparison_reliability(snapshot, refuted),
            claim_status=registry.comparison_status(snapshot, refuted),
            queue=registry.comparison_queue(snapshot, refuted),
            omit_claim_ids=refuted,
        )
        return cls(
            corrected=corrected,
            comparison=comparison,
            scale=scale,
            refuted_claim_ids=refuted,
        )


class StageLogitFn(Protocol):
    """Evaluates one branch's next-token logits for a stage.

    Called with the branch, the generated token prefix so far, and the stage. Must return
    logits of shape ``(vocab,)`` for the next position. The two branches share the stage
    prompt and the generated prefix; only the serialised ledger differs.
    """

    def __call__(
        self,
        branch: BranchState,
        prefix: Sequence[int],
        stage: GenerationStage,
    ) -> torch.Tensor:
        ...


@dataclass
class StageResult:
    """Output of one guided generation stage."""

    token_ids: List[int]
    text: str
    #: Number of positions at which the contrast was actually applied.
    guided_positions: int = 0
    scale: float = 0.0
    finished: bool = True


class GuidedDecoder:
    """Runs the two stages of Eq. (13) under one decoding rule.

    The decoder is deliberately agnostic to the backbone: it is handed a per-branch logit
    function and a token sampler. That keeps the guidance arithmetic testable without a
    30B model and keeps the arithmetic in one place rather than inlined in a backend.
    """

    def __init__(
        self,
        logit_fn: StageLogitFn,
        *,
        decode_fn: Callable[[Sequence[int]], str],
        eos_token_ids: Sequence[int] = (),
        sampler: Optional[Callable[[torch.Tensor], int]] = None,
    ) -> None:
        self.logit_fn = logit_fn
        self.decode_fn = decode_fn
        self.eos_token_ids = set(int(t) for t in eos_token_ids)
        self.sampler = sampler or _greedy_sampler

    def run_stage(
        self,
        stage: GenerationStage,
        branches: BranchPair,
        *,
        max_new_tokens: int,
        prefix: Optional[Sequence[int]] = None,
        logits_processor: Optional[Callable[[Sequence[int], torch.Tensor], torch.Tensor]] = None,
    ) -> StageResult:
        """Generate one stage token by token under Eq. (13).

        Each selected token is appended to the shared prefix before both branches are
        evaluated again, so the two branches never diverge in what they have generated —
        only in what they know about the refutations.
        """
        tokens: List[int] = list(prefix or [])
        generated: List[int] = []
        guided_positions = 0

        for _ in range(int(max_new_tokens)):
            logits_positive = self.logit_fn(branches.corrected, tokens, stage)
            logits_negative = None
            if branches.has_contrast:
                logits_negative = self.logit_fn(branches.comparison, tokens, stage)

            merged = guided_logits(logits_positive, logits_negative, branches.scale)
            if logits_negative is not None and branches.scale != 0.0:
                guided_positions += 1

            if logits_processor is not None:
                # Grammar constraints are applied after guidance so that the contrast can
                # never push probability onto a token the schema forbids.
                merged = logits_processor(generated, merged)

            token = int(self.sampler(merged))
            if token in self.eos_token_ids:
                return StageResult(
                    token_ids=generated,
                    text=self.decode_fn(generated),
                    guided_positions=guided_positions,
                    scale=branches.scale,
                    finished=True,
                )
            generated.append(token)
            tokens.append(token)

        LOGGER.debug(
            "stage %s hit the %d-token budget without an end-of-sequence token",
            stage.value,
            max_new_tokens,
        )
        return StageResult(
            token_ids=generated,
            text=self.decode_fn(generated),
            guided_positions=guided_positions,
            scale=branches.scale,
            finished=False,
        )


def _greedy_sampler(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1).item())


def make_sampler(temperature: float, top_p: float, generator: Optional[torch.Generator] = None):
    """Build a sampler. Temperature 0 is greedy, which keeps evaluation deterministic."""
    if temperature <= 0:
        return _greedy_sampler

    def _sample(logits: torch.Tensor) -> int:
        scaled = logits.float() / float(temperature)
        probs = torch.softmax(scaled, dim=-1)
        if 0 < top_p < 1:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep = cumulative - sorted_probs <= top_p
            keep[0] = True
            filtered = torch.zeros_like(probs)
            filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
            probs = filtered / filtered.sum()
        return int(torch.multinomial(probs, num_samples=1, generator=generator).item())

    return _sample


__all__ = [
    "BranchPair",
    "BranchState",
    "GuidedDecoder",
    "StageLogitFn",
    "StageResult",
    "correction_gain",
    "exponential_tilt",
    "guidance_scale",
    "guided_logits",
    "make_sampler",
]
