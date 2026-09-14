"""Typed due-window verification.

A claim is checked only when its evidence window has closed, and only against the
modality that can test it. Both properties matter: checking early means the verifier sees
the same evidence that produced the claim, and checking on the wrong modality lets a
visual cue answer a question about sound.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Protocol, Sequence

from ost.config import VerifierConfig
from ost.streaming.retention import PerceptionSlot, RetentionBuffers
from ost.types import (
    ClaimRecord,
    Interval,
    Modality,
    VerificationResult,
    Verdict,
    own_modalities,
)
from ost.verification.thresholds import VerdictBands

LOGGER = logging.getLogger(__name__)

#: A window with coverage at or below this fraction is treated as uninformative: the
#: source became unobservable before the claimed event could be assessed (App. A.3).
DEFAULT_MIN_COVERAGE = 0.5


class VerifierScorer(Protocol):
    """Anything that can score a claim against retained evidence.

    The verifier of Eq. (7) is a head over the frozen backbone's pooled hidden states,
    and is the default. Stating it as a protocol is what lets the runtime fall back to a
    documented zero-shot scorer when no head is configured, without the surrounding loop
    behaving differently.
    """

    def score_claim(
        self,
        claim_text: str,
        evidence: Mapping[str, Sequence[PerceptionSlot]],
        modality: Modality,
        interval: Interval,
    ) -> float:
        """Return ``s_i`` in ``[0, 1]``: the probability the claim is supported."""
        ...


@dataclass
class EvidenceProjection:
    """The result of applying ``Pi_{m_i}`` to the retained observations."""

    modality: Modality
    interval: Interval
    slots: Dict[str, List[PerceptionSlot]]
    coverage: float
    #: True when the projection carries no usable evidence for the claim.
    uninformative: bool

    @property
    def is_paired(self) -> bool:
        return self.modality == "audio_visual"


def project_evidence(
    buffers: RetentionBuffers,
    interval: Interval,
    modality: Modality,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> EvidenceProjection:
    """Apply the modality projection ``Pi_{m_i} O_{I_i}``.

    App. A Notation and A.3: audio for audio claims, video for visual claims, and the
    pair for relational claims such as source attribution. A relational claim is
    evaluated jointly on the paired evidence in one forward pass, never as two
    independent checks.
    """
    slots = {name: list(items) for name, items in buffers.evidence_window(interval, modality).items()}
    names = list(own_modalities(modality))

    coverages = [buffers.budget(name).coverage(interval) for name in names]
    # A relational claim needs both streams, so its coverage is the weaker one.
    coverage = min(coverages) if coverages else 0.0
    empty = all(not slots.get(name) for name in names)
    uninformative = empty or coverage <= min_coverage

    return EvidenceProjection(
        modality=modality,
        interval=interval,
        slots=slots,
        coverage=coverage,
        uninformative=uninformative,
    )


class TypedVerifier:
    """``V_phi^{m_i}``: the scoring head together with the bands and margin rule.

    Implements Eq. (4) and Eq. (7). The verifier combines four things the paper keeps
    distinct: the modality projection, the score, the per-modality verdict band, and the
    calibration cap that keeps confirmation unavailable for an uninformative interval.
    """

    def __init__(
        self,
        config: VerifierConfig,
        scorer: VerifierScorer,
        *,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
    ) -> None:
        config.validate()
        self.config = config
        self.bands = VerdictBands.from_config(config)
        self.scorer = scorer
        self.min_coverage = float(min_coverage)

    # -- main entry point ----------------------------------------------------------

    def verify(
        self,
        claim: ClaimRecord,
        buffers: RetentionBuffers,
        *,
        now: Optional[float] = None,
    ) -> VerificationResult:
        """Verify one claim against its retained evidence.

        The claim is only verified once ``I_i`` is complete; calling earlier is a
        programming error, because the whole point of a future window is that the
        verifier compares the prediction against evidence that had not arrived when the
        claim was made.
        """
        current = float(now) if now is not None else buffers.now
        if not claim.interval.is_complete_at(current):
            raise ValueError(
                f"claim {claim.claim_id} has evidence window {claim.interval} which is "
                f"not complete at t={current}; App. A.2 reviews a claim only at the "
                "first chunk boundary at which its window is complete"
            )

        band = self.bands[claim.modality]
        projection = project_evidence(
            buffers, claim.interval, claim.modality, min_coverage=self.min_coverage
        )

        raw_score = float(
            self.scorer.score_claim(
                claim.text, projection.slots, claim.modality, claim.interval
            )
        )
        raw_score = min(max(raw_score, 0.0), 1.0)

        # App. A.3: for an uninformative interval, calibration imposes s_i <= AXIOM_CAP.
        # This keeps confirmation unavailable while preserving the unresolved band, so a
        # window that says nothing about the claim cannot be read as agreement.
        score = raw_score
        capped = False
        if projection.uninformative:
            score, capped = band.cap(raw_score)

        verdict = band.verdict(score)
        margin = band.margin(score)

        # App. A.3: the unresolved outcome also covers an interval in which the source
        # becomes unobservable before the claimed event can be assessed. Refuting on
        # missing evidence would let eviction masquerade as contradiction.
        unobservable = False
        if projection.uninformative and verdict is Verdict.REFUTED:
            has_any = any(projection.slots.get(name) for name in own_modalities(claim.modality))
            if not has_any:
                verdict = Verdict.UNRESOLVED
                margin = 0.0
                unobservable = True

        if verdict is not Verdict.REFUTED:
            # gamma_i grades retraction and guidance, both of which only apply to a
            # refutation (App. A.4, Eq. (6)).
            margin = 0.0

        return VerificationResult(
            claim_id=claim.claim_id,
            verdict=verdict,
            contradiction_margin=margin,
            score=score,
            modality=claim.modality,
            interval=claim.interval,
            capped=capped,
            unobservable=unobservable,
            detail={
                "raw_score": raw_score,
                "coverage": projection.coverage,
                "tau_low": band.low,
                "tau_high": band.high,
                "slot_counts": {k: len(v) for k, v in projection.slots.items()},
            },
        )

    def verify_due(
        self,
        claims: Sequence[ClaimRecord],
        buffers: RetentionBuffers,
        *,
        now: Optional[float] = None,
    ) -> List[VerificationResult]:
        """Verify every due claim, earliest deadline first."""
        ordered = sorted(claims, key=lambda c: (c.next_review_chunk, c.claim_id))
        return [self.verify(claim, buffers, now=now) for claim in ordered]


__all__ = [
    "DEFAULT_MIN_COVERAGE",
    "EvidenceProjection",
    "TypedVerifier",
    "VerifierScorer",
    "project_evidence",
]
