"""Claim admission.

The forecaster proposes freely; admission decides what becomes a registered claim. Every
condition here exists because an inadmissible claim would be unverifiable: a claim with a
past interval cannot be tested against future evidence, one without a modality cannot be
routed to a verifier, and one beyond capacity would displace a claim with an earlier
deadline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ost.config import ClaimConfig
from ost.types import MODALITIES, SUPPORT_SCOPES, ForecastProposal

LOGGER = logging.getLogger(__name__)

#: Cue words that mark a proposal as addressing both modalities separately rather than
#: relationally. App. A.3 registers separately testable audio and visual facts as two
#: claims; a genuinely relational claim (source attribution) stays a single claim.
_RELATIONAL_CUES = (
    "same source",
    "comes from",
    "produced by",
    "emitted by",
    "attributable to",
    "matches the",
    "belongs to",
    "is the source",
    "on-screen source",
    "speaker is",
)


@dataclass
class RejectedProposal:
    """A proposal that did not pass admission, with the reason."""

    proposal: ForecastProposal
    reason: str


@dataclass
class AdmissionOutcome:
    """Result of applying the admission rule at one chunk."""

    admitted: List[ForecastProposal] = field(default_factory=list)
    rejected: List[RejectedProposal] = field(default_factory=list)

    @property
    def any_rejected(self) -> bool:
        return bool(self.rejected)

    def rejection_reasons(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in self.rejected:
            counts[entry.reason] = counts.get(entry.reason, 0) + 1
        return counts


class AdmissionRule:
    """Implements ``Admit_{C_max}`` of Eq. (17) and the App. A.2 admission conditions."""

    def __init__(self, config: ClaimConfig) -> None:
        config.validate()
        self.config = config
        self._allowed_durations = {round(float(d), 6) for d in config.window_durations}

    # -- validity ------------------------------------------------------------------

    def check_valid(
        self,
        proposal: ForecastProposal,
        *,
        issue_time: float,
        stream_end: Optional[float] = None,
    ) -> Optional[str]:
        """Return a rejection reason, or ``None`` when the proposal is admissible.

        App. A.2 requires a query-critical proposition, a strictly future interval, a
        verifying modality, and a support scope of ``supports_state``, ``supports_claim``
        or ``supports_answer``.
        """
        if proposal.is_no_forecast:
            return "no_forecast"
        if not proposal.text.strip():
            return "empty_text"
        if not proposal.query_critical:
            return "not_query_critical"
        if proposal.modality not in MODALITIES:
            return "invalid_modality"
        if proposal.scope not in SUPPORT_SCOPES:
            return "invalid_scope"

        delta = round(float(proposal.delta_seconds), 6)
        if delta not in self._allowed_durations:
            return "window_not_allowed"
        if delta <= 0:
            return "non_future_interval"
        if delta > self.config.max_window_seconds + 1e-9:
            return "window_too_long"

        # A strictly future interval: (t, t + delta] must open after the current boundary.
        if stream_end is not None and issue_time >= stream_end - 1e-9:
            return "stream_ended"
        return None

    # -- ranking -------------------------------------------------------------------

    def rank(self, proposals: Sequence[ForecastProposal]) -> List[ForecastProposal]:
        """Order proposals by admission priority.

        App. A.2: when capacity or depth is reached, answer-supporting claims and earlier
        deadlines take priority. Confidence breaks remaining ties.
        """
        def key(proposal: ForecastProposal) -> Tuple[int, float, float]:
            scope_rank = 0 if proposal.scope == "supports_answer" else 1
            return (scope_rank, float(proposal.delta_seconds), -float(proposal.confidence))

        return sorted(proposals, key=key)

    # -- admission -----------------------------------------------------------------

    def admit(
        self,
        proposals: Sequence[ForecastProposal],
        *,
        issue_time: float,
        active_claim_count: int,
        depth_of: Optional[Dict[int, int]] = None,
        stream_end: Optional[float] = None,
        is_terminal: bool = False,
    ) -> AdmissionOutcome:
        """Apply the admission rule and the capacity bounds.

        The terminal chunk emits no new claims (App. A.5, Alg. 1), so every proposal is
        rejected there.
        """
        outcome = AdmissionOutcome()
        if is_terminal:
            for proposal in proposals:
                outcome.rejected.append(RejectedProposal(proposal, "terminal_chunk"))
            return outcome

        candidates: List[ForecastProposal] = []
        for proposal in proposals:
            reason = self.check_valid(
                proposal, issue_time=issue_time, stream_end=stream_end
            )
            if reason is None:
                candidates.append(proposal)
            elif reason != "no_forecast":
                outcome.rejected.append(RejectedProposal(proposal, reason))

        remaining_active = max(0, self.config.max_active - int(active_claim_count))
        per_chunk = self.config.max_per_chunk
        budget = min(per_chunk, remaining_active)

        for proposal in self.rank(candidates):
            if len(outcome.admitted) >= budget:
                reason = (
                    "chunk_capacity"
                    if len(outcome.admitted) >= per_chunk
                    else "active_capacity"
                )
                outcome.rejected.append(RejectedProposal(proposal, reason))
                continue
            if depth_of is not None and self._exceeds_depth(proposal, depth_of):
                outcome.rejected.append(RejectedProposal(proposal, "depth_capacity"))
                continue
            outcome.admitted.append(proposal)

        if outcome.rejected:
            LOGGER.debug(
                "admission rejected %d proposal(s): %s",
                len(outcome.rejected),
                outcome.rejection_reasons(),
            )
        return outcome

    def _exceeds_depth(
        self, proposal: ForecastProposal, depth_of: Dict[int, int]
    ) -> bool:
        """Whether admitting this proposal would exceed ``D_max`` support depth."""
        if not proposal.citations:
            return False
        deepest = max(
            (int(depth_of.get(int(cid), 0)) for cid in proposal.citations), default=0
        )
        return deepest + 1 > self.config.max_depth

    # -- modality splitting ---------------------------------------------------------

    def split_separable(
        self, proposals: Sequence[ForecastProposal]
    ) -> List[ForecastProposal]:
        """Split separately testable audio-visual proposals into two claims.

        App. A.3: separately testable audio and visual facts are registered as two claims,
        while audio-visual relational claims are evaluated jointly on the paired evidence.
        A proposal only stays relational when its text asserts a relation between the
        streams; otherwise a joint verdict would conflate two independent facts.
        """
        out: List[ForecastProposal] = []
        for proposal in proposals:
            if proposal.modality != "audio_visual" or self._is_relational(proposal.text):
                out.append(proposal)
                continue
            for modality in ("audio", "visual"):
                out.append(
                    ForecastProposal(
                        text=proposal.text,
                        modality=modality,  # type: ignore[arg-type]
                        delta_seconds=proposal.delta_seconds,
                        scope=proposal.scope,
                        citations=proposal.citations,
                        query_critical=proposal.query_critical,
                        confidence=proposal.confidence,
                    )
                )
        return out

    @staticmethod
    def _is_relational(text: str) -> bool:
        lowered = text.lower()
        return any(cue in lowered for cue in _RELATIONAL_CUES)

    # -- rejected dependencies -----------------------------------------------------

    @staticmethod
    def dependent_fields(outcome: AdmissionOutcome) -> Set[str]:
        """State-body fields that depended on a rejected claim.

        App. A.2: a field depending on a rejected claim becomes ``uncertain``, and the gate
        returns Wait. A ``supports_state`` rejection therefore has to be reflected back
        into the state rather than silently dropped.
        """
        fields: Set[str] = set()
        for entry in outcome.rejected:
            if entry.reason == "no_forecast":
                continue
            if entry.proposal.scope != "supports_state":
                continue
            if entry.proposal.modality == "audio":
                fields.add("audio_evidence")
            elif entry.proposal.modality == "visual":
                fields.add("visual_evidence")
            else:
                fields.update({"audio_evidence", "visual_evidence"})
        return fields


__all__ = [
    "AdmissionOutcome",
    "AdmissionRule",
    "RejectedProposal",
]
