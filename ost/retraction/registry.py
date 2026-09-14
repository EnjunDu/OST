"""The span registry, review queue and verdict bookkeeping.

The registry owns every citable span, the claim subset, the provenance edges, the review
queue and the lifecycle transitions. It is append-only: a refutation lowers a reliability
score and records a verdict, it never deletes a record. That is what lets a correction
keep attenuating derived reasoning after the claim itself has settled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set

from ost.config import ClaimConfig, RetractionConfig
from ost.retraction.algebra import (
    ancestors,
    apply_refutation,
    propagate_effective,
    verify_non_amplification,
)
from ost.streaming.clock import DecisionClock
from ost.types import (
    ClaimRecord,
    ClaimStatus,
    ForecastProposal,
    Interval,
    SpanKind,
    SpanRecord,
    SupportScope,
    Verdict,
    VerdictLogEntry,
    VerificationResult,
)

LOGGER = logging.getLogger(__name__)


class RegistryError(ValueError):
    """Raised on an illegal registry operation."""


@dataclass
class RegistrySnapshot:
    """Pre-verification snapshot used to build the comparison branch.

    Sec. 3.3 and App. A.5: to compute the comparison logits, OST temporarily undoes the
    chunk's refutations, restoring the affected records' earlier reliability scores,
    lifecycle states and queue membership, and omitting those refutation verdicts.
    """

    chunk: int
    reliability: Dict[int, float]
    claim_reliability: Dict[int, float]
    claim_status: Dict[int, ClaimStatus]
    claim_next_review: Dict[int, int]
    claim_retry_scheduled: Dict[int, bool]
    queue: Set[int]
    verdict_log_length: int


class SpanRegistry:
    """Registry ``Gamma``, review queue ``Q`` and verdict log ``V``."""

    def __init__(
        self,
        claim_config: ClaimConfig,
        retraction_config: RetractionConfig,
        clock: DecisionClock,
        *,
        kappa: Optional[float] = None,
    ) -> None:
        claim_config.validate()
        retraction_config.validate()
        self.claim_config = claim_config
        self.retraction_config = retraction_config
        self.clock = clock
        self._kappa = kappa

        self.spans: Dict[int, SpanRecord] = {}
        self.claims: Dict[int, ClaimRecord] = {}
        #: Review queue Q: claim identifiers with a scheduled review.
        self.queue: Set[int] = set()
        self.verdict_log: List[VerdictLogEntry] = []
        #: Effective reliability rho_bar, recomputed by :meth:`propagate`.
        self.effective_reliability: Dict[int, float] = {}
        #: Revision links, kept apart from support edges (App. A.4, A.5).
        self.revision_links: Dict[int, List[int]] = {}

        self._next_span_id = 1
        self._next_claim_id = 1

    # -- properties ----------------------------------------------------------------

    @property
    def kappa(self) -> float:
        """Resolved retraction scale."""
        return self.retraction_config.resolved_kappa(self._kappa)

    def set_kappa(self, kappa: Optional[float]) -> None:
        """Install a calibrated retraction scale."""
        if kappa is not None and kappa <= 0:
            raise RegistryError("kappa must be positive")
        self._kappa = kappa

    @property
    def parents(self) -> Dict[int, List[int]]:
        """Support edges only. Revision links never enter ``Anc(.)``."""
        return {span_id: list(span.parents) for span_id, span in self.spans.items()}

    def open_claims(self) -> List[ClaimRecord]:
        return [c for c in self.claims.values() if c.is_open]

    def active_claim_count(self) -> int:
        """Number of claims occupying the ``Q_max`` budget."""
        return len(self.queue)

    # -- span registration ---------------------------------------------------------

    def allocate_span_id(self) -> int:
        span_id = self._next_span_id
        self._next_span_id += 1
        return span_id

    def register_span(
        self,
        kind: SpanKind,
        text: str,
        chunk: int,
        *,
        parents: Sequence[int] = (),
        scope: Optional[SupportScope] = None,
        field_name: Optional[str] = None,
        state_id: Optional[int] = None,
        hidden_state: object = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> SpanRecord:
        """Register a citable span and link its support parents."""
        span = SpanRecord(
            span_id=self.allocate_span_id(),
            kind=kind,
            text=text,
            chunk=chunk,
            scope=scope,
            field_name=field_name,
            state_id=state_id,
            hidden_state=hidden_state,
            metadata=dict(metadata or {}),
        )
        self.spans[span.span_id] = span
        if parents:
            self.link_parents(span.span_id, parents)
        return span

    def link_parents(self, span_id: int, parents: Sequence[int]) -> List[int]:
        """Add support edges, dropping citations that would break acyclicity.

        App. A.2: earlier-only citations keep the graph acyclic. A citation to a span
        registered at the same or a later position is dropped rather than accepted,
        because accepting it would let a descendant bound its own ancestor.
        """
        span = self._span(span_id)
        added: List[int] = []
        for parent in parents:
            parent_id = int(parent)
            if parent_id == span_id:
                continue
            if parent_id not in self.spans:
                LOGGER.debug("dropping citation to unknown span %s", parent_id)
                continue
            if parent_id >= span_id:
                LOGGER.debug(
                    "dropping non-earlier citation %s -> %s", span_id, parent_id
                )
                continue
            if parent_id in span.parents:
                continue
            span.parents.append(parent_id)
            added.append(parent_id)
        return added

    def add_revision_link(self, span_id: int, revised_span_ids: Sequence[int]) -> None:
        """Record that a span revises earlier records.

        App. A.4 and A.5: a mention of a refuted claim in the conflict or revision record
        is a revision link, deliberately separate from the support parents used by
        ``Anc(.)``. This lets a new evidence-supported interpretation carry its own
        provenance while the attenuation of dependent reasoning survives.
        """
        span = self._span(span_id)
        for revised in revised_span_ids:
            revised_id = int(revised)
            if revised_id in self.spans and revised_id not in span.revision_of:
                span.revision_of.append(revised_id)
        self.revision_links[span_id] = list(span.revision_of)

    def auto_link_scope(
        self,
        claim: ClaimRecord,
        *,
        field_span_ids: Mapping[str, int],
        sufficiency_span_id: Optional[int] = None,
    ) -> List[int]:
        """Add the scope-determined automatic support edges.

        App. A.4: ``pa(j) = pa_model(j) union pa_auto(j)``. A ``supports_state`` claim
        becomes a parent of the indicated field; a ``supports_answer`` claim becomes a
        parent of ``sufficiency``. These links apply only to new dependent records and
        earlier active claims, so they never rewrite the body's own parents.
        """
        linked: List[int] = []
        if claim.scope == "supports_state":
            for field_name in ("audio_evidence", "visual_evidence", "conflict"):
                target = field_span_ids.get(field_name)
                if target is not None and target > claim.span_id:
                    linked.extend(self.link_parents(target, [claim.span_id]))
        elif claim.scope == "supports_answer" and sufficiency_span_id is not None:
            linked.extend(self.link_parents(sufficiency_span_id, [claim.span_id]))
        return linked

    # -- claim registration --------------------------------------------------------

    def register_claim(
        self,
        proposal: ForecastProposal,
        *,
        chunk: int,
        issue_time: float,
        state_id: Optional[int] = None,
        support_span_ids: Sequence[int] = (),
        hidden_state: object = None,
    ) -> ClaimRecord:
        """Register an admitted proposal as a pending claim.

        Registration assigns an identifier, the interval ``(t, t + delta]``, the review
        chunk, lifecycle status, dependencies and an initial reliability score of 1
        (Sec. 3.2, App. A.2).
        """
        if proposal.is_no_forecast:
            raise RegistryError("NO_FORECAST proposals are not registered")
        interval = Interval(issue_time, issue_time + float(proposal.delta_seconds))
        review_chunk = self.clock.review_chunk(issue_time, float(proposal.delta_seconds))

        parents = list(dict.fromkeys([*support_span_ids, *proposal.citations]))
        span = self.register_span(
            SpanKind.CLAIM,
            proposal.text,
            chunk,
            parents=parents,
            scope=proposal.scope,
            state_id=state_id,
            hidden_state=hidden_state,
            metadata={
                "modality": proposal.modality,
                "delta_seconds": float(proposal.delta_seconds),
            },
        )

        claim = ClaimRecord(
            claim_id=self._next_claim_id,
            span_id=span.span_id,
            text=proposal.text,
            modality=proposal.modality,
            interval=interval,
            scope=proposal.scope,
            registered_chunk=chunk,
            review_chunk=review_chunk,
            next_review_chunk=review_chunk,
            status=ClaimStatus.PENDING,
            reliability=1.0,
            depth=self.support_depth(span.span_id),
            state_id=state_id,
        )
        self._next_claim_id += 1
        self.claims[claim.claim_id] = claim
        self.queue.add(claim.claim_id)
        span.metadata["claim_id"] = claim.claim_id
        return claim

    def support_depth(self, span_id: int) -> int:
        """Claim-to-claim support depth of a span, bounded by ``D_max``."""
        depth = 0
        for ancestor_id in ancestors(self.parents, span_id):
            span = self.spans.get(ancestor_id)
            if span is not None and span.is_claim:
                depth += 1
        return depth

    def claim_for_span(self, span_id: int) -> Optional[ClaimRecord]:
        span = self.spans.get(span_id)
        if span is None or not span.is_claim:
            return None
        claim_id = span.metadata.get("claim_id")
        if claim_id is None:
            return None
        return self.claims.get(int(claim_id))

    # -- reviews -------------------------------------------------------------------

    def due_claims(self, chunk: int, *, now: Optional[float] = None) -> List[ClaimRecord]:
        """Claims in ``Q`` whose review is due at ``chunk``, earliest deadline first.

        ``now`` is the chunk's closing time. A claim is only returned once its evidence window
        is actually complete: when the stream terminates early, the final chunk can be shorter
        than the nominal interval, so a claim's scheduled review chunk can arrive before its
        window has closed. App. A.2 expires such a claim under the terminal rule rather than
        verifying it against an incomplete interval.
        """
        due: List[ClaimRecord] = []
        for claim_id in self.queue:
            claim = self.claims[claim_id]
            if not claim.reviewable_at(chunk):
                continue
            if now is not None and not claim.interval.is_complete_at(now):
                LOGGER.debug(
                    "claim %s is scheduled for review at chunk %s but its window %s is not "
                    "complete at t=%.3f; deferring to the terminal rule",
                    claim_id,
                    chunk,
                    claim.interval,
                    now,
                )
                continue
            due.append(claim)
        return sorted(due, key=lambda c: (c.next_review_chunk, c.claim_id))

    def apply_verdict(
        self,
        result: VerificationResult,
        *,
        chunk: int,
        time: float,
    ) -> ClaimRecord:
        """Apply one verdict and advance the claim's lifecycle.

        Implements the App. A.3 review rules and the Eq. (5) score update:

        * decisive verdict -> ``settled``; a refutation also lowers ``rho_i``
        * unresolved with a retry left -> ``stale``, retry at ``d_i + ceil(delta'/Delta)``
        * unresolved with no retry left -> ``expired``, unresolved status retained
        """
        claim = self._claim(result.claim_id)
        if not claim.is_open:
            raise RegistryError(
                f"claim {claim.claim_id} is {claim.status.value} and cannot be reviewed"
            )
        is_retry = claim.status is ClaimStatus.STALE

        if result.verdict is Verdict.REFUTED:
            claim.contradiction_margin = float(result.contradiction_margin)
            # Implements Eq. (5): rho_i <- min(rho_i, alpha(gamma_i)).
            claim.reliability = apply_refutation(
                claim.reliability,
                claim.contradiction_margin,
                self.retraction_config.rho_min,
                self.kappa,
            )
            self.spans[claim.span_id].reliability = claim.reliability
            claim.status = ClaimStatus.SETTLED
            self.queue.discard(claim.claim_id)
        elif result.verdict is Verdict.CONFIRMED:
            # App. A.4: confirmed claims keep their weights.
            claim.contradiction_margin = 0.0
            claim.status = ClaimStatus.SETTLED
            self.queue.discard(claim.claim_id)
        else:
            claim.contradiction_margin = 0.0
            if claim.retry_scheduled:
                claim.status = ClaimStatus.EXPIRED
                self.queue.discard(claim.claim_id)
            else:
                claim.status = ClaimStatus.STALE
                claim.retry_scheduled = True
                claim.next_review_chunk = self.clock.retry_chunk(
                    claim.review_chunk, self.claim_config.retry_slack_seconds
                )

        entry = VerdictLogEntry(
            chunk=chunk,
            time=time,
            claim_id=claim.claim_id,
            verdict=result.verdict,
            contradiction_margin=float(result.contradiction_margin),
            score=float(result.score),
            is_retry=is_retry,
            status_after=claim.status,
        )
        self.verdict_log.append(entry)
        return claim

    def expire(self, claim_id: int) -> ClaimRecord:
        """Expire an open claim.

        App. A.3: it leaves ``Q`` and releases its pinned interval, while its unresolved
        status remains in ``Gamma`` and the verdict log.
        """
        claim = self._claim(claim_id)
        if claim.is_open:
            claim.status = ClaimStatus.EXPIRED
        self.queue.discard(claim_id)
        return claim

    def expire_all_open(self, *, chunk: int, time: float) -> List[int]:
        """Expire every open claim at stream termination (Alg. 1)."""
        expired: List[int] = []
        for claim_id in sorted(self.queue):
            claim = self.claims[claim_id]
            claim.status = ClaimStatus.EXPIRED
            self.verdict_log.append(
                VerdictLogEntry(
                    chunk=chunk,
                    time=time,
                    claim_id=claim_id,
                    verdict=Verdict.UNRESOLVED,
                    contradiction_margin=0.0,
                    score=float("nan"),
                    is_retry=claim.retry_scheduled,
                    status_after=ClaimStatus.EXPIRED,
                )
            )
            expired.append(claim_id)
        self.queue.clear()
        return expired

    # -- propagation ---------------------------------------------------------------

    def propagate(self, *, check_invariant: bool = False) -> Dict[int, float]:
        """Recompute effective reliability over the active support subgraph.

        Implements Eq. (5) through the App. A.4 parent-before-child scan. Cost is
        ``O(|V_s| + |E_s|)`` over the active subgraph.
        """
        reliability = {span_id: span.reliability for span_id, span in self.spans.items()}
        effective = propagate_effective(reliability, self.parents)
        for span_id, value in effective.items():
            self.spans[span_id].effective_reliability = value
        self.effective_reliability = effective

        if check_invariant:
            violations = verify_non_amplification(reliability, self.parents, effective)
            if violations:
                raise RegistryError(
                    "provenance non-amplification violated for span/ancestor pairs "
                    f"{violations[:8]}"
                )
        return effective

    def effective_for_span(self, span_id: int) -> float:
        return float(self.effective_reliability.get(span_id, self.spans[span_id].reliability))

    # -- snapshots and branches ----------------------------------------------------

    def snapshot(self, chunk: int) -> RegistrySnapshot:
        """Take the pre-verification snapshot for the comparison branch."""
        return RegistrySnapshot(
            chunk=chunk,
            reliability={sid: span.reliability for sid, span in self.spans.items()},
            claim_reliability={cid: c.reliability for cid, c in self.claims.items()},
            claim_status={cid: c.status for cid, c in self.claims.items()},
            claim_next_review={cid: c.next_review_chunk for cid, c in self.claims.items()},
            claim_retry_scheduled={cid: c.retry_scheduled for cid, c in self.claims.items()},
            queue=set(self.queue),
            verdict_log_length=len(self.verdict_log),
        )

    def comparison_reliability(
        self, snapshot: RegistrySnapshot, refuted_claim_ids: Sequence[int]
    ) -> Dict[int, float]:
        """Effective reliability with this chunk's refutations undone.

        Only the records affected by ``refuted_claim_ids`` are restored; confirmations
        and unresolved verdicts from the same chunk stay in force, as App. A.5 requires.
        """
        reliability = {sid: span.reliability for sid, span in self.spans.items()}
        for claim_id in refuted_claim_ids:
            claim = self.claims.get(int(claim_id))
            if claim is None:
                continue
            restored = snapshot.reliability.get(claim.span_id)
            if restored is not None:
                reliability[claim.span_id] = restored
        return propagate_effective(reliability, self.parents)

    def comparison_queue(
        self, snapshot: RegistrySnapshot, refuted_claim_ids: Sequence[int]
    ) -> Set[int]:
        """Queue membership with this chunk's refutations undone."""
        queue = set(self.queue)
        for claim_id in refuted_claim_ids:
            cid = int(claim_id)
            if cid in snapshot.queue:
                queue.add(cid)
        return queue

    def comparison_status(
        self, snapshot: RegistrySnapshot, refuted_claim_ids: Sequence[int]
    ) -> Dict[int, ClaimStatus]:
        """Lifecycle states with this chunk's refutations undone."""
        status = {cid: claim.status for cid, claim in self.claims.items()}
        for claim_id in refuted_claim_ids:
            cid = int(claim_id)
            if cid in snapshot.claim_status:
                status[cid] = snapshot.claim_status[cid]
        return status

    # -- internals -----------------------------------------------------------------

    def _span(self, span_id: int) -> SpanRecord:
        span = self.spans.get(int(span_id))
        if span is None:
            raise RegistryError(f"unknown span {span_id}")
        return span

    def _claim(self, claim_id: int) -> ClaimRecord:
        claim = self.claims.get(int(claim_id))
        if claim is None:
            raise RegistryError(f"unknown claim {claim_id}")
        return claim

    def stats(self) -> Dict[str, object]:
        by_status: Dict[str, int] = {}
        for claim in self.claims.values():
            by_status[claim.status.value] = by_status.get(claim.status.value, 0) + 1
        return {
            "spans": len(self.spans),
            "claims": len(self.claims),
            "queue": len(self.queue),
            "verdicts": len(self.verdict_log),
            "claims_by_status": by_status,
            "kappa": self.kappa,
        }


__all__ = ["RegistryError", "RegistrySnapshot", "SpanRegistry"]
