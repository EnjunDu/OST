"""Core data types for Omni-Streaming Thinking.

Naming follows the paper. Two symbols are deliberately kept apart because conflating
them silently breaks both retraction and scheduling:

* ``contradiction_margin`` is ``gamma_i`` in [0, 1]: how strongly the retained evidence
  contradicts the claim. It scales retraction and guidance.
* ``review_chunk`` / ``next_review_chunk`` is the scheduled review index ``d_i``.

Reliability is called ``rho`` throughout, matching Eq. (5); ``effective_reliability`` is
``rho_bar``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Enumerations and literals
# --------------------------------------------------------------------------------------

#: Verifying modality m_i. ``audio_visual`` covers relational claims such as source
#: attribution, which App. A.3 evaluates jointly on the paired evidence.
Modality = Literal["audio", "visual", "audio_visual"]
MODALITIES: Tuple[Modality, ...] = ("audio", "visual", "audio_visual")

#: Mandatory ``audio_state`` values of a base Omni-State (Sec. 3.1, App. A.2).
AudioState = Literal["present", "absent", "uncertain"]
AUDIO_STATES: Tuple[AudioState, ...] = ("present", "absent", "uncertain")

#: Aggregate ``audio_state`` values a folded Omni-State may take (App. A.1).
AggregateAudioState = Literal["all_absent", "some_present", "mixed", "uncertain"]
AGGREGATE_AUDIO_STATES: Tuple[AggregateAudioState, ...] = (
    "all_absent",
    "some_present",
    "mixed",
    "uncertain",
)

#: Support scope sigma_i (Sec. 3.2, App. A.2).
SupportScope = Literal["supports_state", "supports_claim", "supports_answer"]
SUPPORT_SCOPES: Tuple[SupportScope, ...] = (
    "supports_state",
    "supports_claim",
    "supports_answer",
)

#: The six Omni-State fields, in paper order (Sec. 3.1).
OMNI_STATE_FIELDS: Tuple[str, ...] = (
    "visual_evidence",
    "audio_state",
    "audio_evidence",
    "conflict",
    "forecast",
    "sufficiency",
)

#: The state body z_s^body is the first four fields (App. A Notation).
STATE_BODY_FIELDS: Tuple[str, ...] = OMNI_STATE_FIELDS[:4]

#: Sentinel emitted by the forecaster when no query-critical proposition needs a future
#: test. Serialised as an empty ``forecast`` field (App. A.5).
NO_FORECAST = "NO_FORECAST"


class Sufficiency(str, Enum):
    """Gate action written into the ``sufficiency`` field (Sec. 3.1, Eq. (16))."""

    WAIT = "wait"
    ANSWER = "answer"

    @property
    def is_answer(self) -> bool:
        return self is Sufficiency.ANSWER


class ClaimStatus(str, Enum):
    """Claim lifecycle (App. A.3).

    ``PENDING`` until the review chunk, ``SETTLED`` on a decisive verdict, ``STALE``
    after an unresolved verdict with a retry scheduled, ``EXPIRED`` once no review
    remains.
    """

    PENDING = "pending"
    STALE = "stale"
    SETTLED = "settled"
    EXPIRED = "expired"

    @property
    def is_open(self) -> bool:
        return self in (ClaimStatus.PENDING, ClaimStatus.STALE)


class Verdict(str, Enum):
    """Verifier outcome nu_i (Sec. 3.3, Eq. (4))."""

    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    UNRESOLVED = "unresolved"

    @property
    def is_decisive(self) -> bool:
        return self in (Verdict.CONFIRMED, Verdict.REFUTED)


class SpanKind(str, Enum):
    """What a registered span represents."""

    STATE_FIELD = "state_field"
    CLAIM = "claim"
    FOLD_SUMMARY = "fold_summary"
    ANSWER = "answer"


class GenerationStage(str, Enum):
    """The two stages of Eq. (13): ``r in {rewrite, forecast}``."""

    REWRITE = "rewrite"
    FORECAST = "forecast"


# --------------------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """A half-open interval ``(start, end]`` on the absolute stream timeline."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"interval end {self.end} precedes start {self.start}")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def contains(self, t: float) -> bool:
        return self.start < t <= self.end

    def is_complete_at(self, t: float) -> bool:
        """True once the whole interval has been observed by time ``t``."""
        return t >= self.end - 1e-9

    def overlaps(self, other: "Interval") -> bool:
        return self.start < other.end and other.start < self.end

    def union(self, other: "Interval") -> "Interval":
        return Interval(min(self.start, other.start), max(self.end, other.end))

    def as_tuple(self) -> Tuple[float, float]:
        return (self.start, self.end)

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"({self.start:.1f},{self.end:.1f}]"


# --------------------------------------------------------------------------------------
# Spans and provenance
# --------------------------------------------------------------------------------------


@dataclass
class SpanRecord:
    """A citable span ``g_i`` in the registry ``Gamma`` (App. A.2).

    ``g_i = (h_i, span_i, rho_i, pa(i), sigma_i)``: pooled hidden states, token span,
    reliability score, support parents, and support scope.
    """

    span_id: int
    kind: SpanKind
    text: str
    chunk: int
    #: Support parents pa(i) = pa_model(i) union pa_auto(i) (App. A.4).
    parents: List[int] = field(default_factory=list)
    #: Revision links are stored apart from support edges and never enter Anc(.).
    revision_of: List[int] = field(default_factory=list)
    #: rho_i, the span's own reliability score. Non-claim spans stay at 1.
    reliability: float = 1.0
    #: rho_bar_i, recomputed by propagation each chunk.
    effective_reliability: float = 1.0
    scope: Optional[SupportScope] = None
    #: Token offsets in the serialised ledger, filled in at serialisation time.
    token_span: Optional[Tuple[int, int]] = None
    #: Which Omni-State field this span realises, when applicable.
    field_name: Optional[str] = None
    state_id: Optional[int] = None
    #: h_i, pooled hidden states. Held only while needed by the verifier.
    hidden_state: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_claim(self) -> bool:
        return self.kind is SpanKind.CLAIM


@dataclass
class ClaimRecord:
    """A registered claim ``c_i = (x_i, m_i, I_i, sigma_i)`` plus lifecycle metadata.

    Created by :meth:`ost.retraction.registry.SpanRegistry.register_claim` from an
    admitted :class:`ForecastProposal`.
    """

    claim_id: int
    span_id: int
    #: x_i, the free-text forecast.
    text: str
    #: m_i, the verifying modality.
    modality: Modality
    #: I_i = (t_i, t_i + delta_i].
    interval: Interval
    #: sigma_i, the support scope.
    scope: SupportScope
    #: Decision chunk r_i at which the claim was registered.
    registered_chunk: int
    #: d_i = ceil((t_i + delta_i) / Delta_chunk), the scheduled review chunk.
    review_chunk: int
    #: The next chunk at which a review is scheduled; equals review_chunk until a retry.
    next_review_chunk: int
    status: ClaimStatus = ClaimStatus.PENDING
    #: rho_i, initialised to 1 at registration (Sec. 3.2).
    reliability: float = 1.0
    #: gamma_i from the most recent decisive review.
    contradiction_margin: float = 0.0
    #: Whether the single permitted retry has been scheduled.
    retry_scheduled: bool = False
    #: Whether the claim's evidence interval is still pinned in the perceptual buffers.
    pinned: bool = True
    #: Support-graph depth, bounded by D_max.
    depth: int = 0
    #: Identifier of the Omni-State whose forecast field holds this claim.
    state_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def delta_seconds(self) -> float:
        """delta_i, the evidence-window duration."""
        return self.interval.duration

    @property
    def is_open(self) -> bool:
        return self.status.is_open

    @property
    def is_settled(self) -> bool:
        return self.status is ClaimStatus.SETTLED

    @property
    def is_expired(self) -> bool:
        return self.status is ClaimStatus.EXPIRED

    @property
    def supports_answer(self) -> bool:
        return self.scope == "supports_answer"

    def reviewable_at(self, chunk: int) -> bool:
        """True when an initial or retry review is due at ``chunk``."""
        return self.is_open and chunk >= self.next_review_chunk

    def has_scheduled_review(self) -> bool:
        """``reviewable_s(i)``: an initial or retry review still remains (App. A.6)."""
        return self.is_open


@dataclass(frozen=True)
class ForecastProposal:
    """A free-text forecast proposal before admission (Eq. (3), Eq. (17)).

    The forecaster emits the forecast text together with its verifying modality,
    evidence-window duration and support scope. Identifiers, intervals, lifecycle state
    and provenance are assigned at registration.
    """

    text: str
    modality: Modality
    delta_seconds: float
    scope: SupportScope = "supports_answer"
    #: Identifiers cited in the dedicated grammar-constrained citation slot.
    citations: Tuple[int, ...] = ()
    #: Whether the proposal is query-critical, required by the admission rule.
    query_critical: bool = True
    #: Forecaster confidence, used only to rank proposals under capacity pressure.
    confidence: float = 1.0

    @property
    def is_no_forecast(self) -> bool:
        return self.text.strip().upper() == NO_FORECAST

    def as_dict(self) -> Dict[str, Any]:
        return {
            "forecast": self.text,
            "modality": self.modality,
            "delta_seconds": self.delta_seconds,
            "scope": self.scope,
            "citations": list(self.citations),
        }


# --------------------------------------------------------------------------------------
# Verification records
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationResult:
    """Output of ``V_phi^{m_i}`` (Eq. (4), Eq. (7))."""

    claim_id: int
    verdict: Verdict
    #: gamma_i, the contradiction margin.
    contradiction_margin: float
    #: s_i, the raw sigmoid score.
    score: float
    modality: Modality
    interval: Interval
    #: Set when calibration clamped the score for an uninformative interval.
    capped: bool = False
    #: Set when the source became unobservable before the claim could be assessed.
    unobservable: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerdictLogEntry:
    """One appended entry of the verdict log ``V`` (App. A.3)."""

    chunk: int
    time: float
    claim_id: int
    verdict: Verdict
    contradiction_margin: float
    score: float
    is_retry: bool = False
    status_after: ClaimStatus = ClaimStatus.SETTLED


# --------------------------------------------------------------------------------------
# Traces
# --------------------------------------------------------------------------------------


@dataclass
class ChunkTrace:
    """What happened at one decision chunk. Diagnostics only; not part of the method."""

    chunk: int
    t_end: float
    verdicts: List[VerdictLogEntry] = field(default_factory=list)
    refuted_claim_ids: List[int] = field(default_factory=list)
    expired_claim_ids: List[int] = field(default_factory=list)
    admitted_claim_ids: List[int] = field(default_factory=list)
    rejected_proposals: int = 0
    #: lambda_s for this chunk (Eq. (6)).
    guidance_scale: float = 0.0
    guidance_applied: bool = False
    gate_action: Sufficiency = Sufficiency.WAIT
    gate_score: Optional[float] = None
    hard_wait: bool = False
    state_id: Optional[int] = None
    folds: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerResult:
    """Final answer plus its unresolved-support flags (Sec. 3.4, App. A.6)."""

    text: str
    normalized: Optional[str] = None
    #: Identifiers in E_s^ans: expired claims the answer still depends on.
    expired_support: List[int] = field(default_factory=list)
    #: Unresolved answer support that had not expired at answering time.
    unresolved_support: List[int] = field(default_factory=list)

    @property
    def low_confidence(self) -> bool:
        """Every answer depending on unresolved claims carries this flag."""
        return bool(self.expired_support or self.unresolved_support)


@dataclass
class EpisodeResult:
    """Result of running OST over one question and one stream."""

    answer: AnswerResult
    #: Chunk index at which the gate fired, or the terminal chunk.
    stop_chunk: int
    #: T = min(T*, T_end).
    stop_time: float
    #: Whether the gate fired before stream termination.
    stopped_early: bool
    traces: List[ChunkTrace] = field(default_factory=list)
    states: List[Any] = field(default_factory=list)
    verdict_log: List[VerdictLogEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_chunks(self) -> int:
        return len(self.traces)

    @property
    def num_claims(self) -> int:
        return sum(len(t.admitted_claim_ids) for t in self.traces)

    @property
    def num_refutations(self) -> int:
        return sum(len(t.refuted_claim_ids) for t in self.traces)

    @property
    def num_guided_chunks(self) -> int:
        return sum(1 for t in self.traces if t.guidance_applied)


def own_modalities(modality: Modality) -> Sequence[str]:
    """Modality projection support: which raw streams ``Pi_m`` exposes (App. A)."""
    if modality == "audio_visual":
        return ("audio", "visual")
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality!r}")
    return (modality,)


__all__ = [
    "AGGREGATE_AUDIO_STATES",
    "AUDIO_STATES",
    "MODALITIES",
    "NO_FORECAST",
    "OMNI_STATE_FIELDS",
    "STATE_BODY_FIELDS",
    "SUPPORT_SCOPES",
    "AggregateAudioState",
    "AnswerResult",
    "AudioState",
    "ChunkTrace",
    "ClaimRecord",
    "ClaimStatus",
    "EpisodeResult",
    "ForecastProposal",
    "GenerationStage",
    "Interval",
    "Modality",
    "SpanKind",
    "SpanRecord",
    "Sufficiency",
    "SupportScope",
    "VerdictLogEntry",
    "VerificationResult",
    "Verdict",
    "own_modalities",
    "replace",
]
