"""The Omni-State memory pyramid.

Long-term reasoning uses a four-way pyramid: base states are merged into progressively
coarser summaries and finally into a long-term root. Folding is a memory operation, not
a reasoning one, so only the two evidence fields are re-summarised by the backbone;
everything that carries verification semantics is propagated by fixed rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ost.config import PyramidConfig
from ost.state.omni_state import NO_CONFLICT, UNCERTAIN, OmniState, StateBody, StateLevel
from ost.types import Interval, Sufficiency

LOGGER = logging.getLogger(__name__)

#: Level names in ascending coarseness. ``Linf`` is the long-term memory root.
LEVELS: Tuple[StateLevel, ...] = ("L1", "L2", "L3", "Linf")

#: Callback that summarises the evidence fields of a set of children.
#: Receives the children in chronological order, returns (visual_evidence,
#: audio_evidence). Supplied by the backbone; a deterministic fallback is used when the
#: backbone is unavailable.
EvidenceSummarizer = Callable[[Sequence[OmniState]], Tuple[str, str]]


def aggregate_audio_state(children: Sequence[OmniState]) -> str:
    """Aggregate ``audio_state`` over a union interval (App. A.1).

    The aggregate vocabulary is ``all_absent``, ``some_present``, ``mixed`` and
    ``uncertain``. ``mixed`` is used when presence and absence are both determined and
    disagree; ``uncertain`` propagates whenever any child is undetermined and the rest
    do not already establish presence.
    """
    if not children:
        return "uncertain"
    values = [child.body.audio_state for child in children]
    present = any(v in ("present", "some_present", "mixed") for v in values)
    absent = any(v in ("absent", "all_absent") for v in values)
    uncertain = any(v == "uncertain" for v in values)

    if present and absent:
        return "mixed"
    if present:
        # A determined presence is not weakened by an undetermined sibling.
        return "some_present"
    if absent and not uncertain:
        return "all_absent"
    return "uncertain"


def merge_conflict_text(children: Sequence[OmniState]) -> str:
    """Carry conflict text forward by rule, preserving order and dropping duplicates."""
    seen: List[str] = []
    for child in children:
        text = (child.body.conflict or "").strip()
        if not text or text == NO_CONFLICT:
            continue
        if text not in seen:
            seen.append(text)
    return " ; ".join(seen) if seen else NO_CONFLICT


def max_sufficiency(children: Sequence[OmniState]) -> Sufficiency:
    """Folded ``sufficiency`` is the maximum over children (App. A.1)."""
    for child in children:
        if child.sufficiency is Sufficiency.ANSWER:
            return Sufficiency.ANSWER
    return Sufficiency.WAIT


def _fallback_summarizer(children: Sequence[OmniState]) -> Tuple[str, str]:
    """Deterministic evidence concatenation used when no backbone is supplied."""

    def _join(values: Iterable[str]) -> str:
        parts: List[str] = []
        for value in values:
            text = (value or "").strip()
            if text and text != UNCERTAIN and text not in parts:
                parts.append(text)
        return " ".join(parts)

    visual = _join(child.body.visual_evidence for child in children)
    audio = _join(child.body.audio_evidence for child in children)
    return visual, audio


@dataclass
class FoldEvent:
    """Record of one merge, for tracing and for the archive."""

    level: StateLevel
    target_level: StateLevel
    child_state_ids: List[int]
    summary_state_id: int
    interval: Interval
    inherited_claim_ids: List[int]


@dataclass
class Pyramid:
    """Four-way Omni-State pyramid with capacities ``(K1, K2, K3)`` and a root.

    Level 1 holds base states covering one decision interval each. Each higher level
    holds summaries covering ``branching`` times the span below it. When a level exceeds
    its capacity, its oldest ``branching`` records are merged into one record on the next
    level; overflow at the top level merges into the long-term root.
    """

    config: PyramidConfig
    decision_interval_seconds: float
    summarizer: Optional[EvidenceSummarizer] = None
    #: Assigns identifiers to newly created summary states.
    id_allocator: Optional[Callable[[], int]] = None
    #: App. A.1's append-only archive. Disable to bound memory on a very long stream, at the
    #: cost of losing the record of folded states.
    keep_archive: bool = True

    levels: Dict[StateLevel, List[OmniState]] = field(default_factory=dict)
    archive: List[OmniState] = field(default_factory=list)
    fold_events: List[FoldEvent] = field(default_factory=list)
    _next_id: int = 10_000

    def __post_init__(self) -> None:
        self.config.validate()
        if not self.levels:
            self.levels = {name: [] for name in LEVELS}

    # -- capacities ---------------------------------------------------------------

    def capacity(self, level: StateLevel) -> int:
        if level == "Linf":
            return self.config.root_capacity
        index = LEVELS.index(level)
        return self.config.capacities[index]

    def span_seconds(self, level: StateLevel) -> float:
        """Nominal interval covered by one record at ``level``."""
        if level == "Linf":
            return float("inf")
        spans = self.config.level_span_seconds(self.decision_interval_seconds)
        return spans[LEVELS.index(level)]

    # -- insertion and folding ----------------------------------------------------

    def append(self, state: OmniState) -> List[FoldEvent]:
        """Append a completed base state and fold any overflowing level."""
        if not state.is_complete:
            raise ValueError(
                "only completed Omni-States enter the pyramid; sufficiency is unset"
            )
        self.levels["L1"].append(state)
        self._archive(state)
        return self.fold_overflow()

    def fold_overflow(self) -> List[FoldEvent]:
        """Merge every level that exceeds its capacity, lowest level first."""
        events: List[FoldEvent] = []
        for index, level in enumerate(LEVELS[:-1]):
            target = LEVELS[index + 1]
            while len(self.levels[level]) > self.capacity(level):
                event = self._fold_once(level, target)
                if event is None:
                    break
                events.append(event)
        self._trim_root()
        self.fold_events.extend(events)
        return events

    def _fold_once(self, level: StateLevel, target: StateLevel) -> Optional[FoldEvent]:
        n = self.config.branching
        records = self.levels[level]
        if len(records) < n:
            # Not enough records to form a summary; the level stays over capacity until
            # more arrive. Guard against an infinite loop in fold_overflow.
            LOGGER.debug(
                "level %s over capacity with %d records but branching is %d",
                level,
                len(records),
                n,
            )
            return None
        children = records[:n]
        del records[:n]
        summary = self.fold(children, target)
        self.levels[target].append(summary)
        self._archive(summary)
        return FoldEvent(
            level=level,
            target_level=target,
            child_state_ids=[c.state_id for c in children],
            summary_state_id=summary.state_id,
            interval=summary.interval,
            inherited_claim_ids=list(summary.claim_ids),
        )

    def fold(self, children: Sequence[OmniState], level: StateLevel) -> OmniState:
        """Merge ``children`` into one summary state at ``level``.

        Implements the App. A.1 fold rule. The backbone summarises the two evidence
        fields from the children's content; fixed rules carry forward the conflict text,
        claim identifiers, intervals, verdicts and reliability scores. The folded state
        inherits the children's claim ancestors, so a compressed account of a refuted
        assumption retains its reduced reliability score.
        """
        if not children:
            raise ValueError("cannot fold an empty set of children")
        ordered = sorted(children, key=lambda s: s.interval.start)
        interval = ordered[0].interval
        for child in ordered[1:]:
            interval = interval.union(child.interval)

        summarizer = self.summarizer or _fallback_summarizer
        visual_evidence, audio_evidence = summarizer(ordered)

        audio_state = aggregate_audio_state(ordered)
        if audio_state not in ("all_absent",) and not (audio_evidence or "").strip():
            # Keep the App. A.2 contract: evidence text is required unless absent.
            audio_evidence = UNCERTAIN

        body = StateBody(
            visual_evidence=(visual_evidence or "").strip() or UNCERTAIN,
            audio_state=audio_state,
            audio_evidence=(audio_evidence or "").strip(),
            conflict=merge_conflict_text(ordered),
        )

        claim_ids: List[int] = []
        for child in ordered:
            for claim_id in child.claim_ids:
                if claim_id not in claim_ids:
                    claim_ids.append(claim_id)

        forecast_parts = [c.forecast.strip() for c in ordered if c.forecast.strip()]

        summary = OmniState(
            state_id=self._allocate_id(),
            chunk=ordered[-1].chunk,
            interval=interval,
            body=body,
            level=level,
            forecast=" | ".join(forecast_parts),
            claim_ids=claim_ids,
            child_state_ids=[c.state_id for c in ordered],
        )
        summary.metadata["fold_of"] = [c.state_id for c in ordered]
        summary.metadata["fold_level"] = level
        return summary.finalize(max_sufficiency(ordered))

    def _trim_root(self) -> None:
        root = self.levels["Linf"]
        capacity = self.capacity("Linf")
        while len(root) > capacity:
            children = root[: self.config.branching] if len(root) >= self.config.branching else list(root)
            del root[: len(children)]
            merged = self.fold(children, "Linf")
            root.insert(0, merged)
            self._archive(merged)

    # -- views ---------------------------------------------------------------------

    def active_states(self) -> List[OmniState]:
        """Records in the active reasoning ledger, oldest first."""
        out: List[OmniState] = []
        for level in LEVELS:
            out.extend(self.levels[level])
        return sorted(out, key=lambda s: (s.interval.start, s.state_id))

    def base_states(self) -> List[OmniState]:
        return list(self.levels["L1"])

    def state_by_id(self, state_id: int) -> Optional[OmniState]:
        for state in self.active_states():
            if state.state_id == state_id:
                return state
        for state in self.archive:
            if state.state_id == state_id:
                return state
        return None

    def holds_chunk(self, chunk: int) -> bool:
        """Whether a base state for ``chunk`` is still at level 1."""
        return any(state.chunk == chunk for state in self.levels["L1"])

    # -- internals -----------------------------------------------------------------

    def _allocate_id(self) -> int:
        if self.id_allocator is not None:
            return self.id_allocator()
        self._next_id += 1
        return self._next_id

    def _archive(self, state: OmniState) -> None:
        if self.keep_archive:
            self.archive.append(state)


__all__ = [
    "LEVELS",
    "EvidenceSummarizer",
    "FoldEvent",
    "Pyramid",
    "aggregate_audio_state",
    "max_sufficiency",
    "merge_conflict_text",
]
