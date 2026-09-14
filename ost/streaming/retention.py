"""Separate audio and visual retention.

A claim can only be checked if its evidence is still available when the interval closes.
Under one shared budget, dense visual tokens can displace the audio a pending claim needs,
so each modality gets its own budget and evicts inside it. Intervals awaiting verification
are pinned and are never evicted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ost.config import RetentionConfig, RetentionScoreWeights
from ost.types import Interval, Modality

LOGGER = logging.getLogger(__name__)


@dataclass
class PerceptionSlot:
    """One second of perception on the absolute timeline."""

    #: Closing time of the slot; the slot covers ``(time - interval, time]``.
    time: float
    #: Token cost of the slot inside its modality budget.
    token_cost: int = 1
    #: Whether the modality actually carried input here. App. A.1 keeps presence masks
    #: for missing inputs rather than dropping the slot.
    present: bool = True
    #: Component scores; combined into ``score`` by :func:`retention_score`.
    visual_change: float = 0.0
    audio_salience: float = 0.0
    query_relevance: float = 0.0
    co_occurrence: float = 0.0
    #: Opaque per-slot payload (frame path, waveform slice, features).
    payload: object = None
    #: Number of pins currently held on this slot.
    pins: int = 0

    @property
    def pinned(self) -> bool:
        return self.pins > 0

    def score(self, weights: RetentionScoreWeights) -> float:
        return retention_score(self, weights)


def retention_score(slot: PerceptionSlot, weights: RetentionScoreWeights) -> float:
    """Combine the four retention-score components (App. A.1).

    The components are visual change, audio salience, query relevance and audio-visual
    co-occurrence. Their weighting is configurable; equal weights are the default.
    """
    w_v, w_a, w_q, w_c = weights.as_tuple()
    return (
        w_v * slot.visual_change
        + w_a * slot.audio_salience
        + w_q * slot.query_relevance
        + w_c * slot.co_occurrence
    )


class ModalityBudget:
    """A single-modality perceptual buffer with its own eviction.

    Eviction never crosses modalities, never touches pinned slots, and never touches the
    dense floor window, which App. A.1 requires to be retained at full density.
    """

    def __init__(
        self,
        modality: str,
        token_budget: int,
        floor_seconds: float,
        weights: RetentionScoreWeights,
    ) -> None:
        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        self.modality = modality
        self.token_budget = int(token_budget)
        self.floor_seconds = float(floor_seconds)
        self.weights = weights
        self._slots: Dict[float, PerceptionSlot] = {}
        self.evicted_count = 0

    # -- state --------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._slots)

    @property
    def tokens_used(self) -> int:
        return sum(slot.token_cost for slot in self._slots.values())

    def slots(self) -> List[PerceptionSlot]:
        return [self._slots[t] for t in sorted(self._slots)]

    def times(self) -> List[float]:
        return sorted(self._slots)

    def get(self, time: float) -> Optional[PerceptionSlot]:
        return self._slots.get(_key(time))

    # -- ingestion ----------------------------------------------------------------

    def ingest(self, slot: PerceptionSlot) -> None:
        self._slots[_key(slot.time)] = slot

    def evict(self, now: float) -> List[float]:
        """Evict lowest-scoring evictable slots until the budget is respected."""
        evicted: List[float] = []
        while self.tokens_used > self.token_budget:
            candidates = [
                slot
                for slot in self._slots.values()
                if not slot.pinned and slot.time <= now - self.floor_seconds + 1e-9
            ]
            if not candidates:
                # Everything left is pinned or inside the dense floor. App. A.1 makes
                # both inviolable, so the budget is exceeded rather than silently
                # dropping evidence a pending claim needs.
                LOGGER.debug(
                    "%s budget over capacity (%d/%d) with no evictable slots",
                    self.modality,
                    self.tokens_used,
                    self.token_budget,
                )
                break
            victim = min(candidates, key=lambda s: (s.score(self.weights), s.time))
            del self._slots[_key(victim.time)]
            evicted.append(victim.time)
            self.evicted_count += 1
        return evicted

    # -- pinning -------------------------------------------------------------------

    def pin_interval(self, interval: Interval) -> int:
        count = 0
        for slot in self._slots.values():
            if interval.contains(slot.time):
                slot.pins += 1
                count += 1
        return count

    def release_interval(self, interval: Interval) -> int:
        count = 0
        for slot in self._slots.values():
            if interval.contains(slot.time) and slot.pins > 0:
                slot.pins -= 1
                count += 1
        return count

    # -- reads ---------------------------------------------------------------------

    def window(self, interval: Interval, *, present_only: bool = False) -> List[PerceptionSlot]:
        out = [
            self._slots[t]
            for t in sorted(self._slots)
            if interval.contains(self._slots[t].time)
        ]
        if present_only:
            out = [slot for slot in out if slot.present]
        return out

    def dense_window(self, now: float, seconds: float) -> List[PerceptionSlot]:
        return self.window(Interval(max(0.0, now - seconds), now))

    def sparse_retrieval(
        self, now: float, dense_seconds: float, slots: int
    ) -> List[PerceptionSlot]:
        """Top-scoring slots outside the dense window (App. A.1)."""
        if slots <= 0:
            return []
        cutoff = now - dense_seconds
        older = [slot for slot in self._slots.values() if slot.time <= cutoff + 1e-9]
        older.sort(key=lambda s: (-s.score(self.weights), s.time))
        return sorted(older[:slots], key=lambda s: s.time)

    def coverage(self, interval: Interval) -> float:
        """Fraction of the interval's perception slots still retained."""
        expected = self.window(interval)
        if not expected:
            return 0.0
        return sum(1 for slot in expected if slot.present) / float(len(expected))


class RetentionBuffers:
    """The audio and visual buffers plus the pin bookkeeping.

    Audio and visual features share an absolute timeline; the two buffers are indexed on
    the same perception timestamps but sized and evicted independently.
    """

    def __init__(
        self,
        config: RetentionConfig,
        decision_interval_seconds: float,
        perception_interval_seconds: float = 1.0,
    ) -> None:
        config.validate()
        self.config = config
        self.decision_interval_seconds = float(decision_interval_seconds)
        self.perception_interval_seconds = float(perception_interval_seconds)
        self.visual = ModalityBudget(
            "visual", config.visual_tokens, config.floor_seconds, config.score_weights
        )
        self.audio = ModalityBudget(
            "audio", config.audio_tokens, config.floor_seconds, config.score_weights
        )
        #: Pinned intervals keyed by claim identifier.
        self._pins: Dict[int, Tuple[Interval, Tuple[str, ...]]] = {}
        self.now: float = 0.0

    # -- properties ----------------------------------------------------------------

    @property
    def dense_floor_seconds(self) -> float:
        """The retention floor both perceptual buffers keep at full density."""
        return self.config.floor_seconds

    @property
    def pinned_claim_ids(self) -> Set[int]:
        return set(self._pins)

    def budget(self, modality: str) -> ModalityBudget:
        if modality == "audio":
            return self.audio
        if modality == "visual":
            return self.visual
        raise ValueError(f"no single budget for modality {modality!r}")

    # -- ingestion -----------------------------------------------------------------

    def ingest(
        self,
        time: float,
        visual: Optional[PerceptionSlot] = None,
        audio: Optional[PerceptionSlot] = None,
    ) -> None:
        """Ingest one perception timestamp into both buffers."""
        self.now = max(self.now, float(time))
        if visual is not None:
            self.visual.ingest(visual)
        if audio is not None:
            self.audio.ingest(audio)
        # Newly ingested slots may fall inside an already-pinned interval.
        for interval, modalities in self._pins.values():
            if not interval.contains(time):
                continue
            for name in modalities:
                slot = self.budget(name).get(time)
                if slot is not None and slot.pins == 0:
                    slot.pins += 1

    def evict(self) -> Dict[str, List[float]]:
        """Run eviction inside each modality budget."""
        return {
            "visual": self.visual.evict(self.now),
            "audio": self.audio.evict(self.now),
        }

    # -- pinning -------------------------------------------------------------------

    def pin(self, claim_id: int, interval: Interval, modality: Modality) -> None:
        """Pin a pending evidence window until its claim settles or expires."""
        if claim_id in self._pins:
            return
        modalities = ("audio", "visual") if modality == "audio_visual" else (modality,)
        self._pins[claim_id] = (interval, modalities)
        for name in modalities:
            self.budget(name).pin_interval(interval)

    def release(self, claim_id: int) -> bool:
        """Release a claim's pin. Called on settlement and on expiry."""
        entry = self._pins.pop(claim_id, None)
        if entry is None:
            return False
        interval, modalities = entry
        for name in modalities:
            self.budget(name).release_interval(interval)
        return True

    def release_all(self) -> List[int]:
        released = sorted(self._pins)
        for claim_id in released:
            self.release(claim_id)
        return released

    # -- reads ---------------------------------------------------------------------

    def evidence_window(self, interval: Interval, modality: Modality) -> Dict[str, List[PerceptionSlot]]:
        """Retained evidence for a modality over an interval.

        Implements the modality projection of App. A: audio claims read audio, visual
        claims read video, relational claims read the pair.
        """
        if modality == "audio_visual":
            return {
                "audio": self.audio.window(interval),
                "visual": self.visual.window(interval),
            }
        return {modality: self.budget(modality).window(interval)}

    def context_window(self, *, for_answer: bool = False) -> Dict[str, List[PerceptionSlot]]:
        """Dense window plus sparse query-conditioned retrieval (App. A.1)."""
        dense = (
            self.config.answer_window_seconds
            if for_answer
            else self.config.state_window_seconds
        )
        out: Dict[str, List[PerceptionSlot]] = {}
        for name in ("visual", "audio"):
            budget = self.budget(name)
            slots = budget.dense_window(self.now, dense)
            sparse = budget.sparse_retrieval(
                self.now, dense, self.config.sparse_retrieval_slots
            )
            merged = {slot.time: slot for slot in sparse}
            merged.update({slot.time: slot for slot in slots})
            out[name] = [merged[t] for t in sorted(merged)]
        return out

    def stats(self) -> Dict[str, object]:
        return {
            "visual_slots": len(self.visual),
            "audio_slots": len(self.audio),
            "visual_tokens": self.visual.tokens_used,
            "audio_tokens": self.audio.tokens_used,
            "visual_evicted": self.visual.evicted_count,
            "audio_evicted": self.audio.evicted_count,
            "pinned_claims": len(self._pins),
            "now": self.now,
        }


def _key(time: float) -> float:
    return round(float(time), 6)


__all__ = [
    "ModalityBudget",
    "PerceptionSlot",
    "RetentionBuffers",
    "retention_score",
]
