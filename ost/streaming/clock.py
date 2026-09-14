"""The decision grid.

Perception runs at a fixed interval and decisions are taken on a coarser grid. All
claim deadlines and reviews follow this grid, so it lives in one place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, List, Optional

from ost.types import Interval

_EPS = 1e-9


@dataclass(frozen=True)
class DecisionChunk:
    """One decision chunk ``s`` with closing time ``t_s``."""

    index: int
    t_start: float
    t_end: float
    is_terminal: bool

    @property
    def interval(self) -> Interval:
        return Interval(self.t_start, self.t_end)

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


class DecisionClock:
    """Chunk grid of Eq. (1).

    Implements Eq. (1): ``t_s = min(s * decision_interval, T_end)``, so the final chunk
    may be shorter than the nominal interval. Perception timestamps are the finer grid
    the retention buffers are indexed on.
    """

    def __init__(
        self,
        decision_interval_seconds: float,
        perception_interval_seconds: float = 1.0,
        end_time: Optional[float] = None,
    ) -> None:
        if decision_interval_seconds <= 0:
            raise ValueError("decision_interval_seconds must be positive")
        if perception_interval_seconds <= 0:
            raise ValueError("perception_interval_seconds must be positive")
        self.decision_interval = float(decision_interval_seconds)
        self.perception_interval = float(perception_interval_seconds)
        self._end_time = float(end_time) if end_time is not None else None

    # -- basic accessors ----------------------------------------------------------

    @property
    def end_time(self) -> Optional[float]:
        """``T_end``, the observed termination time, once it is known."""
        return self._end_time

    def with_end_time(self, end_time: float) -> "DecisionClock":
        return DecisionClock(self.decision_interval, self.perception_interval, end_time)

    def boundary(self, index: int) -> float:
        """``t_s`` for chunk ``index`` (1-based, matching the paper)."""
        if index < 1:
            raise ValueError("decision chunk indices start at 1")
        nominal = index * self.decision_interval
        if self._end_time is None:
            return nominal
        return min(nominal, self._end_time)

    def num_chunks(self) -> int:
        """Number of decision chunks in the stream. Requires a known ``T_end``."""
        if self._end_time is None:
            raise ValueError("num_chunks requires a known end_time")
        if self._end_time <= _EPS:
            return 1
        return max(1, math.ceil(self._end_time / self.decision_interval - _EPS))

    # -- iteration ----------------------------------------------------------------

    def chunks(self) -> Iterator[DecisionChunk]:
        """Yield every decision chunk, including the terminal one."""
        total = self.num_chunks()
        for index in range(1, total + 1):
            t_end = self.boundary(index)
            t_start = self.boundary(index - 1) if index > 1 else 0.0
            yield DecisionChunk(
                index=index,
                t_start=t_start,
                t_end=t_end,
                is_terminal=index == total,
            )

    def perception_timestamps(self, chunk: DecisionChunk) -> List[float]:
        """Perception timestamps falling inside ``(t_start, t_end]``."""
        stamps: List[float] = []
        step = self.perception_interval
        first = math.floor(chunk.t_start / step + _EPS) + 1
        t = first * step
        while t <= chunk.t_end + _EPS:
            stamps.append(round(min(t, chunk.t_end), 6))
            t += step
        if not stamps:
            stamps.append(round(chunk.t_end, 6))
        return stamps

    # -- claim scheduling ---------------------------------------------------------

    def review_chunk(self, issue_time: float, delta_seconds: float) -> int:
        """``d_i = ceil((t_i + delta_i) / Delta_chunk)``.

        Implements the App. A.2 review schedule: a claim is reviewed at the first chunk
        boundary at which its evidence window is complete.
        """
        if delta_seconds <= 0:
            raise ValueError("delta_seconds must be positive")
        due = issue_time + delta_seconds
        return max(1, math.ceil(due / self.decision_interval - _EPS))

    def retry_chunk(self, review_chunk: int, retry_slack_seconds: float) -> int:
        """``d_i + ceil(delta' / Delta_chunk)`` (App. A.3)."""
        if retry_slack_seconds < 0:
            raise ValueError("retry_slack_seconds must be non-negative")
        offset = math.ceil(retry_slack_seconds / self.decision_interval - _EPS)
        return review_chunk + max(offset, 0)

    def chunk_containing(self, t: float) -> int:
        """Index of the chunk whose half-open interval contains ``t``."""
        if t <= _EPS:
            return 1
        return max(1, math.ceil(t / self.decision_interval - _EPS))


__all__ = ["DecisionChunk", "DecisionClock"]
