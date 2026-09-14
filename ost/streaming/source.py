"""Causal audio-visual stream sources.

A streaming model must never see past the current boundary. The sources here decode one
perception step at a time and refuse any input that would leak future evidence, because a
single episode-level audio track handed over whole would silently invalidate every timing
result.
"""

from __future__ import annotations

import abc
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

from ost.streaming.retention import PerceptionSlot

LOGGER = logging.getLogger(__name__)


class StreamError(RuntimeError):
    """Raised when a stream cannot be decoded causally."""


class StreamSource(abc.ABC):
    """One question's audio-visual stream, readable only up to the current time."""

    @property
    @abc.abstractmethod
    def duration_seconds(self) -> float:
        """``T_end``, the observed termination time."""

    @abc.abstractmethod
    def perceive(
        self, timestamp: float, *, question: str = ""
    ) -> Tuple[Optional[PerceptionSlot], Optional[PerceptionSlot]]:
        """Return the ``(visual, audio)`` slots closing at ``timestamp``.

        Either may be ``None`` when the modality carries no input at that second; the
        retention buffer keeps a presence mask rather than dropping the slot.
        """

    def window_clip(self, t_start: float, t_end: float) -> Optional[Path]:
        """One clip covering ``(t_start, t_end]``, when the source is media-backed.

        The dense context window is several seconds long, and handing it over as one clip is
        both closer to what the state context reads and necessary in practice: a single
        one-second clip contains one frame, which the multimodal preprocessor rejects.
        """
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> "StreamSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------------------
# Media-backed source
# --------------------------------------------------------------------------------------


def ffmpeg_binary() -> str:
    """Locate ``ffmpeg``.

    Resolution order is ``OST_FFMPEG`` then ``PATH``. The path is never hard-coded, since
    a cluster-specific location would make the repository unrunnable elsewhere.
    """
    override = os.environ.get("OST_FFMPEG")
    if override:
        if not Path(override).exists():
            raise StreamError(f"OST_FFMPEG points at a missing file: {override}")
        return override
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise StreamError(
        "ffmpeg was not found. Install it and put it on PATH, or set OST_FFMPEG to its "
        "absolute path."
    )


def probe_duration_seconds(media_path: str | os.PathLike[str]) -> float:
    """Container duration via ``ffprobe``."""
    path = Path(media_path)
    if not path.exists():
        raise StreamError(f"media file not found: {path}")
    ffprobe = os.environ.get("OST_FFPROBE") or shutil.which("ffprobe")
    if not ffprobe:
        raise StreamError(
            "ffprobe was not found. Install it and put it on PATH, or set OST_FFPROBE."
        )
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise StreamError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise StreamError(f"ffprobe returned no duration for {path}") from exc


class MediaStreamSource(StreamSource):
    """A stream backed by a media file, decoded one perception step at a time.

    Each call materialises only the requested window. Audio rides inside the clipped video
    segment rather than being supplied as a separate whole-episode track, which is what
    keeps the audio causal.
    """

    def __init__(
        self,
        media_path: str | os.PathLike[str],
        *,
        cache_dir: str | os.PathLike[str],
        perception_interval_seconds: float = 1.0,
        frames_per_second: float = 1.0,
        audio_sample_rate: int = 16_000,
        duration_seconds: Optional[float] = None,
        max_duration_seconds: Optional[float] = None,
    ) -> None:
        self.media_path = Path(media_path)
        if not self.media_path.exists():
            raise StreamError(f"media file not found: {self.media_path}")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.interval = float(perception_interval_seconds)
        self.frames_per_second = float(frames_per_second)
        self.audio_sample_rate = int(audio_sample_rate)

        probed = (
            float(duration_seconds)
            if duration_seconds is not None
            else probe_duration_seconds(self.media_path)
        )
        self._duration = (
            min(probed, float(max_duration_seconds))
            if max_duration_seconds is not None
            else probed
        )
        self._clip_cache: Dict[Tuple[float, float], Path] = {}

    @property
    def duration_seconds(self) -> float:
        return self._duration

    def perceive(
        self, timestamp: float, *, question: str = ""
    ) -> Tuple[Optional[PerceptionSlot], Optional[PerceptionSlot]]:
        """Record one perception step without decoding it yet.

        Decoding is deferred to :meth:`window_clip`, which materialises the whole context
        window in one pass. Extracting a clip per second would decode the same span several
        times over, and a one-second clip is not usable on its own anyway.
        """
        if timestamp > self._duration + 1e-9:
            raise StreamError(
                f"requested perception at t={timestamp} beyond T_end={self._duration}"
            )
        start = max(0.0, timestamp - self.interval)
        span = {"start": start, "end": min(timestamp, self._duration)}

        visual = PerceptionSlot(
            time=timestamp,
            present=True,
            visual_change=1.0,
            payload={"kind": "video", **span},
        )
        audio = PerceptionSlot(
            time=timestamp,
            present=True,
            audio_salience=1.0,
            co_occurrence=1.0,
            payload={"kind": "audio_in_video", **span},
        )
        return visual, audio

    def window_clip(self, t_start: float, t_end: float) -> Optional[Path]:
        start = max(0.0, float(t_start))
        end = min(float(t_end), self._duration)
        if end <= start:
            return None
        return self._extract_clip(start, end)

    def _extract_clip(self, start: float, end: float) -> Path:
        duration = max(end - start, 1.0 / max(self.frames_per_second, 1.0))
        # The key includes the span: a window clip and a per-second clip can share a start
        # time while covering different durations.
        key = (round(start, 3), round(duration, 3))
        cached = self._clip_cache.get(key)
        if cached is not None and cached.exists():
            return cached
        target = (
            self.cache_dir
            / f"{self.media_path.stem}_{key[0]:.3f}_{key[1]:.3f}.mp4"
        )
        if not target.exists():
            fps = max(self.frames_per_second, 2.0 / max(duration, 1e-6))
            command = [
                ffmpeg_binary(),
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(self.media_path),
                "-vf",
                # At least two frames: the preprocessor rejects a single-frame video.
                f"fps={fps:g}",
                # An explicit output rate writes a frame rate into the container. Without it
                # some clips carry no rate and the preprocessor cannot compute a frame budget.
                "-r",
                f"{fps:g}",
                "-vsync",
                "cfr",
                "-ar",
                str(self.audio_sample_rate),
                "-ac",
                "1",
                str(target),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0 or not target.exists():
                raise StreamError(
                    f"ffmpeg failed to extract ({start:.3f}, {end:.3f}] from "
                    f"{self.media_path.name}: {result.stderr.strip()[:400]}"
                )
        self._clip_cache[key] = target
        return target

    def close(self) -> None:
        self._clip_cache.clear()


def _lexical_overlap(question: str, text: str) -> float:
    """Cheap query-relevance proxy used by the retention scorer."""
    if not question or not text:
        return 0.0
    q_terms = {t for t in question.lower().split() if len(t) > 3}
    t_terms = {t for t in text.lower().split() if len(t) > 3}
    if not q_terms:
        return 0.0
    return len(q_terms & t_terms) / float(len(q_terms))


__all__ = [
    "MediaStreamSource",
    "StreamError",
    "StreamSource",
    "ffmpeg_binary",
    "probe_duration_seconds",
]
