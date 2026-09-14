"""The verdict-reliability attention bias.

Retraction is realised at decoding time as an additive log-reliability bias on the
reasoning-token keys. Because the bias is additive in log space, a span with effective
reliability ``rho_bar`` contributes exactly a factor ``rho_bar`` to its post-softmax
attention mass, which is what makes "reduce the influence of this reasoning" a
well-defined operation rather than a heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

#: Effective reliabilities below this floor are treated as this floor, so the bias stays
#: finite. ``rho_min`` is 0.05 by default, well above it.
MIN_RELIABILITY = 1e-6


@dataclass(frozen=True)
class SpanTokenRange:
    """A registered span's token range in the serialised reasoning context."""

    span_id: int
    start: int
    end: int
    effective_reliability: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"span {self.span_id} has end {self.end} < start {self.start}")

    @property
    def length(self) -> int:
        return self.end - self.start


def reliability_bias_vector(
    kv_len: int,
    span_ranges: Sequence[SpanTokenRange],
    *,
    reasoning_range: Optional[Tuple[int, int]] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Per-key log-reliability bias of length ``kv_len``.

    Implements the key-indexed part of Eq. (12)::

        B_ver[:, k] = log rho_bar_j   for k in span(g_j)
        B_ver[:, k] = 0               otherwise

    Overlapping spans take the minimum applicable effective weight, so a token covered by
    both a reliable and an attenuated span is attenuated. Keys outside
    ``reasoning_range`` are left at zero: App. A.4 leaves perceptual attention unchanged.
    """
    if kv_len < 0:
        raise ValueError("kv_len must be non-negative")
    bias = torch.zeros(kv_len, device=device, dtype=dtype)
    if kv_len == 0:
        return bias

    lo, hi = reasoning_range if reasoning_range is not None else (0, kv_len)
    lo = max(0, int(lo))
    hi = min(kv_len, int(hi))
    if hi <= lo:
        return bias

    # Track the minimum reliability per key so overlaps resolve to the minimum.
    min_reliability = torch.ones(kv_len, device=device, dtype=dtype)
    touched = torch.zeros(kv_len, device=device, dtype=torch.bool)

    for span in span_ranges:
        start = max(lo, int(span.start))
        end = min(hi, int(span.end))
        if end <= start:
            continue
        value = max(float(span.effective_reliability), MIN_RELIABILITY)
        window = slice(start, end)
        candidate = torch.full(
            (end - start,), value, device=device, dtype=dtype
        )
        min_reliability[window] = torch.minimum(min_reliability[window], candidate)
        touched[window] = True

    if touched.any():
        bias[touched] = torch.log(min_reliability[touched])
    return bias


def build_verdict_bias(
    q_len: int,
    kv_len: int,
    span_ranges: Sequence[SpanTokenRange],
    *,
    reasoning_range: Optional[Tuple[int, int]] = None,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    causal_offset: Optional[int] = None,
) -> torch.Tensor:
    """Additive attention mask of shape ``(batch, 1, q_len, kv_len)``.

    Implements Eq. (12). The bias is constant along the query axis, since it depends only
    on which span a *key* belongs to. A causal mask is included when ``causal_offset`` is
    given, so the returned tensor can be passed directly as the single additive mask that
    the ``sdpa`` and ``eager`` attention backends accept. Fused kernels such as
    ``flash_attention_2`` reject a floating-point 4-D mask, which is why the runtime
    pins the backend.
    """
    key_bias = reliability_bias_vector(
        kv_len,
        span_ranges,
        reasoning_range=reasoning_range,
        device=device,
        dtype=dtype,
    )
    mask = key_bias.view(1, 1, 1, kv_len).expand(batch_size, 1, q_len, kv_len).clone()

    if causal_offset is not None:
        causal = causal_mask(q_len, kv_len, causal_offset, device=device, dtype=dtype)
        mask = mask + causal
    return mask


def causal_mask(
    q_len: int,
    kv_len: int,
    offset: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Additive causal mask ``B_causal`` of shape ``(1, 1, q_len, kv_len)``.

    ``offset`` is the number of cached keys preceding the current query block, so
    query ``i`` may attend to keys ``0 .. offset + i``.
    """
    q_idx = torch.arange(q_len, device=device).view(q_len, 1) + int(offset)
    k_idx = torch.arange(kv_len, device=device).view(1, kv_len)
    blocked = k_idx > q_idx
    mask = torch.zeros(q_len, kv_len, device=device, dtype=dtype)
    mask.masked_fill_(blocked, torch.finfo(dtype).min)
    return mask.view(1, 1, q_len, kv_len)


def effective_attention_scale(bias: torch.Tensor) -> torch.Tensor:
    """Recover ``rho_bar`` per key from a log-space bias, for inspecting a built mask."""
    return torch.exp(bias)


def span_ranges_from_offsets(
    offsets: Mapping[int, Tuple[int, int]],
    effective: Mapping[int, float],
) -> List[SpanTokenRange]:
    """Zip serialisation offsets with effective reliabilities."""
    ranges: List[SpanTokenRange] = []
    for span_id, (start, end) in offsets.items():
        ranges.append(
            SpanTokenRange(
                span_id=int(span_id),
                start=int(start),
                end=int(end),
                effective_reliability=float(effective.get(span_id, 1.0)),
            )
        )
    return sorted(ranges, key=lambda r: (r.start, r.span_id))


def has_attenuation(span_ranges: Iterable[SpanTokenRange], tolerance: float = 1e-9) -> bool:
    """Whether any span would actually attenuate attention."""
    return any(r.effective_reliability < 1.0 - tolerance for r in span_ranges)


__all__ = [
    "MIN_RELIABILITY",
    "SpanTokenRange",
    "build_verdict_bias",
    "causal_mask",
    "effective_attention_scale",
    "has_attenuation",
    "reliability_bias_vector",
    "span_ranges_from_offsets",
]
