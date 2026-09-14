"""Output schemas and parsers for the state, forecast and gate lanes.

The model emits fixed field slots and places supporting identifiers in a dedicated slot
(App. A.4). This module owns the JSON schemas used for grammar-constrained decoding, the
text renderers, and the parsers that turn generated text back into typed objects. The
renderers and the parsers live together so that the surface form the loop reads back is
the one it asked for.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ost.state.omni_state import NO_CONFLICT, UNCERTAIN, StateBody
from ost.types import (
    AUDIO_STATES,
    MODALITIES,
    NO_FORECAST,
    SUPPORT_SCOPES,
    ForecastProposal,
    SupportScope,
)


class SchemaError(ValueError):
    """Raised when generated text does not satisfy the lane schema."""


# --------------------------------------------------------------------------------------
# JSON schemas for grammar-constrained decoding
# --------------------------------------------------------------------------------------


def state_body_schema() -> Dict[str, Any]:
    """Schema for the rewrite lane: the four state-body fields plus a citation slot."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["visual_evidence", "audio_state", "audio_evidence", "conflict"],
        "properties": {
            "visual_evidence": {"type": "string", "maxLength": 400},
            "audio_state": {"type": "string", "enum": list(AUDIO_STATES)},
            "audio_evidence": {"type": "string", "maxLength": 400},
            "conflict": {"type": "string", "maxLength": 300},
            # Dedicated grammar-constrained slot for supporting identifiers (App. A.4).
            "supports": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "maxItems": 8,
            },
        },
    }


def forecast_schema(window_durations: Sequence[float], max_items: int) -> Dict[str, Any]:
    """Schema for the forecast lane (Eq. (3), Eq. (17)).

    ``x_i`` stays free text; the envelope pins the verifying modality, the evidence
    window duration drawn from the fixed set, and the support scope.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["forecasts"],
        "properties": {
            "forecasts": {
                "type": "array",
                "maxItems": max_items,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["forecast", "modality", "delta_seconds", "scope"],
                    "properties": {
                        "forecast": {"type": "string", "maxLength": 300},
                        "modality": {"type": "string", "enum": list(MODALITIES)},
                        "delta_seconds": {
                            "type": "number",
                            "enum": [float(d) for d in window_durations],
                        },
                        "scope": {"type": "string", "enum": list(SUPPORT_SCOPES)},
                        "supports": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0},
                            "maxItems": 4,
                        },
                    },
                },
            }
        },
    }


# --------------------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------------------


def render_forecast_field(proposals: Sequence[ForecastProposal]) -> str:
    """Render the ``forecast`` Omni-State field.

    App. A.5: ``NO_FORECAST`` is serialised as an empty field.
    """
    live = [p for p in proposals if not p.is_no_forecast]
    if not live:
        return ""
    return " | ".join(
        f"{p.text} [{p.modality}, +{p.delta_seconds:g}s, {p.scope}]" for p in live
    )


# --------------------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------------------

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

# The schema names are the contract, but an unconstrained lane reaches for near-synonyms.
# Accepting them costs nothing and turns an unusable lane into a usable one when grammar is
# unavailable; grammar-constrained decoding still emits the canonical names.
_FORECAST_TEXT_KEYS = ("forecast", "prediction", "text", "claim", "proposition")
_MODALITY_KEYS = ("modality", "verifying_modality", "channel")
_WINDOW_KEYS = ("delta_seconds", "window_duration", "delta", "window_seconds", "deadline")
_SCOPE_KEYS = ("scope", "support_scope", "supports")


def _first(item: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """First present, non-empty value among ``keys``."""
    for key in keys:
        if key in item:
            value = item[key]
            if value is not None and value != "" and value != []:
                return value
    return None


def _load_json_object(text: str, lane: str) -> Dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise SchemaError(f"{lane} lane produced empty output")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT.search(stripped)
    if match is None:
        raise SchemaError(
            f"{lane} lane produced no JSON object; first 200 chars: {stripped[:200]!r}"
        )
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise SchemaError(
            f"{lane} lane produced malformed JSON: {exc}; "
            f"first 200 chars: {stripped[:200]!r}"
        ) from exc


def parse_state_body(text: str) -> Tuple[StateBody, List[int]]:
    """Parse the rewrite lane output into a state body and its cited identifiers.

    Missing optional text falls back to ``uncertain`` rather than raising, because the
    paper's contract is that a field lacking usable support is uncertain. A missing or
    invalid ``audio_state`` is likewise treated as ``uncertain``.
    """
    payload = _load_json_object(text, "state")
    if not isinstance(payload, dict):
        raise SchemaError("state lane output is not a JSON object")

    audio_state = str(payload.get("audio_state", UNCERTAIN)).strip().lower()
    if audio_state not in AUDIO_STATES:
        audio_state = UNCERTAIN

    body = StateBody(
        visual_evidence=str(payload.get("visual_evidence", "") or "").strip(),
        audio_state=audio_state,
        audio_evidence=str(payload.get("audio_evidence", "") or "").strip(),
        conflict=str(payload.get("conflict", NO_CONFLICT) or NO_CONFLICT).strip(),
    )
    # App. A.2: audio_evidence is required unless audio_state is absent. When the model
    # claims presence without evidence, the field's support is unusable, so the state
    # becomes uncertain rather than silently asserting an unsupported presence.
    if body.audio_state == "present" and not body.audio_evidence:
        body.audio_state = UNCERTAIN
    if body.audio_state == UNCERTAIN and not body.audio_evidence:
        body.audio_evidence = UNCERTAIN
    if not body.conflict:
        body.conflict = NO_CONFLICT

    supports = _parse_int_list(payload.get("supports"))
    return body.validate(), supports


def parse_forecasts(
    text: str,
    window_durations: Sequence[float],
    *,
    default_scope: SupportScope = "supports_answer",
) -> List[ForecastProposal]:
    """Parse the forecast lane output into proposals.

    Unparsable or out-of-schema entries are dropped rather than repaired: the admission
    rule of App. A.2 requires a verifying modality and a strictly future interval, and a
    malformed entry cannot supply them.
    """
    stripped = text.strip()
    if not stripped or stripped.upper() == NO_FORECAST:
        return []

    payload = _load_json_object(text, "forecast")
    raw_items = payload.get("forecasts", [])
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        raise SchemaError("forecast lane 'forecasts' must be a list")

    allowed = {round(float(d), 6) for d in window_durations}
    proposals: List[ForecastProposal] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        forecast_text = str(_first(item, _FORECAST_TEXT_KEYS) or "").strip()
        if not forecast_text or forecast_text.upper() == NO_FORECAST:
            continue
        modality = str(_first(item, _MODALITY_KEYS) or "").strip().lower()
        if modality not in MODALITIES:
            continue
        try:
            delta = round(float(_first(item, _WINDOW_KEYS)), 6)
        except (TypeError, ValueError):
            continue
        if delta not in allowed:
            continue
        scope = str(_first(item, _SCOPE_KEYS) or default_scope).strip().lower()
        if scope not in SUPPORT_SCOPES:
            scope = default_scope
        proposals.append(
            ForecastProposal(
                text=forecast_text,
                modality=modality,  # type: ignore[arg-type]
                delta_seconds=delta,
                scope=scope,  # type: ignore[arg-type]
                citations=tuple(
                _parse_int_list(item.get("supports"))
                if not isinstance(item.get("supports"), str)
                else item.get("cites")
            ),
                query_critical=bool(item.get("query_critical", True)),
                confidence=_safe_float(item.get("confidence"), 1.0),
            )
        )
    return proposals


def _parse_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [int(value)]
    if isinstance(value, str):
        return [int(tok) for tok in re.findall(r"\d+", value)]
    if isinstance(value, Iterable):
        out: List[int] = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out
    return []


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "SchemaError",
    "forecast_schema",
    "parse_forecasts",
    "parse_state_body",
    "render_forecast_field",
    "state_body_schema",
]
