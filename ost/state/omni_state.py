"""The six-field Omni-State.

The state carries the model's current interpretation at one decision chunk. Its first
four fields form the state body the policy rewrites; the forecaster fills the fifth and
the gate the sixth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from ost.types import (
    AGGREGATE_AUDIO_STATES,
    AUDIO_STATES,
    OMNI_STATE_FIELDS,
    STATE_BODY_FIELDS,
    Interval,
    Sufficiency,
)

#: Pyramid level of a record: L1 holds base states, L2/L3 summaries, Linf the root.
StateLevel = Literal["L1", "L2", "L3", "Linf"]

#: Text written into a field whose required support is unavailable or unresolved.
UNCERTAIN = "uncertain"

#: Value of ``conflict`` when no cross-modal conflict is represented.
NO_CONFLICT = "none"


class OmniStateError(ValueError):
    """Raised when an Omni-State violates its schema."""


@dataclass
class StateBody:
    """``z_s^body``: the first four Omni-State fields (App. A Notation)."""

    visual_evidence: str = ""
    audio_state: str = UNCERTAIN
    audio_evidence: str = ""
    conflict: str = NO_CONFLICT

    def validate(self, *, aggregate: bool = False) -> "StateBody":
        """Check the App. A.2 field contract.

        Every base Omni-State emits all six fields. ``audio_state`` is mandatory and
        restricted to ``present``, ``absent`` or ``uncertain``; ``audio_evidence`` is
        required unless ``audio_state`` is ``absent``. A folded state instead takes an
        aggregate ``audio_state`` (App. A.1).
        """
        allowed = AGGREGATE_AUDIO_STATES if aggregate else AUDIO_STATES
        if self.audio_state not in allowed:
            raise OmniStateError(
                f"audio_state must be one of {allowed}; got {self.audio_state!r}"
            )
        absent = self.audio_state in ("absent", "all_absent")
        if not absent and not self.audio_evidence.strip():
            raise OmniStateError(
                "audio_evidence is required unless audio_state is absent; "
                f"audio_state={self.audio_state!r}"
            )
        return self

    def as_dict(self) -> Dict[str, str]:
        return {name: getattr(self, name) for name in STATE_BODY_FIELDS}

    def serialize(self) -> str:
        return "\n".join(f"{name}: {getattr(self, name)}" for name in STATE_BODY_FIELDS)


@dataclass
class OmniState:
    """A complete or partially filled Omni-State ``z_s``.

    Construction order mirrors the paper: the policy produces the body, the forecaster
    fills ``forecast`` from admitted claims giving the provisional state ``z_bar_s``, and
    the gate writes ``sufficiency`` to complete ``z_s`` for storage in memory.
    """

    state_id: int
    chunk: int
    interval: Interval
    body: StateBody = field(default_factory=StateBody)
    level: StateLevel = "L1"
    #: Serialised representation of the admitted claims; empty for ``NO_FORECAST``.
    forecast: str = ""
    #: ``None`` until the gate writes its decision.
    sufficiency: Optional[Sufficiency] = None
    #: Claim identifiers whose records fill the forecast field.
    claim_ids: List[int] = field(default_factory=list)
    #: Registered span identifiers, keyed by field name.
    field_span_ids: Dict[str, int] = field(default_factory=dict)
    #: Children of a folded record.
    child_state_ids: List[int] = field(default_factory=list)
    #: Raw decoded text, retained for debugging and for exact token offsets.
    raw_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- schema -------------------------------------------------------------------

    @property
    def is_fold(self) -> bool:
        return self.level != "L1"

    @property
    def is_complete(self) -> bool:
        """True once ``sufficiency`` has been written."""
        return self.sufficiency is not None

    def validate(self) -> "OmniState":
        self.body.validate(aggregate=self.is_fold)
        if self.sufficiency is not None and not isinstance(self.sufficiency, Sufficiency):
            raise OmniStateError("sufficiency must be a Sufficiency value")
        return self

    # -- staged construction -------------------------------------------------------

    def provisional(self, forecast: str, claim_ids: Optional[List[int]] = None) -> "OmniState":
        """Assemble ``z_bar_s`` from the body and the admitted claims."""
        self.forecast = forecast
        self.claim_ids = list(claim_ids or [])
        return self

    def finalize(self, sufficiency: Sufficiency) -> "OmniState":
        """Write the gate decision, completing ``z_s``."""
        self.sufficiency = sufficiency
        return self.validate()

    def mark_field_uncertain(self, field_name: str) -> None:
        """Set a field to ``uncertain``.

        App. A.2 and A.4: a field depending on a rejected claim, or on unavailable or
        unresolved required support, becomes uncertain.
        """
        if field_name not in STATE_BODY_FIELDS:
            raise OmniStateError(f"{field_name!r} is not a state-body field")
        if field_name == "audio_state":
            self.body.audio_state = UNCERTAIN
        else:
            setattr(self.body, field_name, UNCERTAIN)

    # -- views ---------------------------------------------------------------------

    def fields(self) -> Dict[str, str]:
        """All six fields as text, in paper order."""
        out = dict(self.body.as_dict())
        out["forecast"] = self.forecast
        out["sufficiency"] = self.sufficiency.value if self.sufficiency else ""
        return {name: out[name] for name in OMNI_STATE_FIELDS}

    def serialize(self, *, include_header: bool = True) -> str:
        """Ledger serialisation. Field order is fixed so token offsets are stable."""
        body = "\n".join(f"{name}: {value}" for name, value in self.fields().items())
        if not include_header:
            return body
        header = (
            f"[state id={self.state_id} level={self.level} "
            f"interval={self.interval}]"
        )
        return f"{header}\n{body}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "chunk": self.chunk,
            "interval": list(self.interval.as_tuple()),
            "level": self.level,
            **self.fields(),
            "claim_ids": list(self.claim_ids),
            "child_state_ids": list(self.child_state_ids),
        }


__all__ = [
    "NO_CONFLICT",
    "UNCERTAIN",
    "OmniState",
    "OmniStateError",
    "StateBody",
    "StateLevel",
]
