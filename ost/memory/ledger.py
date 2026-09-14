"""The reasoning ledger.

The ledger is what the model reads back: earlier Omni-States together with their claim
and verdict records. It is re-serialised at every chunk so that token offsets are exact,
because the verdict-reliability bias of Eq. (12) needs to know precisely which key
positions belong to which registered span.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ost.retraction.registry import SpanRegistry
from ost.state.omni_state import OmniState
from ost.state.pyramid import Pyramid
from ost.types import ClaimRecord, ClaimStatus, VerdictLogEntry

LOGGER = logging.getLogger(__name__)

#: Tokenizer callback: text in, token ids out. Used to compute exact span offsets.
TokenizeFn = Callable[[str], Sequence[int]]


@dataclass
class VerdictLog:
    """Append-only verdict log ``V`` (App. A.3)."""

    entries: List[VerdictLogEntry] = field(default_factory=list)

    def append(self, entry: VerdictLogEntry) -> None:
        self.entries.append(entry)

    def extend(self, entries: Iterable[VerdictLogEntry]) -> None:
        self.entries.extend(entries)

    def for_claim(self, claim_id: int) -> List[VerdictLogEntry]:
        return [e for e in self.entries if e.claim_id == claim_id]

    def latest_for_claim(self, claim_id: int) -> Optional[VerdictLogEntry]:
        matches = self.for_claim(claim_id)
        return matches[-1] if matches else None

    def __len__(self) -> int:
        return len(self.entries)


@dataclass
class SerializedLedger:
    """A serialised ledger together with its span token offsets."""

    text: str
    token_ids: List[int]
    #: Span identifier to ``(start, end)`` token offsets inside ``token_ids``.
    span_offsets: Dict[int, Tuple[int, int]]
    #: Token range covering the whole ledger, i.e. the reasoning-token region.
    reasoning_range: Tuple[int, int]

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


class ReasoningLedger:
    """The active reasoning ledger ``M_hat_s`` plus its serialisation.

    App. A.1: the active ledger contains the current memory pyramid and the Omni-States
    with open claims; older records remain in the append-only archive. Its capacity is
    fixed, while the archive grows with stream length.
    """

    def __init__(
        self,
        pyramid: Pyramid,
        registry: SpanRegistry,
        *,
        token_budget: Optional[int] = None,
        count_tokens: Optional[Callable[[str], int]] = None,
    ) -> None:
        self.pyramid = pyramid
        self.registry = registry
        self.verdict_log = VerdictLog()
        #: B_R from Alg. 1: the active reasoning ledger's token budget.
        self.token_budget = int(token_budget) if token_budget else None
        self._count_tokens = count_tokens
        #: Records dropped by the most recent budget trim, for tracing.
        self.trimmed_states = 0

    # -- selection -----------------------------------------------------------------

    def active_states(self) -> List[OmniState]:
        """States in the active ledger: the pyramid plus any state with an open claim.

        App. A.1 gives the active context a fixed capacity while the archive grows with the
        stream. When a token budget is configured, the oldest records are dropped first, but a
        state carrying an open claim is always kept: dropping it would make the claim's own
        premise unreadable at the moment it is about to be verified.
        """
        states = list(self.pyramid.active_states())
        seen = {state.state_id for state in states}
        open_state_ids = {
            claim.state_id
            for claim in self.registry.claims.values()
            if claim.is_open and claim.state_id is not None
        }
        for state_id in sorted(open_state_ids - seen):
            state = self.pyramid.state_by_id(state_id)
            if state is not None:
                states.append(state)
        ordered = sorted(states, key=lambda s: (s.interval.start, s.state_id))
        return self._apply_token_budget(ordered, protected=open_state_ids)

    def _apply_token_budget(
        self, states: List[OmniState], *, protected: Set[int]
    ) -> List[OmniState]:
        """Trim the oldest unprotected states until the ledger fits B_R."""
        self.trimmed_states = 0
        if not self.token_budget or self._count_tokens is None or not states:
            return states

        measure = self._count_tokens
        total = sum(measure(state.serialize()) for state in states)
        if total <= self.token_budget:
            return states

        kept = list(states)
        for state in states:
            if total <= self.token_budget:
                break
            if state.state_id in protected:
                continue
            if len(kept) <= 1:
                # Never trim to nothing: the current chunk's own state must survive.
                break
            total -= measure(state.serialize())
            kept.remove(state)
            self.trimmed_states += 1

        if total > self.token_budget:
            LOGGER.debug(
                "active ledger is %d tokens over its budget of %d after trimming %d "
                "record(s); the remainder is protected by open claims",
                total - self.token_budget,
                self.token_budget,
                self.trimmed_states,
            )
        return kept

    def open_claims(self) -> List[ClaimRecord]:
        return sorted(self.registry.open_claims(), key=lambda c: c.claim_id)

    # -- serialisation -------------------------------------------------------------

    def render(
        self,
        *,
        effective_reliability: Optional[Mapping[int, float]] = None,
        claim_status: Optional[Mapping[int, ClaimStatus]] = None,
        omit_claim_ids: Sequence[int] = (),
        include_verdicts: bool = True,
    ) -> Tuple[str, List[Tuple[int, int, int]]]:
        """Render the ledger to text with character spans per registered span.

        ``omit_claim_ids`` drops the verdicts for the given claims, which is how App. A.5
        builds the comparison branch: the observations, confirmations and unresolved
        verdicts stay, only this chunk's refutations disappear.

        Returns the text and a list of ``(span_id, char_start, char_end)`` triples.
        """
        omit = {int(cid) for cid in omit_claim_ids}
        status_view = dict(claim_status or {cid: c.status for cid, c in self.registry.claims.items()})
        reliability_view = dict(effective_reliability or self.registry.effective_reliability)

        parts: List[str] = []
        char_spans: List[Tuple[int, int, int]] = []
        cursor = 0

        def emit(text: str, span_id: Optional[int] = None) -> None:
            nonlocal cursor
            if span_id is not None:
                char_spans.append((span_id, cursor, cursor + len(text)))
            parts.append(text)
            cursor += len(text)

        emit("## Reasoning ledger\n")

        for state in self.active_states():
            emit(f"[state {state.state_id} {state.level} {state.interval}]\n")
            for field_name, value in state.fields().items():
                if field_name == "sufficiency" and not value:
                    continue
                span_id = state.field_span_ids.get(field_name)
                emit(f"{field_name}: ")
                emit(value if value else "-", span_id)
                emit("\n")

        claims = [c for c in self.registry.claims.values() if c.claim_id not in omit or True]
        if claims:
            emit("## Claims\n")
            for claim in sorted(claims, key=lambda c: c.claim_id):
                status = status_view.get(claim.claim_id, claim.status)
                rho = reliability_view.get(claim.span_id, claim.reliability)
                emit(
                    f"[claim {claim.claim_id} {claim.modality} due={claim.interval} "
                    f"scope={claim.scope} status={status.value} rho={rho:.3f}] "
                )
                emit(claim.text, claim.span_id)
                emit("\n")

        if include_verdicts:
            visible = [e for e in self.verdict_log.entries if e.claim_id not in omit]
            if visible:
                emit("## Verdicts\n")
                for entry in visible:
                    emit(
                        f"claim {entry.claim_id}: {entry.verdict.value} "
                        f"(gamma={entry.contradiction_margin:.3f}) at t={entry.time:.1f}\n"
                    )

        return "".join(parts), char_spans

    def serialize_with_offsets(
        self,
        tokenize: TokenizeFn,
        *,
        effective_reliability: Optional[Mapping[int, float]] = None,
        claim_status: Optional[Mapping[int, ClaimStatus]] = None,
        omit_claim_ids: Sequence[int] = (),
    ) -> SerializedLedger:
        """Serialise the ledger and compute exact token offsets per span.

        Offsets are derived by tokenising prefixes, which is exact for any tokenizer
        without assuming a character-to-token ratio. App. A.4 requires exact offsets
        because an off-by-one bias would attenuate the wrong reasoning.
        """
        text, char_spans = self.render(
            effective_reliability=effective_reliability,
            claim_status=claim_status,
            omit_claim_ids=omit_claim_ids,
        )
        token_ids = list(tokenize(text))

        # Map character boundaries to token boundaries by tokenising prefixes. Cache the
        # boundaries actually needed rather than every character position.
        boundaries = sorted({0, *(c for _, c, _ in char_spans), *(c for _, _, c in char_spans)})
        char_to_token: Dict[int, int] = {}
        for boundary in boundaries:
            if boundary == 0:
                char_to_token[0] = 0
                continue
            char_to_token[boundary] = len(tokenize(text[:boundary]))

        span_offsets: Dict[int, Tuple[int, int]] = {}
        for span_id, start_char, end_char in char_spans:
            start = min(char_to_token.get(start_char, 0), len(token_ids))
            end = min(char_to_token.get(end_char, start), len(token_ids))
            if end <= start:
                continue
            existing = span_offsets.get(span_id)
            if existing is None:
                span_offsets[span_id] = (start, end)
            else:
                span_offsets[span_id] = (min(existing[0], start), max(existing[1], end))

        return SerializedLedger(
            text=text,
            token_ids=token_ids,
            span_offsets=span_offsets,
            reasoning_range=(0, len(token_ids)),
        )

    # -- diagnostics ---------------------------------------------------------------

    def stats(self) -> Dict[str, object]:
        return {
            "active_states": len(self.active_states()),
            "open_claims": len(self.open_claims()),
            "archive_states": len(self.pyramid.archive),
            "verdicts": len(self.verdict_log),
            "token_budget": self.token_budget,
            "trimmed_states": self.trimmed_states,
        }


__all__ = ["ReasoningLedger", "SerializedLedger", "TokenizeFn", "VerdictLog"]
