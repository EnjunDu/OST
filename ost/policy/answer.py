"""Answer generation.

The answer is decoded from the corrected memory, with the reliability mask rebuilt to
include the state that fired the gate. Answer options are deliberately withheld from the
streaming lanes and supplied only during normalisation, so response timing is never
contaminated by the answer space collapsing early.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from ost.config import DecodingConfig
from ost.models.backbone import Backbone, LaneRequest, MediaWindow
from ost.state.omni_state import OmniState
from ost.types import AnswerResult, GenerationStage

LOGGER = logging.getLogger(__name__)

ANSWER_INSTRUCTION = (
    "You are answering a question about an audio-visual stream you have been watching.\n"
    "Use the reasoning ledger and the retained evidence below. Reasoning that has been "
    "refuted carries a reduced reliability score and should not drive your answer.\n"
    "Answer the question directly and concisely. Do not restate the question."
)

#: Prefix marking an answer whose support is still unresolved (Sec. 3.4, App. A.6).
LOW_CONFIDENCE_MARKER = "[low-confidence]"


def build_answer_prompt(
    question: str,
    state: OmniState,
    *,
    evidence_summary: str = "",
    expired_support: Sequence[int] = (),
) -> str:
    """Assemble the answer lane prompt.

    Expired answer support is named in the prompt as well as flagged on the result, so the
    model can hedge rather than assert a value its own scheduled test never settled.
    """
    parts = [ANSWER_INSTRUCTION, f"\nQuestion: {question}"]
    if evidence_summary:
        parts.append(f"Retained evidence: {evidence_summary}")
    parts.append("Current state:")
    parts.append(state.serialize(include_header=False))
    if expired_support:
        ids = ", ".join(str(i) for i in expired_support)
        parts.append(
            f"Unresolved support: claims {ids} expired without a decisive check. "
            "State your answer but make its uncertainty explicit."
        )
    return "\n".join(parts)


@dataclass
class AnswerDecoder:
    """Decodes the final answer from the rebuilt answer context."""

    backbone: Backbone
    decoding: DecodingConfig

    def decode(
        self,
        *,
        question: str,
        state: OmniState,
        ledger_text: str,
        span_ranges: Sequence,
        media: Optional[MediaWindow],
        evidence_summary: str = "",
        expired_support: Sequence[int] = (),
        unresolved_support: Sequence[int] = (),
    ) -> AnswerResult:
        """Generate the answer with the reliability mask in force.

        Alg. 1 lines 25-26: the answer context and reliability mask are rebuilt to include
        the current Omni-State before the answer is generated, and the expired
        answer-supporting identifiers are flagged on the result.
        """
        request = LaneRequest(
            stage=GenerationStage.REWRITE,  # the answer lane uses the state policy
            prompt=build_answer_prompt(
                question,
                state,
                evidence_summary=evidence_summary,
                expired_support=expired_support,
            ),
            ledger_text=ledger_text,
            span_ranges=tuple(span_ranges),
            media=media,
            max_new_tokens=self.decoding.max_new_tokens_answer,
            metadata={"lane": "answer"},
        )
        with self.backbone.adapter_scope(policy_adapter=True):
            text = self.backbone.generate(request)

        return AnswerResult(
            text=text.strip(),
            expired_support=[int(i) for i in expired_support],
            unresolved_support=[int(i) for i in unresolved_support],
        )


__all__ = [
    "ANSWER_INSTRUCTION",
    "LOW_CONFIDENCE_MARKER",
    "AnswerDecoder",
    "build_answer_prompt",
]
