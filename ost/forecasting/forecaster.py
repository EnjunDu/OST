"""The forecaster lane.

The forecaster turns the current interpretation into predictions that later evidence can
settle. It reads the rewritten state body, so a correction propagates into the next round
of predictions rather than only into the record of the past.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

from ost.config import ClaimConfig, DecodingConfig, ForecasterConfig
from ost.forecasting.admission import AdmissionOutcome, AdmissionRule
from ost.guidance.verdict_conditioned import BranchPair, GuidedDecoder, StageResult
from ost.models.backbone import Backbone, LaneRequest, MediaWindow
from ost.state.omni_state import StateBody
from ost.state.schema import (
    SchemaError,
    forecast_schema,
    parse_forecasts,
    render_forecast_field,
)
from ost.types import ForecastProposal, GenerationStage, SupportScope

LOGGER = logging.getLogger(__name__)

FORECAST_INSTRUCTION = (
    "You are the forecaster of a streaming audio-visual reasoning system.\n"
    "Read the question, the retained evidence, the reasoning ledger and the current state "
    "body. Propose at most {max_claims} prediction(s) about evidence that has NOT yet "
    "arrived and that a later check could confirm or refute.\n"
    "Only propose a prediction that is critical to answering the question. If nothing needs "
    "a future test, reply with an empty forecasts list.\n"
    "\n"
    "Reply with JSON only, using exactly these keys:\n"
    '{{"forecasts": [{{"forecast": "<what you predict will be observed>", '
    '"modality": "audio" | "visual" | "audio_visual", '
    '"delta_seconds": {windows}, '
    '"scope": "supports_state" | "supports_claim" | "supports_answer"}}]}}\n'
    "\n"
    "'forecast' is required and must be a concrete statement about future evidence. "
    "'modality' names the channel that can verify it; use 'audio_visual' only for a "
    "relation between the two. 'delta_seconds' is how many seconds of future evidence the "
    "check needs. 'scope' says what the prediction supports."
)


def build_forecast_prompt(
    question: str,
    state_body: StateBody,
    *,
    max_claims: int,
    window_durations: Sequence[float],
    evidence_summary: str = "",
) -> str:
    """Assemble the forecast lane prompt."""
    windows = " | ".join(f"{d:g}" for d in window_durations)
    parts = [
        FORECAST_INSTRUCTION.format(max_claims=max_claims, windows=windows),
        f"\nQuestion: {question}",
    ]
    if evidence_summary:
        parts.append(f"Retained evidence: {evidence_summary}")
    parts.append("Current state body:")
    parts.append(state_body.serialize())
    return "\n".join(parts)


@dataclass
class ForecastOutcome:
    """Result of one forecast stage."""

    proposals: List[ForecastProposal]
    admission: AdmissionOutcome
    raw_text: str
    guided_positions: int = 0
    guidance_scale: float = 0.0

    @property
    def emitted_no_forecast(self) -> bool:
        """Whether the lane declined to predict anything."""
        return not self.proposals

    @property
    def admitted(self) -> List[ForecastProposal]:
        return self.admission.admitted


class Forecaster:
    """``F_forecast`` of Eq. (17), decoded under the rule of Eq. (13).

    App. A.5: this pass uses the learned forecast prompt with the policy LoRA disabled,
    and it reads the realised rewritten state body in both branches, so the contrast acts
    conditional on the corrected interpretation.
    """

    def __init__(
        self,
        backbone: Backbone,
        forecaster_config: ForecasterConfig,
        claim_config: ClaimConfig,
        decoding_config: DecodingConfig,
        *,
        admission: Optional[AdmissionRule] = None,
    ) -> None:
        forecaster_config.validate()
        claim_config.validate()
        decoding_config.validate()
        self.backbone = backbone
        self.config = forecaster_config
        self.claim_config = claim_config
        self.decoding = decoding_config
        self.admission = admission or AdmissionRule(claim_config)

    # -- generation ------------------------------------------------------------------

    def propose(
        self,
        *,
        question: str,
        state_body: StateBody,
        ledger_text: str,
        span_ranges: Sequence,
        media: Optional[MediaWindow],
        branches: Optional[BranchPair] = None,
        decoder: Optional[GuidedDecoder] = None,
        evidence_summary: str = "",
        default_scope: SupportScope = "supports_answer",
    ) -> ForecastOutcome:
        """Generate forecast proposals for one chunk.

        When a guided decoder and a branch pair with a contrast are supplied, generation
        follows Eq. (13); otherwise the lane is generated directly, which is the behaviour
        App. A.5 specifies when no claim was refuted at this chunk.
        """
        prompt = build_forecast_prompt(
            question,
            state_body,
            max_claims=self.claim_config.max_per_chunk,
            window_durations=self.claim_config.window_durations,
            evidence_summary=evidence_summary,
        )
        request = LaneRequest(
            stage=GenerationStage.FORECAST,
            prompt=prompt,
            ledger_text=ledger_text,
            span_ranges=tuple(span_ranges),
            media=media,
            max_new_tokens=self.decoding.max_new_tokens_forecast,
            json_schema=(
                forecast_schema(
                    self.claim_config.window_durations, self.claim_config.max_per_chunk
                )
                if self.decoding.grammar
                else None
            ),
        )

        # App. A.5: the forecaster runs with the policy LoRA disabled.
        with self.backbone.adapter_scope(policy_adapter=not self.config.disable_policy_adapter):
            if decoder is not None and branches is not None and branches.has_contrast:
                result = self._generate_guided(decoder, request, branches)
                raw_text = result.text
                guided_positions = result.guided_positions
                scale = result.scale
            else:
                raw_text = self.backbone.generate(request)
                guided_positions = 0
                scale = 0.0

        try:
            proposals = parse_forecasts(
                raw_text, self.claim_config.window_durations, default_scope=default_scope
            )
        except SchemaError as exc:
            # An unparseable forecast lane means no claim is registered for this chunk, which
            # would otherwise look identical to the model deciding nothing needs a test.
            LOGGER.warning("forecast lane output did not parse: %s", exc)
            proposals = []
        if raw_text.strip() and not proposals:
            LOGGER.info(
                "forecast lane produced no admissible proposal; first 160 chars: %r",
                raw_text.strip()[:160],
            )
        # App. A.3: separately testable audio and visual facts become two claims.
        proposals = self.admission.split_separable(proposals)

        return ForecastOutcome(
            proposals=proposals,
            admission=AdmissionOutcome(),
            raw_text=raw_text,
            guided_positions=guided_positions,
            guidance_scale=scale,
        )

    def _generate_guided(
        self, decoder: GuidedDecoder, request: LaneRequest, branches: BranchPair
    ) -> StageResult:
        return decoder.run_stage(
            GenerationStage.FORECAST,
            branches,
            max_new_tokens=request.max_new_tokens,
        )

    # -- admission --------------------------------------------------------------------

    def admit(
        self,
        outcome: ForecastOutcome,
        *,
        issue_time: float,
        active_claim_count: int,
        depth_of: Optional[Mapping[int, int]] = None,
        stream_end: Optional[float] = None,
        is_terminal: bool = False,
    ) -> ForecastOutcome:
        """Apply the admission rule, giving ``P_t`` of Eq. (17)."""
        outcome.admission = self.admission.admit(
            outcome.proposals,
            issue_time=issue_time,
            active_claim_count=active_claim_count,
            depth_of=dict(depth_of or {}),
            stream_end=stream_end,
            is_terminal=is_terminal,
        )
        return outcome

    @staticmethod
    def render_field(proposals: Sequence[ForecastProposal]) -> str:
        """Render the Omni-State ``forecast`` field from admitted proposals."""
        return render_forecast_field(proposals)


__all__ = [
    "FORECAST_INSTRUCTION",
    "ForecastOutcome",
    "Forecaster",
    "build_forecast_prompt",
]
