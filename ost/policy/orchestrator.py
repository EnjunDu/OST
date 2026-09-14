"""The OST online inference loop.

This module is the executable form of the paper's inference algorithm. The ordering
inside a chunk is load-bearing and follows the algorithm exactly: verify before
rewriting, register the rewritten body before forecasting so new claims can cite it,
apply the gate after registration so the hard support check sees this chunk's claims, and
fold only after the state has been stored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import torch

from ost.config import OSTConfig
from ost.forecasting.admission import AdmissionRule
from ost.forecasting.forecaster import Forecaster
from ost.gating.answer_gate import AnswerGate
from ost.guidance.verdict_conditioned import (
    BranchPair,
    BranchState,
    GuidedDecoder,
    make_sampler,
)
from ost.memory.ledger import ReasoningLedger, SerializedLedger
from ost.models.backbone import Backbone, LaneRequest, MediaWindow
from ost.policy.answer import AnswerDecoder
from ost.retraction.attention_bias import SpanTokenRange, span_ranges_from_offsets
from ost.retraction.registry import SpanRegistry
from ost.state.omni_state import OmniState, StateBody
from ost.state.pyramid import Pyramid
from ost.state.schema import parse_state_body, state_body_schema
from ost.streaming.clock import DecisionChunk, DecisionClock
from ost.streaming.retention import RetentionBuffers
from ost.streaming.source import StreamSource
from ost.types import (
    AnswerResult,
    ChunkTrace,
    ClaimRecord,
    ClaimStatus,
    EpisodeResult,
    ForecastProposal,
    GenerationStage,
    SpanKind,
    Verdict,
)
from ost.verification.verifier import TypedVerifier

LOGGER = logging.getLogger(__name__)

STATE_INSTRUCTION = (
    "You maintain the Omni-State of a streaming audio-visual reasoning system.\n"
    "Write the four state-body fields for the current decision chunk:\n"
    "  visual_evidence: what the video shows, from the retained frames only.\n"
    "  audio_state: 'present', 'absent' or 'uncertain'.\n"
    "  audio_evidence: the audible events and speech. Required unless audio_state is "
    "'absent'.\n"
    "  conflict: any disagreement between what you see and what you hear, else 'none'.\n"
    "Refuted reasoning in the ledger carries a reduced reliability score; prefer the new "
    "evidence over an interpretation the stream has contradicted. If a field still needs "
    "evidence that has not arrived, write 'uncertain'.\n"
    "Reply with JSON only."
)


def build_state_prompt(question: str, evidence_summary: str = "") -> str:
    parts = [STATE_INSTRUCTION, f"\nQuestion: {question}"]
    if evidence_summary:
        parts.append(f"Retained evidence: {evidence_summary}")
    return "\n".join(parts)


@dataclass
class OrchestratorComponents:
    """The learned and rule-based pieces the loop drives."""

    backbone: Backbone
    verifier: TypedVerifier
    gate: AnswerGate
    forecaster: Forecaster
    answer_decoder: AnswerDecoder


class Orchestrator:
    """Implements the online inference algorithm."""

    def __init__(
        self,
        config: OSTConfig,
        components: OrchestratorComponents,
    ) -> None:
        config.validate()
        self.config = config
        self.components = components
        self._sampler = make_sampler(
            config.decoding.temperature,
            config.decoding.top_p,
            _seeded_generator(config.decoding.seed),
        )

    # -- main entry point -------------------------------------------------------------

    def run(
        self,
        *,
        question: str,
        source: StreamSource,
        options: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> EpisodeResult:
        """Run OST over one question and one stream.

        Returns as soon as the gate fires, or at stream termination.
        """
        cfg = self.config
        clock = DecisionClock(
            cfg.streaming.decision_interval_seconds,
            cfg.streaming.perception_interval_seconds,
            end_time=source.duration_seconds,
        )
        buffers = RetentionBuffers(
            cfg.retention,
            cfg.streaming.decision_interval_seconds,
            cfg.streaming.perception_interval_seconds,
        )
        registry = SpanRegistry(cfg.claims, cfg.retraction, clock)
        pyramid = Pyramid(
            cfg.memory.pyramid,
            cfg.streaming.decision_interval_seconds,
            summarizer=self._make_summarizer(),
            id_allocator=None,
            keep_archive=cfg.memory.keep_archive,
        )
        ledger = ReasoningLedger(
            pyramid,
            registry,
            token_budget=cfg.retention.reasoning_tokens,
            count_tokens=lambda text: len(self.components.backbone.tokenize(text)),
        )

        traces: List[ChunkTrace] = []
        answer: Optional[AnswerResult] = None
        stop_chunk = 0
        stop_time = 0.0
        stopped_early = False

        for chunk in clock.chunks():
            trace = ChunkTrace(chunk=chunk.index, t_end=chunk.t_end)

            # Alg. 1 line 3: ingest observations up to t_s into the separate buffers.
            self._ingest(source, buffers, clock, chunk, question)

            # Alg. 1 line 4: snapshot the registry before any verdict is applied, so the
            # comparison branch can restore the pre-refutation view.
            snapshot = registry.snapshot(chunk.index)
            refuted_ids: List[int] = []
            margins: List[float] = []

            # Alg. 1 lines 5-11: review every due claim on its own modality and interval.
            for claim in registry.due_claims(chunk.index, now=chunk.t_end):
                result = self.components.verifier.verify(claim, buffers, now=chunk.t_end)
                updated = registry.apply_verdict(
                    result, chunk=chunk.index, time=chunk.t_end
                )
                ledger.verdict_log.append(registry.verdict_log[-1])
                trace.verdicts.append(registry.verdict_log[-1])

                if result.verdict is Verdict.REFUTED:
                    refuted_ids.append(claim.claim_id)
                    margins.append(result.contradiction_margin)

                if updated.status in (ClaimStatus.SETTLED, ClaimStatus.EXPIRED):
                    buffers.release(claim.claim_id)
                    updated.pinned = False
                    if updated.status is ClaimStatus.EXPIRED:
                        trace.expired_claim_ids.append(claim.claim_id)

            # Alg. 1 line 12: at termination, expire all remaining claims.
            if chunk.is_terminal:
                for claim_id in registry.expire_all_open(chunk=chunk.index, time=chunk.t_end):
                    buffers.release(claim_id)
                    registry.claims[claim_id].pinned = False
                    if claim_id not in trace.expired_claim_ids:
                        trace.expired_claim_ids.append(claim_id)

            buffers.evict()

            # Alg. 1 line 13: propagate effective reliability, then assemble the branches.
            registry.propagate()
            branches = BranchPair.from_snapshot(
                registry, snapshot, refuted_ids, margins, cfg.guidance.lambda_0
            )
            trace.refuted_claim_ids = list(refuted_ids)
            trace.guidance_scale = branches.scale
            trace.guidance_applied = branches.has_contrast

            evidence_summary = self._evidence_summary(buffers, for_answer=False)
            state_media = self._media_window(buffers, for_answer=False, source=source)

            # Alg. 1 line 14: rewrite the state body under the guided decoding rule.
            body, cited = self._rewrite_stage(
                question=question,
                ledger=ledger,
                branches=branches,
                media=state_media,
                evidence_summary=evidence_summary,
            )

            state = OmniState(
                state_id=self._allocate_state_id(pyramid),
                chunk=chunk.index,
                interval=chunk.interval,
                body=body,
            )

            # Alg. 1 line 15: register the body's support parents. Revision links to the
            # refuted claims stay separate from the support graph.
            self._register_body(registry, state, cited, chunk.index, refuted_ids)

            # Alg. 1 lines 16-19: forecast from the rewritten body unless the stream ended.
            admitted = []
            if not chunk.is_terminal:
                admitted = self._forecast_stage(
                    question=question,
                    state=state,
                    ledger=ledger,
                    registry=registry,
                    buffers=buffers,
                    branches=branches,
                    media=state_media,
                    evidence_summary=evidence_summary,
                    chunk=chunk,
                    trace=trace,
                )

            # Alg. 1 line 20: assemble the provisional state and register forecast support.
            state.provisional(
                Forecaster.render_field([c.proposal for c in admitted]),
                [c.claim.claim_id for c in admitted],
            )
            sufficiency_span = registry.register_span(
                SpanKind.STATE_FIELD,
                "",
                chunk.index,
                field_name="sufficiency",
                state_id=state.state_id,
            )
            state.field_span_ids["sufficiency"] = sufficiency_span.span_id
            for entry in admitted:
                registry.auto_link_scope(
                    entry.claim,
                    field_span_ids=state.field_span_ids,
                    sufficiency_span_id=sufficiency_span.span_id,
                )
            registry.propagate()

            # Alg. 1 lines 21-22: apply the gate, then collect expired answer support.
            serialized = self._serialize(ledger, branches.corrected)
            decision = self.components.gate.decide(
                registry,
                question=question,
                state=state,
                ledger_text=serialized.text,
                is_terminal=chunk.is_terminal,
            )
            trace.gate_action = decision.action
            trace.gate_score = decision.sufficiency_score
            trace.hard_wait = decision.hard_wait

            # Alg. 1 line 23: finalise sufficiency and append the state to memory.
            state.finalize(decision.action)
            trace.state_id = state.state_id
            fold_events = pyramid.append(state)
            trace.folds = [
                {
                    "level": e.level,
                    "target": e.target_level,
                    "children": e.child_state_ids,
                    "summary": e.summary_state_id,
                }
                for e in fold_events
            ]
            traces.append(trace)

            # Alg. 1 lines 24-26: answer from the rebuilt context and mask.
            if decision.is_answer:
                answer_serialized = self._serialize(ledger, branches.corrected)
                answer = self.components.answer_decoder.decode(
                    question=question,
                    state=state,
                    ledger_text=answer_serialized.text,
                    span_ranges=self._span_ranges(answer_serialized, branches.corrected),
                    media=self._media_window(
                        buffers, for_answer=True, source=source
                    ),
                    evidence_summary=self._evidence_summary(buffers, for_answer=True),
                    expired_support=decision.expired_support,
                    unresolved_support=[
                        cid
                        for cid in decision.unresolved_support
                        if cid not in decision.expired_support
                    ],
                )
                stop_chunk = chunk.index
                stop_time = chunk.t_end
                stopped_early = not chunk.is_terminal
                break

            stop_chunk = chunk.index
            stop_time = chunk.t_end

        if answer is None:
            # Eq. (8) guarantees an answer at T_end, so reaching here means the stream
            # yielded no chunks at all.
            raise RuntimeError(
                "the stream produced no decision chunk; check the source duration"
            )

        return EpisodeResult(
            answer=answer,
            stop_chunk=stop_chunk,
            stop_time=stop_time,
            stopped_early=stopped_early,
            traces=traces,
            states=pyramid.archive,
            verdict_log=list(registry.verdict_log),
            metadata={
                **dict(metadata or {}),
                "registry": registry.stats(),
                "retention": buffers.stats(),
                "ledger": ledger.stats(),
                "options_supplied_to_lanes": False,
            },
        )

    # -- stages -------------------------------------------------------------------------

    def _rewrite_stage(
        self,
        *,
        question: str,
        ledger: ReasoningLedger,
        branches: BranchPair,
        media: Optional[MediaWindow],
        evidence_summary: str,
    ) -> Tuple[StateBody, List[int]]:
        """Generate ``z_s^body`` under Eq. (13)."""
        prompt = build_state_prompt(question, evidence_summary)
        serialized_positive = self._serialize(ledger, branches.corrected)
        request = LaneRequest(
            stage=GenerationStage.REWRITE,
            prompt=prompt,
            ledger_text=serialized_positive.text,
            span_ranges=self._span_ranges(serialized_positive, branches.corrected),
            media=media,
            max_new_tokens=self.config.decoding.max_new_tokens_state,
            json_schema=state_body_schema() if self.config.decoding.grammar else None,
        )

        with self.components.backbone.adapter_scope(policy_adapter=True):
            if branches.has_contrast:
                decoder = self._make_decoder(ledger, prompt, media, request)
                result = decoder.run_stage(
                    GenerationStage.REWRITE,
                    branches,
                    max_new_tokens=request.max_new_tokens,
                )
                raw = result.text
            else:
                raw = self.components.backbone.generate(request)

        body, cited = parse_state_body(raw)
        return body, cited

    def _forecast_stage(
        self,
        *,
        question: str,
        state: OmniState,
        ledger: ReasoningLedger,
        registry: SpanRegistry,
        buffers: RetentionBuffers,
        branches: BranchPair,
        media: Optional[MediaWindow],
        evidence_summary: str,
        chunk: DecisionChunk,
        trace: ChunkTrace,
    ) -> List["_AdmittedClaim"]:
        """Forecast, admit and register, giving ``P_t``."""
        serialized = self._serialize(ledger, branches.corrected)
        outcome = self.components.forecaster.propose(
            question=question,
            state_body=state.body,
            ledger_text=serialized.text,
            span_ranges=self._span_ranges(serialized, branches.corrected),
            media=media,
            branches=branches,
            decoder=(
                self._make_decoder(ledger, "", media, None)
                if branches.has_contrast
                else None
            ),
            evidence_summary=evidence_summary,
        )
        depth_of = {
            span_id: registry.support_depth(span_id) for span_id in registry.spans
        }
        outcome = self.components.forecaster.admit(
            outcome,
            issue_time=chunk.t_end,
            active_claim_count=registry.active_claim_count(),
            depth_of=depth_of,
            stream_end=None,
            is_terminal=chunk.is_terminal,
        )
        trace.rejected_proposals = len(outcome.admission.rejected)

        # App. A.2: a field depending on a rejected claim becomes uncertain.
        for field_name in AdmissionRule.dependent_fields(outcome.admission):
            state.mark_field_uncertain(field_name)

        admitted: List[_AdmittedClaim] = []
        support_spans = [
            span_id
            for name, span_id in state.field_span_ids.items()
            if name in ("visual_evidence", "audio_evidence", "conflict")
        ]
        for proposal in outcome.admission.admitted:
            claim = registry.register_claim(
                proposal,
                chunk=chunk.index,
                issue_time=chunk.t_end,
                state_id=state.state_id,
                support_span_ids=support_spans,
            )
            # Sec. 3.2: pin the pending evidence window so eviction cannot remove the
            # evidence this claim will be verified against.
            buffers.pin(claim.claim_id, claim.interval, claim.modality)
            admitted.append(_AdmittedClaim(proposal=proposal, claim=claim))
            trace.admitted_claim_ids.append(claim.claim_id)
        return admitted

    def _register_body(
        self,
        registry: SpanRegistry,
        state: OmniState,
        cited: Sequence[int],
        chunk: int,
        refuted_ids: Sequence[int],
    ) -> None:
        """Register the body's fields as spans and attach citations and revision links."""
        for field_name in ("visual_evidence", "audio_state", "audio_evidence", "conflict"):
            span = registry.register_span(
                SpanKind.STATE_FIELD,
                getattr(state.body, field_name),
                chunk,
                parents=cited,
                field_name=field_name,
                state_id=state.state_id,
            )
            state.field_span_ids[field_name] = span.span_id

        # App. A.5: a mention of a refuted claim in the conflict record is a revision
        # link, not a support parent, so the new interpretation carries its own provenance
        # while the attenuation of dependent reasoning survives.
        conflict_span = state.field_span_ids.get("conflict")
        if conflict_span is not None and refuted_ids:
            revised = [
                registry.claims[cid].span_id
                for cid in refuted_ids
                if cid in registry.claims
            ]
            registry.add_revision_link(conflict_span, revised)

    # -- helpers -------------------------------------------------------------------------

    def _ingest(
        self,
        source: StreamSource,
        buffers: RetentionBuffers,
        clock: DecisionClock,
        chunk: DecisionChunk,
        question: str,
    ) -> None:
        """Ingest one chunk's perception timestamps into the separate buffers.

        ``RetentionBuffers.ingest`` re-pins slots that fall inside an already-pinned
        evidence window, so a claim registered before its evidence arrived still holds its
        interval once the slots appear.
        """
        for timestamp in clock.perception_timestamps(chunk):
            visual, audio = source.perceive(timestamp, question=question)
            buffers.ingest(timestamp, visual=visual, audio=audio)

    def _serialize(self, ledger: ReasoningLedger, branch: BranchState) -> SerializedLedger:
        return ledger.serialize_with_offsets(
            self.components.backbone.tokenize,
            effective_reliability=branch.effective_reliability,
            claim_status=branch.claim_status,
            omit_claim_ids=branch.omit_claim_ids,
        )

    def _span_ranges(
        self, serialized: SerializedLedger, branch: BranchState
    ) -> List[SpanTokenRange]:
        return span_ranges_from_offsets(
            serialized.span_offsets, branch.effective_reliability
        )

    def _make_decoder(
        self,
        ledger: ReasoningLedger,
        prompt: str,
        media: Optional[MediaWindow],
        template: Optional[LaneRequest],
    ) -> GuidedDecoder:
        """Build a guided decoder whose branches differ only in the serialised ledger."""
        backbone = self.components.backbone

        def logit_fn(branch: BranchState, prefix: Sequence[int], stage: GenerationStage):
            serialized = self._serialize(ledger, branch)
            request = LaneRequest(
                stage=stage,
                prompt=template.prompt if template is not None else prompt,
                ledger_text=serialized.text,
                span_ranges=self._span_ranges(serialized, branch),
                media=media,
                max_new_tokens=(
                    template.max_new_tokens if template is not None else 128
                ),
                json_schema=template.json_schema if template is not None else None,
            )
            return backbone.next_token_logits(request, prefix)

        return GuidedDecoder(
            logit_fn,
            decode_fn=backbone.detokenize,
            eos_token_ids=backbone.eos_token_ids,
            sampler=self._sampler,
        )

    def _make_summarizer(self):
        """Backbone-based evidence summariser for pyramid folds (App. A.1)."""
        backbone = self.components.backbone

        def summarize(children):
            visual = " ".join(
                c.body.visual_evidence for c in children if c.body.visual_evidence
            ).strip()
            audio = " ".join(
                c.body.audio_evidence for c in children if c.body.audio_evidence
            ).strip()
            return visual, audio

        return summarize

    def _media_window(
        self,
        buffers: RetentionBuffers,
        *,
        for_answer: bool,
        source: Optional[StreamSource] = None,
    ) -> MediaWindow:
        window = buffers.context_window(for_answer=for_answer)
        span = (
            self.config.retention.answer_window_seconds
            if for_answer
            else self.config.retention.state_window_seconds
        )
        t_start = max(0.0, buffers.now - span)
        clip = source.window_clip(t_start, buffers.now) if source is not None else None
        return MediaWindow(
            visual=window["visual"],
            audio=window["audio"],
            t_start=t_start,
            t_end=buffers.now,
            clip_path=str(clip) if clip is not None else None,
        )

    def _evidence_summary(self, buffers: RetentionBuffers, *, for_answer: bool) -> str:
        window = buffers.context_window(for_answer=for_answer)
        visual = len(window["visual"])
        audio = len([s for s in window["audio"] if s.present])
        return (
            f"{visual} retained visual second(s), {audio} retained audio second(s) "
            f"up to t={buffers.now:.1f}s"
        )

    @staticmethod
    def _allocate_state_id(pyramid: Pyramid) -> int:
        return len(pyramid.archive) + 1


@dataclass
class _AdmittedClaim:
    """An admitted proposal together with the claim record it became."""

    proposal: ForecastProposal
    claim: ClaimRecord


def _seeded_generator(seed: int) -> Optional[torch.Generator]:
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


__all__ = [
    "STATE_INSTRUCTION",
    "Orchestrator",
    "OrchestratorComponents",
    "build_state_prompt",
]
