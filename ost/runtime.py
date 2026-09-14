"""Runtime assembly.

One place that turns a configuration plus a set of checkpoints into a running
orchestrator, so every entry point builds the same object.

Each learned component is optional: the loop runs on an unmodified backbone, which is
what makes the mechanism inspectable without any checkpoint. What was actually loaded is
recorded rather than assumed, because a run with a zero-shot verifier is a different
run from one with a fitted head and must not be mistaken for it.

Every checkpoint can be named either on the command line or in the configuration's
``checkpoints`` fields; the command line wins. Relative paths resolve against the working
directory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from ost.config import OSTConfig
from ost.forecasting.admission import AdmissionRule
from ost.forecasting.forecaster import Forecaster
from ost.gating.answer_gate import AnswerGate, HeadGateScorer, LearnedGateHead
from ost.models.backbone import Backbone, build_backbone
from ost.models.soft_prompt import load_soft_prompt
from ost.policy.answer import AnswerDecoder
from ost.policy.orchestrator import Orchestrator, OrchestratorComponents
from ost.types import GenerationStage
from ost.verification.scoring_head import HeadScorer, VerifierScoringHead
from ost.verification.verifier import TypedVerifier

LOGGER = logging.getLogger(__name__)


class RuntimeError_(RuntimeError):
    """Raised when a required runtime component cannot be assembled."""


@dataclass
class RuntimeComponents:
    """What was actually loaded, so a run can be described honestly."""

    backbone: Backbone
    orchestrator: Orchestrator
    verifier_head_loaded: bool = False
    gate_head_loaded: bool = False
    forecast_prompt_loaded: bool = False
    policy_adapter_loaded: bool = False
    calibrated_thresholds: bool = False
    verdict_bias_probe: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> Dict[str, Any]:
        """Summarise the run's provenance.

        Deliberately flags rather than paths: a run record is meant to be shareable, and a
        filesystem layout is not part of what a result depends on.
        """
        return {
            "backbone": self.backbone.info(),
            "verifier_head_loaded": self.verifier_head_loaded,
            "gate_head_loaded": self.gate_head_loaded,
            "forecast_prompt_loaded": self.forecast_prompt_loaded,
            "policy_adapter_loaded": self.policy_adapter_loaded,
            "calibrated_thresholds": self.calibrated_thresholds,
            "zero_shot_verifier": not self.verifier_head_loaded,
            "verdict_bias_probe": dict(self.verdict_bias_probe),
        }


def _resolve_checkpoint(
    explicit: Optional[str],
    configured: Optional[str],
    *,
    what: str,
    cli_flag: str,
    config_key: str,
) -> Optional[Path]:
    """Pick a checkpoint path from the command line, then the configuration.

    A path that was asked for but is absent is an error rather than a silent fallback to
    the unmodified component: a missing checkpoint that degrades quietly is the failure
    most likely to be mistaken for a weak method.
    """
    candidate = explicit or configured
    if not candidate:
        return None
    path = Path(candidate).expanduser()
    if not path.exists():
        raise RuntimeError_(
            f"no {what} at {path}. It was requested via {cli_flag} or {config_key}; "
            "point it at the checkpoint or remove the setting to run without it."
        )
    return path


def build_runtime(
    config: OSTConfig,
    *,
    verifier_head_path: Optional[str] = None,
    gate_head_path: Optional[str] = None,
    forecast_prompt_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    thresholds_path: Optional[str] = None,
    probe_verdict_bias: bool = True,
) -> RuntimeComponents:
    """Assemble an orchestrator from a configuration and its checkpoints."""
    config = config.validate()

    verifier_head_file = _resolve_checkpoint(
        verifier_head_path,
        config.verifier.head_path,
        what="verifier scoring head",
        cli_flag="--verifier_head",
        config_key="verifier.head_path",
    )
    gate_head_file = _resolve_checkpoint(
        gate_head_path,
        config.gate.head_path,
        what="gate head",
        cli_flag="--gate_head",
        config_key="gate.head_path",
    )
    forecast_prompt_file = _resolve_checkpoint(
        forecast_prompt_path,
        config.forecaster.prompt_path,
        what="forecast soft prompt",
        cli_flag="--forecast_prompt",
        config_key="forecaster.prompt_path",
    )
    adapter_file = _resolve_checkpoint(
        adapter_path,
        config.policy.adapter_path,
        what="policy adapter",
        cli_flag="--adapter",
        config_key="policy.adapter_path",
    )
    thresholds_file = _resolve_checkpoint(
        thresholds_path,
        config.verifier.thresholds_path,
        what="verdict-band calibration",
        cli_flag="--thresholds",
        config_key="verifier.thresholds_path",
    )

    if thresholds_file is not None:
        config = _apply_thresholds(config, thresholds_file)

    backbone = build_backbone(
        config.model,
        decoding_config=config.decoding,
        forecaster_prompt_path=str(forecast_prompt_file) if forecast_prompt_file else None,
        adapter_path=str(adapter_file) if adapter_file else None,
    )
    backbone.load()

    probe: Dict[str, Any] = {}
    if probe_verdict_bias and hasattr(backbone, "probe_verdict_bias"):
        probe = backbone.probe_verdict_bias()
        if not probe.get("applied", False):
            raise RuntimeError_(
                "the verdict-reliability attention bias of Eq. (12) has no effect on this "
                "backbone, so retraction would be a no-op. Refusing to run. "
                f"Probe: {probe}"
            )

    verifier_head = None
    if verifier_head_file is not None:
        verifier_head = VerifierScoringHead.load(
            verifier_head_file, default_geometry=config.verifier.head_geometry()
        )
        if verifier_head.hidden_dim != backbone.hidden_size:
            raise RuntimeError_(
                f"the verifier head expects a {verifier_head.hidden_dim}-wide pooled "
                f"representation but this backbone is {backbone.hidden_size}; the head "
                "belongs to a different backbone"
            )
        verifier_head.to(backbone.device)
        LOGGER.info("loaded the verifier scoring head from %s", verifier_head_file)
    else:
        LOGGER.warning(
            "no verifier scoring head configured; falling back to a zero-shot verifier. "
            "The run record marks this."
        )
    scorer = (
        HeadScorer(verifier_head, backbone)
        if verifier_head is not None
        else _BackboneVerifierScorer(backbone)
    )
    verifier = TypedVerifier(config.verifier, scorer)

    gate_head = None
    if gate_head_file is not None:
        gate_head = LearnedGateHead.load(gate_head_file)
        gate_head.to(backbone.device)
        LOGGER.info("loaded the gate head from %s", gate_head_file)
    gate_scorer = HeadGateScorer(gate_head, backbone) if gate_head is not None else None
    gate = AnswerGate(config.gate, gate_scorer)

    if forecast_prompt_file is not None:
        embeddings, num_tokens, hidden_size = load_soft_prompt(forecast_prompt_file)
        if hidden_size != backbone.hidden_size:
            raise RuntimeError_(
                f"the forecast soft prompt is {hidden_size} wide but this backbone is "
                f"{backbone.hidden_size}; the prompt belongs to a different backbone"
            )
        if num_tokens != config.forecaster.prompt_tokens:
            LOGGER.warning(
                "the forecast soft prompt holds %d embeddings but forecaster."
                "prompt_tokens is %d; using the checkpoint's own length",
                num_tokens,
                config.forecaster.prompt_tokens,
            )
        backbone.set_soft_prompt(GenerationStage.FORECAST, embeddings)

    components = OrchestratorComponents(
        backbone=backbone,
        verifier=verifier,
        gate=gate,
        forecaster=Forecaster(
            backbone,
            config.forecaster,
            config.claims,
            config.decoding,
            admission=AdmissionRule(config.claims),
        ),
        answer_decoder=AnswerDecoder(backbone, config.decoding),
    )

    return RuntimeComponents(
        backbone=backbone,
        orchestrator=Orchestrator(config, components),
        verifier_head_loaded=verifier_head is not None,
        gate_head_loaded=gate_head is not None,
        forecast_prompt_loaded=forecast_prompt_file is not None,
        policy_adapter_loaded=adapter_file is not None,
        calibrated_thresholds=thresholds_file is not None,
        verdict_bias_probe=probe,
    )


def _apply_thresholds(config: OSTConfig, path: Path) -> OSTConfig:
    """Install calibrated per-modality verdict bands and the retraction scale.

    The bands decide which scores count as a refutation, so they belong with the head they
    were calibrated against rather than in the runtime defaults.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    overrides: Dict[str, Any] = {}
    for modality, band in (payload.get("thresholds") or {}).items():
        overrides[f"verifier.thresholds.{modality}.low"] = float(band["low"])
        overrides[f"verifier.thresholds.{modality}.high"] = float(band["high"])
    if payload.get("kappa") is not None:
        overrides["retraction.kappa"] = float(payload["kappa"])
    if not overrides:
        raise RuntimeError_(
            f"{path} carries no thresholds and no kappa, so it would change nothing"
        )
    LOGGER.info("applying the verdict-band calibration from %s", path)
    return config.merged(overrides)


class _BackboneVerifierScorer:
    """Zero-shot verifier used when no scoring head is configured.

    Eq. (7)'s verifier is a head over pooled hidden states. This fallback instead asks the
    frozen backbone directly, so the claim-verify-retract loop is inspectable without any
    checkpoint. It is not the paper's verifier, and
    :meth:`RuntimeComponents.describe` marks a run that used it.
    """

    def __init__(self, backbone: Backbone) -> None:
        self.backbone = backbone

    def score_claim(self, claim_text, evidence, modality, interval):  # noqa: ANN001
        import math

        from ost.models.backbone import LaneRequest
        from ost.verification.scoring_head import media_from_evidence

        prompt = (
            "You are verifying a prediction against evidence from one modality.\n"
            f"Prediction: {claim_text}\n"
            f"Verifying modality: {modality}\n"
            f"Evidence interval: ({interval.start:.1f}, {interval.end:.1f}]\n"
            "Does the evidence in this interval support the prediction? "
            "Answer exactly 'yes' or 'no'."
        )
        request = LaneRequest(
            stage=GenerationStage.FORECAST,
            prompt=prompt,
            ledger_text="",
            media=media_from_evidence(evidence, interval),
            max_new_tokens=4,
        )
        yes = float(self.backbone.sequence_log_prob(request, "yes", length_normalized=True))
        no = float(self.backbone.sequence_log_prob(request, "no", length_normalized=True))
        # A two-way softmax over the yes/no continuations gives a score in [0, 1].
        largest = max(yes, no)
        exp_yes = math.exp(yes - largest)
        exp_no = math.exp(no - largest)
        return exp_yes / (exp_yes + exp_no)


__all__ = ["RuntimeComponents", "build_runtime"]
