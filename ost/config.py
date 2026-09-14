"""Configuration objects for Omni-Streaming Thinking.

Every quantity in the paper's default configuration table is represented here as a
configurable field. Nothing that affects the method is hard-coded at a call site.

Configuration is layered: dataclass defaults (this file) < YAML file < CLI overrides.
Paths are never baked in; they arrive through YAML, CLI or environment variables.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from ost.types import MODALITIES

try:  # PyYAML is a hard requirement but keep the import error actionable.
    import yaml
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        "PyYAML is required to load OST configuration files. Install it with "
        "`pip install pyyaml`."
    ) from exc


class ConfigError(ValueError):
    """Raised when a configuration is internally inconsistent."""


# --------------------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------------------


@dataclass
class StreamingConfig:
    """Perception and decision grid.

    Implements the chunk grid of Eq. (1): ``t_s = min(s * decision_interval, T_end)``.
    """

    perception_interval_seconds: float = 1.0
    decision_interval_seconds: float = 4.0
    #: Video sampling rate handed to the backbone, in frames per second.
    frames_per_second: float = 1.0
    audio_sample_rate: int = 16_000

    def validate(self) -> None:
        if self.perception_interval_seconds <= 0:
            raise ConfigError("streaming.perception_interval_seconds must be positive")
        if self.decision_interval_seconds <= 0:
            raise ConfigError("streaming.decision_interval_seconds must be positive")
        ratio = self.decision_interval_seconds / self.perception_interval_seconds
        if abs(ratio - round(ratio)) > 1e-9:
            raise ConfigError(
                "streaming.decision_interval_seconds must be an integer multiple of "
                "streaming.perception_interval_seconds"
            )


# --------------------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------------------


@dataclass
class ClaimConfig:
    """Claim admission, scheduling and lifecycle limits (Table 3, App. A.2)."""

    #: Allowed evidence-window durations delta_i, fixed throughout the stream.
    window_durations: Tuple[float, ...] = (4.0, 8.0, 16.0)
    #: delta_max: longest admissible evidence window.
    max_window_seconds: float = 16.0
    #: delta_prime: retry slack granting right context to the acoustic front end.
    retry_slack_seconds: float = 4.0
    #: C_max: claims admitted per decision chunk.
    max_per_chunk: int = 2
    #: Q_max: simultaneously active (reviewable) claims.
    max_active: int = 8
    #: D_max: maximum claim-to-claim support depth.
    max_depth: int = 3

    def validate(self) -> None:
        if not self.window_durations:
            raise ConfigError("claims.window_durations must not be empty")
        if any(d <= 0 for d in self.window_durations):
            raise ConfigError("claims.window_durations entries must be positive")
        if abs(max(self.window_durations) - self.max_window_seconds) > 1e-9:
            raise ConfigError(
                "claims.max_window_seconds must equal max(claims.window_durations); "
                f"got {self.max_window_seconds} vs {max(self.window_durations)}"
            )
        if self.retry_slack_seconds < 0:
            raise ConfigError("claims.retry_slack_seconds must be non-negative")
        for name in ("max_per_chunk", "max_active", "max_depth"):
            if getattr(self, name) < 1:
                raise ConfigError(f"claims.{name} must be at least 1")


# --------------------------------------------------------------------------------------
# Retention and memory
# --------------------------------------------------------------------------------------


@dataclass
class RetentionScoreWeights:
    """Weights of the retention score components (App. A.1)."""

    visual_change: float = 1.0
    audio_salience: float = 1.0
    query_relevance: float = 1.0
    co_occurrence: float = 1.0

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (
            self.visual_change,
            self.audio_salience,
            self.query_relevance,
            self.co_occurrence,
        )


@dataclass
class RetentionConfig:
    """Separate audio and visual retention contract (Sec. 3.2, App. A.1).

    The two modalities never share a budget: dense visual tokens must not be able to
    displace the audio a pending claim needs for verification.
    """

    visual_tokens: int = 4096
    audio_tokens: int = 2048
    #: B_R: the active reasoning ledger's token budget.
    reasoning_tokens: int = 3072
    #: Perceptual retention floor, delta_max + delta_prime + decision_interval.
    floor_seconds: float = 24.0
    #: Dense window read by the state context.
    state_window_seconds: float = 4.0
    #: Dense window read by the answer context.
    answer_window_seconds: float = 8.0
    #: Number of sparse query-conditioned retrieval slots outside the dense window.
    sparse_retrieval_slots: int = 8
    score_weights: RetentionScoreWeights = field(default_factory=RetentionScoreWeights)

    def validate(self) -> None:
        for name in ("visual_tokens", "audio_tokens", "reasoning_tokens"):
            if getattr(self, name) < 1:
                raise ConfigError(f"retention.{name} must be positive")
        if self.floor_seconds <= 0:
            raise ConfigError("retention.floor_seconds must be positive")
        if self.state_window_seconds <= 0 or self.answer_window_seconds <= 0:
            raise ConfigError("retention dense windows must be positive")


@dataclass
class PyramidConfig:
    """Four-way Omni-State pyramid (Sec. 3.2, App. A.1, Table 3)."""

    branching: int = 4
    #: (K1, K2, K3) capacities for the 4 s, 16 s and 64 s levels.
    capacities: Tuple[int, int, int] = (6, 4, 4)
    #: Capacity of the long-term memory root.
    root_capacity: int = 1

    def validate(self) -> None:
        if self.branching < 2:
            raise ConfigError("pyramid.branching must be at least 2")
        if len(self.capacities) != 3:
            raise ConfigError("pyramid.capacities must have exactly three entries")
        if any(c < 1 for c in self.capacities):
            raise ConfigError("pyramid.capacities entries must be positive")
        if self.root_capacity < 1:
            raise ConfigError("pyramid.root_capacity must be positive")

    def level_span_seconds(self, decision_interval_seconds: float) -> Tuple[float, ...]:
        """Interval covered by one record at each level."""
        spans = [decision_interval_seconds]
        for _ in range(len(self.capacities) - 1):
            spans.append(spans[-1] * self.branching)
        return tuple(spans)


@dataclass
class MemoryConfig:
    """Active reasoning ledger versus append-only archive (App. A.1)."""

    pyramid: PyramidConfig = field(default_factory=PyramidConfig)
    #: Keep the archive in memory. Disable to bound RSS on very long streams.
    keep_archive: bool = True

    def validate(self) -> None:
        self.pyramid.validate()


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------

#: Calibration cap for uninformative intervals (App. A.3).
AXIOM_CAP_DEFAULT = 0.55
#: Minimum separation between the cap and the confirmation threshold (App. A.3).
AXIOM_CAP_MARGIN = 0.05


@dataclass
class ModalityThresholds:
    """Verdict band for one modality (App. A.3)."""

    low: float = 0.35
    high: float = 0.65

    def validate(self, modality: str, axiom_cap: float) -> None:
        if not 0.0 < self.low < self.high < 1.0:
            raise ConfigError(
                f"verifier.thresholds.{modality} must satisfy 0 < low < high < 1; "
                f"got low={self.low}, high={self.high}"
            )
        # Implements the App. A.3 cap constraint:
        #   tau_low <= AXIOM_CAP <= tau_high - AXIOM_CAP_MARGIN
        # A small tolerance: the constraint is a design requirement, not a bit-exact one, and
        # binary floating point makes 0.55 + 0.05 slightly exceed 0.60.
        tolerance = 1e-9
        if self.low > axiom_cap + tolerance:
            raise ConfigError(
                f"verifier.thresholds.{modality}.low must be <= axiom_cap "
                f"({axiom_cap}); got {self.low}"
            )
        if self.high - AXIOM_CAP_MARGIN < axiom_cap - tolerance:
            raise ConfigError(
                f"verifier.thresholds.{modality}.high must be >= axiom_cap + "
                f"{AXIOM_CAP_MARGIN} ({axiom_cap + AXIOM_CAP_MARGIN}); got {self.high}"
            )


@dataclass
class VerifierConfig:
    """Typed due-window verifier (Sec. 3.3, App. A.3)."""

    #: Backbone hidden width feeding the scoring head.
    hidden_dim: int = 2048
    #: Bottleneck width of the hidden -> head_dim -> 1 head.
    head_dim: int = 512
    #: Width of the modality embedding concatenated to the pooled representation.
    modality_embedding_dim: int = 32
    axiom_cap: float = AXIOM_CAP_DEFAULT
    thresholds: Dict[str, ModalityThresholds] = field(
        default_factory=lambda: {m: ModalityThresholds() for m in MODALITIES}
    )
    #: Path to the scoring head checkpoint. Never a repository-internal default; without
    #: one the runtime falls back to a zero-shot verifier and says so in the run record.
    head_path: Optional[str] = None
    #: Path to a JSON file carrying calibrated per-modality bands and the retraction
    #: scale. The bands decide which scores count as a refutation, so they belong with
    #: the head they were calibrated against.
    thresholds_path: Optional[str] = None

    def head_geometry(self) -> Dict[str, int]:
        """Geometry used when a head checkpoint carries no metadata of its own."""
        return {
            "hidden_dim": self.hidden_dim,
            "head_dim": self.head_dim,
            "modality_embedding_dim": self.modality_embedding_dim,
        }

    def validate(self) -> None:
        if self.hidden_dim < 1 or self.head_dim < 1:
            raise ConfigError("verifier head dimensions must be positive")
        if self.modality_embedding_dim < 1:
            raise ConfigError("verifier.modality_embedding_dim must be positive")
        if not 0.0 < self.axiom_cap < 1.0:
            raise ConfigError("verifier.axiom_cap must lie in (0, 1)")
        missing = [m for m in MODALITIES if m not in self.thresholds]
        if missing:
            raise ConfigError(f"verifier.thresholds missing modalities: {missing}")
        for modality, band in self.thresholds.items():
            if modality not in MODALITIES:
                raise ConfigError(f"unknown verifier modality {modality!r}")
            band.validate(modality, self.axiom_cap)


# --------------------------------------------------------------------------------------
# Retraction
# --------------------------------------------------------------------------------------


@dataclass
class RetractionConfig:
    """Retraction algebra (Sec. 3.3 Eq. (5), App. A.4, Table 3)."""

    #: rho_min, the reliability floor.
    rho_min: float = 0.05
    #: kappa. ``None`` means derive it from rho_min and a held-out margin quantile.
    kappa: Optional[float] = None
    #: Upper bound on kappa from Table 3.
    kappa_cap: float = 4.0

    def validate(self) -> None:
        if not 0.0 < self.rho_min < 1.0:
            raise ConfigError("retraction.rho_min must lie in (0, 1)")
        if self.kappa is not None and self.kappa <= 0:
            raise ConfigError("retraction.kappa must be positive when set")
        if self.kappa_cap <= 0:
            raise ConfigError("retraction.kappa_cap must be positive")

    def resolved_kappa(self, margin_quantile_value: Optional[float] = None) -> float:
        """Return kappa, deriving it from Table 3 when it is not pinned.

        Implements Table 3: ``kappa = min(kappa_cap, ln(1 / rho_min) / q_0.95(gamma))``.
        """
        if self.kappa is not None:
            return float(self.kappa)
        if not margin_quantile_value or margin_quantile_value <= 0:
            # Without held-out refutations the cap is the only defined value.
            return float(self.kappa_cap)
        derived = math.log(1.0 / self.rho_min) / float(margin_quantile_value)
        return float(min(self.kappa_cap, derived))


# --------------------------------------------------------------------------------------
# Guidance and decoding
# --------------------------------------------------------------------------------------


@dataclass
class GuidanceConfig:
    """Verdict-conditioned continuation (Sec. 3.3 Eq. (6), App. A.5, Table 3)."""

    #: lambda_0, the base guidance scale. The paper reports guidance on by default.
    lambda_0: float = 1.0

    def validate(self) -> None:
        if self.lambda_0 < 0:
            raise ConfigError("guidance.lambda_0 must be non-negative")


@dataclass
class DecodingConfig:
    """Decoding settings shared by the state, forecast, gate and answer lanes."""

    #: The verdict bias is an additive float mask; only these backends accept one.
    attention_backend: str = "sdpa"
    grammar: bool = True
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens_state: int = 224
    max_new_tokens_forecast: int = 96
    max_new_tokens_answer: int = 128
    seed: int = 0

    def validate(self) -> None:
        if self.attention_backend not in {"sdpa", "eager"}:
            raise ConfigError(
                "decoding.attention_backend must be 'sdpa' or 'eager': the verdict "
                "bias of Eq. (12) is a 4-D floating-point mask that fused kernels "
                "such as flash_attention_2 reject"
            )
        if self.temperature < 0:
            raise ConfigError("decoding.temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ConfigError("decoding.top_p must lie in (0, 1]")


# --------------------------------------------------------------------------------------
# Gate, forecaster, policy
# --------------------------------------------------------------------------------------


@dataclass
class GateConfig:
    """Answer gate (Sec. 3.4, Eq. (16))."""

    #: When False, only the hard registry rule of Eq. (16) applies, which is the matched
    #: hard-support-gate control.
    learned_head_enabled: bool = True
    #: Decision threshold on the scalar sufficiency head.
    threshold: float = 0.5
    #: Path to the gate head checkpoint.
    head_path: Optional[str] = None

    def validate(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ConfigError("gate.threshold must lie in (0, 1)")


@dataclass
class ForecasterConfig:
    """Forecaster lane (Sec. 3.2, App. A.2, A.5)."""

    #: Path to the forecast soft-prompt checkpoint. Its length comes from the checkpoint.
    prompt_path: Optional[str] = None
    #: The forecast pass always runs with the policy adapter disabled (App. A.5).
    disable_policy_adapter: bool = True

    def validate(self) -> None:
        if not self.disable_policy_adapter:
            raise ConfigError(
                "forecaster.disable_policy_adapter must stay True: App. A.5 specifies "
                "the forecaster uses its learned prompt with the policy adapter disabled"
            )


@dataclass
class PolicyConfig:
    """State policy configuration.

    The adapter's own geometry travels with its checkpoint, so nothing about it is
    configured here.
    """

    #: Path to the policy adapter checkpoint.
    adapter_path: Optional[str] = None

    def validate(self) -> None:
        return None


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Backbone selection. The backbone is frozen throughout (Sec. 3.5)."""

    #: Registered backend name; see :func:`ost.models.backbone.available_backbones`.
    backend: str = "qwen_omni"
    #: Filesystem path or hub identifier. Supplied by CLI, YAML or environment.
    path: Optional[str] = None
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    #: Qwen3-Omni's MoE residuals break under automatic device sharding, so a single
    #: device is the default. Set explicitly if a custom map is known to work.
    device_map: Optional[str] = None
    trust_remote_code: bool = False
    use_audio_in_video: bool = True

    def validate(self) -> None:
        if not self.backend:
            raise ConfigError("model.backend must be set")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ConfigError(f"unsupported model.dtype {self.dtype!r}")

    def resolved_path(self) -> str:
        path = self.path or os.environ.get("OST_MODEL_PATH")
        if not path:
            raise ConfigError(
                "No backbone path configured. Pass --model_path, set model.path in the "
                "YAML config, or export OST_MODEL_PATH."
            )
        return path


# --------------------------------------------------------------------------------------
# Root configuration
# --------------------------------------------------------------------------------------


@dataclass
class OSTConfig:
    """Root runtime configuration.

    Defaults reproduce the paper's default configuration table.
    """

    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    claims: ClaimConfig = field(default_factory=ClaimConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    retraction: RetractionConfig = field(default_factory=RetractionConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    forecaster: ForecasterConfig = field(default_factory=ForecasterConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # -- validation ---------------------------------------------------------------

    def validate(self) -> "OSTConfig":
        for section in (
            self.streaming,
            self.claims,
            self.retention,
            self.memory,
            self.verifier,
            self.retraction,
            self.guidance,
            self.decoding,
            self.gate,
            self.forecaster,
            self.policy,
            self.model,
        ):
            section.validate()
        self._validate_cross_section()
        return self

    def _validate_cross_section(self) -> None:
        chunk = self.streaming.decision_interval_seconds
        delta_max = self.claims.max_window_seconds
        delta_prime = self.claims.retry_slack_seconds

        # Implements the App. A.1 constraint K1 * Delta_chunk >= delta_max + delta',
        # which keeps a claim's Omni-State at level 1 until the claim settles or expires.
        k1 = self.memory.pyramid.capacities[0]
        if k1 * chunk < delta_max + delta_prime - 1e-9:
            raise ConfigError(
                "pyramid level-1 capacity is too small to hold a claim's Omni-State "
                f"until settlement: K1 * decision_interval = {k1 * chunk} s < "
                f"delta_max + delta_prime = {delta_max + delta_prime} s"
            )

        # App. A.1: both perceptual buffers keep delta_max + delta' + Delta_chunk at
        # full density.
        required_floor = delta_max + delta_prime + chunk
        if self.retention.floor_seconds + 1e-9 < required_floor:
            raise ConfigError(
                "retention.floor_seconds is below the perceptual retention floor of "
                f"delta_max + delta_prime + decision_interval = {required_floor} s; "
                f"got {self.retention.floor_seconds} s"
            )

        for duration in self.claims.window_durations:
            ratio = duration / chunk
            if abs(ratio - round(ratio)) > 1e-9:
                raise ConfigError(
                    f"claim window duration {duration} s is not a multiple of the "
                    f"decision interval {chunk} s, so its review would not fall on the "
                    "decision grid"
                )

        if self.retention.state_window_seconds > self.retention.floor_seconds:
            raise ConfigError(
                "retention.state_window_seconds exceeds retention.floor_seconds"
            )
        if self.retention.answer_window_seconds > self.retention.floor_seconds:
            raise ConfigError(
                "retention.answer_window_seconds exceeds retention.floor_seconds"
            )

    # -- serialisation ------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=False)

    # -- construction -------------------------------------------------------------

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OSTConfig":
        return _build_dataclass(cls, payload)

    @classmethod
    def load(
        cls,
        path: Optional[str | os.PathLike[str]] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> "OSTConfig":
        """Load a configuration from YAML with optional dotted-key overrides."""
        payload: Dict[str, Any] = {}
        if path is not None:
            payload = _read_config_file(Path(path))
        if overrides:
            payload = _deep_merge(payload, _expand_dotted(overrides))
        return cls.from_dict(payload).validate()

    def merged(self, overrides: Mapping[str, Any]) -> "OSTConfig":
        merged = _deep_merge(self.to_dict(), _expand_dotted(overrides))
        return OSTConfig.from_dict(merged).validate()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _read_config_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    text = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"configuration file {path} must contain a mapping")
    base = loaded.pop("_base_", None)
    if base is None:
        return loaded
    bases = [base] if isinstance(base, str) else list(base)
    merged: Dict[str, Any] = {}
    for entry in bases:
        merged = _deep_merge(merged, _read_config_file((path.parent / entry).resolve()))
    return _deep_merge(merged, loaded)


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {k: copy.deepcopy(v) for k, v in base.items()}
    for key, value in update.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _expand_dotted(overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Turn ``{"guidance.lambda_0": 0.5}`` into a nested mapping."""
    out: Dict[str, Any] = {}
    for key, value in overrides.items():
        if value is None:
            continue
        cursor = out
        parts = str(key).split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ConfigError(f"conflicting override for {key!r}")
        cursor[parts[-1]] = value
    return out


def _build_dataclass(cls: type, payload: Mapping[str, Any]) -> Any:
    if not dataclasses.is_dataclass(cls):
        raise ConfigError(f"{cls!r} is not a dataclass")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(payload) - set(fields)
    if unknown:
        raise ConfigError(
            f"unknown configuration key(s) for {cls.__name__}: {sorted(unknown)}"
        )
    kwargs: Dict[str, Any] = {}
    for name, spec in fields.items():
        if name not in payload:
            continue
        kwargs[name] = _coerce(spec.type, payload[name], f"{cls.__name__}.{name}")
    return cls(**kwargs)


_DATACLASS_REGISTRY: Dict[str, type] = {}


def _resolve_annotation(annotation: Any) -> Any:
    """Resolve string annotations produced by ``from __future__ import annotations``."""
    if not isinstance(annotation, str):
        return annotation
    if not _DATACLASS_REGISTRY:
        module = globals()
        for value in list(module.values()):
            if dataclasses.is_dataclass(value) and isinstance(value, type):
                _DATACLASS_REGISTRY[value.__name__] = value
    text = annotation.strip()
    return _DATACLASS_REGISTRY.get(text, annotation)


def _coerce(annotation: Any, value: Any, where: str) -> Any:
    annotation = _resolve_annotation(annotation)

    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where} must be a mapping")
        return _build_dataclass(annotation, value)

    text = annotation if isinstance(annotation, str) else ""

    if text.startswith("Dict[str, ModalityThresholds]"):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where} must be a mapping")
        return {
            str(k): _build_dataclass(ModalityThresholds, v)
            if isinstance(v, Mapping)
            else v
            for k, v in value.items()
        }
    if text.startswith("Optional[") and value is None:
        return None
    if text.startswith("Tuple[") or text.startswith("Tuple "):
        if isinstance(value, (list, tuple)):
            return tuple(value)
        raise ConfigError(f"{where} must be a sequence")
    return value


def default_config() -> OSTConfig:
    """Paper-default configuration, validated."""
    return OSTConfig().validate()


__all__ = [
    "AXIOM_CAP_DEFAULT",
    "AXIOM_CAP_MARGIN",
    "ClaimConfig",
    "ConfigError",
    "DecodingConfig",
    "ForecasterConfig",
    "GateConfig",
    "GuidanceConfig",
    "MemoryConfig",
    "ModalityThresholds",
    "ModelConfig",
    "OSTConfig",
    "PolicyConfig",
    "PyramidConfig",
    "RetentionConfig",
    "RetentionScoreWeights",
    "RetractionConfig",
    "StreamingConfig",
    "VerifierConfig",
    "default_config",
]
