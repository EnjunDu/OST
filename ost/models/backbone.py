"""Backbone interface.

The backbone is frozen throughout (Sec. 3.5): everything OST adds sits on top of this
interface as an adapter, a soft prompt or a small head. Stating the interface explicitly
is what keeps backend quirks out of the method code.
"""

from __future__ import annotations

import abc
import contextlib
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch

from ost.retraction.attention_bias import SpanTokenRange
from ost.streaming.retention import PerceptionSlot
from ost.types import GenerationStage, Modality


@dataclass
class MediaWindow:
    """The perceptual evidence handed to the backbone for one call.

    Only the retained slots inside the requested window are passed. The full media file is
    never handed over, which is what keeps the stream causal.
    """

    visual: List[PerceptionSlot] = field(default_factory=list)
    audio: List[PerceptionSlot] = field(default_factory=list)
    #: Absolute interval the window covers, for logging and cache keys.
    t_start: float = 0.0
    t_end: float = 0.0
    #: One clip covering the whole window. Preferred over the per-slot clips: a single
    #: one-second clip holds one frame, and the multimodal preprocessor requires at least
    #: two, so passing the dense window as one clip is both correct and necessary.
    clip_path: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not self.visual and not self.audio

    def slot_counts(self) -> Dict[str, int]:
        return {"visual": len(self.visual), "audio": len(self.audio)}


@dataclass
class LaneRequest:
    """A single generation or scoring request on one lane."""

    stage: GenerationStage
    #: Serialised prompt text, excluding the ledger.
    prompt: str
    #: Serialised reasoning ledger; its token offsets drive the verdict bias.
    ledger_text: str
    #: Token offsets of registered spans inside the serialised ledger.
    span_ranges: Sequence[SpanTokenRange] = ()
    media: Optional[MediaWindow] = None
    max_new_tokens: int = 128
    #: JSON schema for grammar-constrained decoding, when enabled.
    json_schema: Optional[Mapping[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Backbone(abc.ABC):
    """Frozen multimodal backbone plus the adapters OST runs on top of it."""

    # -- capabilities ---------------------------------------------------------------

    @property
    @abc.abstractmethod
    def hidden_size(self) -> int:
        """Width of the pooled representation the verifier and gate heads consume."""

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        ...

    @property
    def supports_verdict_bias(self) -> bool:
        """Whether this backbone can apply the additive bias of Eq. (12)."""
        return True

    # -- tokenisation ----------------------------------------------------------------

    @abc.abstractmethod
    def tokenize(self, text: str) -> List[int]:
        ...

    @abc.abstractmethod
    def detokenize(self, token_ids: Sequence[int]) -> str:
        ...

    @property
    @abc.abstractmethod
    def eos_token_ids(self) -> Tuple[int, ...]:
        ...

    # -- generation -------------------------------------------------------------------

    @abc.abstractmethod
    def next_token_logits(
        self,
        request: LaneRequest,
        generated: Sequence[int],
    ) -> torch.Tensor:
        """Logits of shape ``(vocab,)`` for the next token of ``request``.

        The implementation must apply the verdict bias derived from
        ``request.span_ranges`` to the reasoning-token keys only, leaving perceptual
        attention unchanged.
        """

    @abc.abstractmethod
    def generate(self, request: LaneRequest) -> str:
        """Unguided generation for a lane. Used when no contrast is needed."""

    # -- scoring ----------------------------------------------------------------------

    @abc.abstractmethod
    def pooled_representation(
        self,
        text: str,
        media: Optional[MediaWindow] = None,
        *,
        modality: Optional[Modality] = None,
    ) -> torch.Tensor:
        """Pooled hidden state of shape ``(hidden_size,)`` for a verifier or gate head."""

    @abc.abstractmethod
    def sequence_log_prob(
        self,
        request: LaneRequest,
        target_text: str,
        *,
        length_normalized: bool = False,
    ) -> torch.Tensor:
        """Log probability of ``target_text`` under the lane.

        With ``length_normalized`` the result is ``log p_bar = (1/|x|) sum log p``, the
        geometric mean token log probability, which is what lets continuations of
        different lengths be compared.
        """

    # -- adapters ----------------------------------------------------------------------

    @contextlib.contextmanager
    def adapter_scope(self, *, policy_adapter: bool) -> Iterator[None]:
        """Enable or disable the policy adapter for the duration of the block.

        App. A.5: the state policy uses its LoRA, while the forecaster uses its learned
        prompt with that LoRA disabled. The verifier pass likewise runs with the policy
        LoRA disabled (App. A.3).
        """
        yield

    def set_soft_prompt(self, stage: GenerationStage, embeddings: Optional[torch.Tensor]) -> None:
        """Install a learned soft prompt for a lane. No-op for backends without one."""
        return None

    # -- lifecycle ----------------------------------------------------------------------

    def load(self) -> None:
        """Materialise weights. Called lazily by the runtime."""
        return None

    def unload(self) -> None:
        return None

    def info(self) -> Dict[str, Any]:
        return {"backend": type(self).__name__, "hidden_size": self.hidden_size}


class BackboneError(RuntimeError):
    """Raised when the backbone cannot satisfy a request."""


_REGISTRY: Dict[str, Callable[..., Backbone]] = {}


def register_backbone(name: str) -> Callable[[Callable[..., Backbone]], Callable[..., Backbone]]:
    def decorator(factory: Callable[..., Backbone]) -> Callable[..., Backbone]:
        _REGISTRY[name] = factory
        return factory

    return decorator


def build_backbone(config, **kwargs) -> Backbone:  # noqa: ANN001
    """Instantiate the configured backbone.

    The import is deferred so that the heavy multimodal stack is only pulled in when a
    backend that needs it is actually requested.
    """
    name = config.backend
    if name not in _REGISTRY:
        # Import side effects register the built-in backend.
        if name == "qwen_omni":
            from ost.models import qwen_omni  # noqa: F401
    factory = _REGISTRY.get(name)
    if factory is None:
        raise BackboneError(
            f"unknown model.backend {name!r}; registered backends: {sorted(_REGISTRY)}"
        )
    return factory(config, **kwargs)


def available_backbones() -> List[str]:
    return sorted(_REGISTRY)


__all__ = [
    "Backbone",
    "BackboneError",
    "LaneRequest",
    "MediaWindow",
    "available_backbones",
    "build_backbone",
    "register_backbone",
]
