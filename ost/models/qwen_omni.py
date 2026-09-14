"""Qwen3-Omni backbone.

The backbone is frozen. This wrapper adds the three things the method needs from it: a
verdict-reliability attention bias on reasoning keys, per-lane adapter and soft-prompt
scoping, and token-level logits for the two-branch guided decoding rule.

The verdict bias is applied by injecting an additive term into the attention mask of the
text decoder. Because a silently ignored mask would turn retraction into a no-op while
everything still appeared to run, :meth:`QwenOmniBackbone.probe_verdict_bias` checks
empirically that the bias changes the logits, and the runtime refuses to proceed if it
does not.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch

from ost.models.backbone import (
    Backbone,
    BackboneError,
    LaneRequest,
    MediaWindow,
    register_backbone,
)
from ost.retraction.attention_bias import (
    SpanTokenRange,
    has_attenuation,
    reliability_bias_vector,
)
from ost.types import GenerationStage, Modality

LOGGER = logging.getLogger(__name__)

#: Attention backends that accept a floating-point additive mask. Fused kernels do not.
SUPPORTED_ATTENTION = ("sdpa", "eager")

#: How many processed lane inputs to keep. Guided decoding needs the two branches of the
#: current chunk; older entries only hold media tensors alive.
_INPUT_CACHE_SIZE = 4


@dataclass
class _BiasState:
    """The bias currently installed on the attention modules."""

    key_bias: Optional[torch.Tensor] = None
    applications: int = 0


class QwenOmniBackbone(Backbone):
    """Frozen Qwen3-Omni thinker with OST's lane scoping and verdict bias."""

    def __init__(
        self,
        model_config,  # noqa: ANN001 - ost.config.ModelConfig
        *,
        decoding_config=None,  # noqa: ANN001 - ost.config.DecodingConfig
        forecaster_prompt_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
    ) -> None:
        model_config.validate()
        self.config = model_config
        self.decoding = decoding_config
        self.model_path = model_config.resolved_path()
        self.attn_implementation = (
            decoding_config.attention_backend if decoding_config is not None else "sdpa"
        )
        if self.attn_implementation not in SUPPORTED_ATTENTION:
            raise BackboneError(
                f"attention backend {self.attn_implementation!r} cannot accept the "
                "additive 4-D verdict bias of Eq. (12); use 'sdpa' or 'eager'"
            )
        self.adapter_path = adapter_path
        self.forecaster_prompt_path = forecaster_prompt_path

        self._model = None
        self._processor = None
        self._tokenizer = None
        self._bias = _BiasState()
        self._hook_handles: List[Any] = []
        self._policy_adapter_enabled = True
        self._soft_prompts: Dict[GenerationStage, Optional[torch.Tensor]] = {}
        self._grammar_compiler = None
        self._grammar_cache: Dict[str, Any] = {}
        self._grammar_unavailable = False
        self._input_cache: Dict[Any, Tuple[Dict[str, Any], int]] = {}
        self.load_info: Dict[str, Any] = {}

    # -- loading ---------------------------------------------------------------------

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration

        dtype = getattr(torch, self.config.dtype)
        LOGGER.info("loading backbone from %s (%s)", self.model_path, self.config.dtype)

        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=self.config.trust_remote_code
        )
        self._tokenizer = getattr(self._processor, "tokenizer", self._processor)

        # A single device is the default: this checkpoint's mixture-of-experts residuals
        # break under automatic sharding, producing cross-device errors mid-generation.
        device_map = self.config.device_map or self.config.device
        self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.model_path,
            trust_remote_code=self.config.trust_remote_code,
            device_map=device_map,
            dtype=dtype,
            attn_implementation=self.attn_implementation,
            experts_implementation="eager",
        )
        self._model.eval()
        # The talker synthesises speech; OST never uses it and it costs memory.
        if hasattr(self._model, "disable_talker"):
            self._model.disable_talker()

        # Sec. 3.5: the backbone is frozen throughout.
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

        self.load_info = {
            "model_path": self.model_path,
            "dtype": self.config.dtype,
            "attn_implementation": self.attn_implementation,
            "device": str(self.device),
            "hidden_size": self.hidden_size,
            "adapter_applied": False,
            "forecast_prompt_loaded": False,
        }

        if self.adapter_path:
            self._apply_adapter(Path(self.adapter_path))
        if self.forecaster_prompt_path:
            self._load_soft_prompt(GenerationStage.FORECAST, Path(self.forecaster_prompt_path))
        self._install_bias_hooks()

    def unload(self) -> None:
        self._input_cache.clear()
        self._remove_bias_hooks()
        self._model = None
        self._processor = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _require_model(self):
        if self._model is None:
            self.load()
        return self._model

    @property
    def thinker(self):
        """The text decoder. Scored forwards and guided decoding must call this.

        The top-level conditional-generation wrapper has no ``forward`` of its own, so
        calling it directly fails; the thinker is the module that accepts an attention
        mask and returns logits.
        """
        model = self._require_model()
        return getattr(model, "thinker", model)

    # -- capabilities -----------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        embedding = self.thinker.get_input_embeddings()
        return int(embedding.embedding_dim)

    @property
    def device(self) -> torch.device:
        return next(self._require_model().parameters()).device

    # -- tokenisation ------------------------------------------------------------------

    def tokenize(self, text: str) -> List[int]:
        self._require_model()
        return list(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def detokenize(self, token_ids: Sequence[int]) -> str:
        self._require_model()
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=True)

    @property
    def eos_token_ids(self) -> Tuple[int, ...]:
        self._require_model()
        ids: List[int] = []
        for candidate in (
            getattr(self._tokenizer, "eos_token_id", None),
            getattr(self._model.generation_config, "eos_token_id", None),
        ):
            if candidate is None:
                continue
            if isinstance(candidate, (list, tuple)):
                ids.extend(int(c) for c in candidate)
            else:
                ids.append(int(candidate))
        return tuple(dict.fromkeys(ids))

    # -- verdict bias -------------------------------------------------------------------

    #: Module-name fragments identifying the perceptual towers. App. A.4 leaves perceptual
    #: attention unchanged, and these towers also take a differently-shaped mask, so biasing
    #: them would be both wrong and a runtime error.
    _PERCEPTUAL_TOWERS = ("audio_tower", "visual", "vision", "talker", "code2wav")

    def _attention_modules(self) -> List[Any]:
        """The text decoder's self-attention modules, excluding the perceptual towers."""
        import inspect

        modules = []
        skipped_towers = 0
        skipped_signature = 0
        for name, module in self.thinker.named_modules():
            if not name.endswith("self_attn") or not hasattr(module, "forward"):
                continue
            lowered = name.lower()
            if any(tower in lowered for tower in self._PERCEPTUAL_TOWERS):
                skipped_towers += 1
                continue
            # The mask is injected as a keyword argument, so the module's forward has to name
            # it explicitly. A module that only receives it through **kwargs would forward it
            # twice to the attention interface.
            try:
                parameters = inspect.signature(module.forward).parameters
            except (TypeError, ValueError):
                skipped_signature += 1
                continue
            entry = parameters.get("attention_mask")
            if entry is None or entry.kind is inspect.Parameter.VAR_KEYWORD:
                skipped_signature += 1
                continue
            modules.append(module)

        if not modules:
            raise BackboneError(
                "could not locate the text decoder's self-attention modules, so the verdict "
                "bias of Eq. (12) cannot be applied"
            )
        LOGGER.debug(
            "verdict bias targets %d text self-attention module(s); skipped %d perceptual "
            "and %d with an incompatible signature",
            len(modules),
            skipped_towers,
            skipped_signature,
        )
        return modules

    def _install_bias_hooks(self) -> None:
        """Install pre-hooks that add the verdict bias to each attention mask.

        Hooking the attention modules keeps the bias independent of how a given release
        builds its causal mask. The hook has to be able to *create* a mask, not only extend
        one: with an all-ones 2-D mask the decoder takes a mask-free causal fast path, and a
        hook that only edited an existing mask would silently do nothing.
        """
        self._remove_bias_hooks()
        state = self._bias

        def pre_hook(module, args, kwargs):  # noqa: ANN001
            bias = state.key_bias
            if bias is None:
                return None

            hidden = kwargs.get("hidden_states")
            if hidden is None and args:
                hidden = args[0]
            if not torch.is_tensor(hidden) or hidden.dim() < 2:
                return None
            batch, q_len = int(hidden.shape[0]), int(hidden.shape[1])

            mask = kwargs.get("attention_mask")
            mask_in_args = False
            if mask is None and len(args) >= 3 and torch.is_tensor(args[2]):
                mask, mask_in_args = args[2], True

            past_len = _past_length(kwargs.get("past_key_values"), kwargs.get("past_key_value"))
            kv_len = past_len + q_len
            if mask is not None and torch.is_tensor(mask):
                kv_len = int(mask.shape[-1])

            dtype = hidden.dtype if hidden.is_floating_point() else torch.float32
            addition = torch.zeros(kv_len, device=hidden.device, dtype=dtype)
            width = min(kv_len, int(bias.shape[0]))
            addition[:width] = bias[:width].to(device=hidden.device, dtype=dtype)

            if mask is None:
                # No mask means the decoder was going to run a mask-free causal attention,
                # so the causal structure has to be reconstructed here alongside the bias.
                updated = _causal_additive_mask(
                    q_len, kv_len, past_len, device=hidden.device, dtype=dtype
                )
                updated = updated + addition.view(1, 1, 1, kv_len)
                updated = updated.expand(batch, 1, q_len, kv_len)
            else:
                base = mask
                if not torch.is_floating_point(base):
                    # A boolean keep-mask becomes an additive mask so the bias composes.
                    base = torch.zeros(
                        base.shape, device=base.device, dtype=dtype
                    ).masked_fill(~base.bool(), torch.finfo(dtype).min)
                while base.dim() < 4:
                    base = base.unsqueeze(1)
                updated = base.to(dtype) + addition.view(1, 1, 1, kv_len)

            state.applications += 1
            if mask_in_args:
                new_args = list(args)
                new_args[2] = updated
                return tuple(new_args), kwargs
            kwargs["attention_mask"] = updated
            return args, kwargs

        for module in self._attention_modules():
            self._hook_handles.append(
                module.register_forward_pre_hook(pre_hook, with_kwargs=True)
            )
        LOGGER.debug("installed %d verdict-bias hooks", len(self._hook_handles))

    def _remove_bias_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    @contextlib.contextmanager
    def verdict_bias(
        self,
        span_ranges: Sequence[SpanTokenRange],
        *,
        offset: int,
        kv_len: int,
    ) -> Iterator[int]:
        """Install the Eq. (12) bias for the duration of the block.

        ``offset`` is where the serialised ledger starts inside the model's input, so span
        offsets computed against the ledger land on the right keys.
        """
        if not span_ranges or not has_attenuation(span_ranges):
            yield 0
            return
        shifted = [
            SpanTokenRange(
                span_id=span.span_id,
                start=span.start + offset,
                end=span.end + offset,
                effective_reliability=span.effective_reliability,
            )
            for span in span_ranges
        ]
        bias = reliability_bias_vector(
            kv_len,
            shifted,
            reasoning_range=(offset, kv_len),
            device=self.device,
            dtype=torch.float32,
        )
        previous = self._bias.key_bias
        before = self._bias.applications
        self._bias.key_bias = bias
        try:
            yield self._bias.applications
        finally:
            self._bias.key_bias = previous
            LOGGER.debug(
                "verdict bias applied in %d attention call(s)",
                self._bias.applications - before,
            )

    def probe_verdict_bias(self) -> Dict[str, Any]:
        """Check empirically that the verdict bias changes the logits.

        Retraction is the mechanism most able to fail invisibly: if the additive mask is
        dropped, everything still runs and only the numbers are wrong. This probe compares
        logits with and without an aggressive bias and reports whether they differ.
        """
        self._require_model()
        text = (
            "Reasoning ledger. Earlier claim: the announcement will confirm Gate 6. "
            "New evidence: the announcement names Gate 12. The gate is"
        )
        token_ids = self.tokenize(text)
        if len(token_ids) < 8:
            raise BackboneError("probe text tokenised to too few tokens")
        input_ids = torch.tensor([token_ids], device=self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            baseline = self._decoder_logits(input_ids, attention_mask)
            spans = [
                SpanTokenRange(
                    span_id=1,
                    start=0,
                    end=max(4, len(token_ids) // 2),
                    effective_reliability=0.01,
                )
            ]
            with self.verdict_bias(spans, offset=0, kv_len=len(token_ids)) as _:
                biased = self._decoder_logits(input_ids, attention_mask)
            applications = self._bias.applications

        delta = float((baseline - biased).abs().max())
        applied = delta > 1e-4
        result = {
            "applied": applied,
            "max_logit_delta": delta,
            "attention_calls_biased": applications,
            "hooks": len(self._hook_handles),
        }
        if not applied:
            LOGGER.error(
                "verdict bias probe failed: logits are unchanged, so Eq. (12) would be a "
                "no-op. Check that the attention backend is 'sdpa' or 'eager' and that "
                "the decoder receives a floating-point additive mask."
            )
        return result

    def _decoder_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Last-position logits from the text decoder."""
        outputs = self.thinker(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        return logits[0, -1].float()

    # -- inputs -------------------------------------------------------------------------

    def _media_content(self, media: Optional[MediaWindow]) -> List[Dict[str, Any]]:
        """Build the chat content list for a media window.

        Audio rides inside the clipped video segment. A separately supplied episode-level
        audio track would span the whole clip and leak future evidence, so it is never
        attached.
        """
        if media is None or media.is_empty:
            return []
        if media.clip_path:
            # One clip for the whole dense window. Preferred: a per-second clip holds a
            # single frame, which the preprocessor rejects.
            return [{"type": "video", "video": media.clip_path}]
        clips: List[str] = []
        for slot in media.visual:
            payload = slot.payload
            if isinstance(payload, Mapping) and payload.get("clip"):
                path = str(payload["clip"])
                if path not in clips:
                    clips.append(path)
        return [{"type": "video", "video": path} for path in clips]

    def _build_inputs(self, request: LaneRequest) -> Tuple[Dict[str, Any], int]:
        """Tokenise a lane request. Returns the model inputs and the ledger token offset.

        Results are cached on the prompt, the ledger text and the media window. Guided decoding
        evaluates two branches per generated token, and each evaluation would otherwise
        re-decode the media: for a four-second window that is hundreds of redundant video
        decodes per chunk, which makes the guidance path unusably slow rather than merely twice
        as expensive as the method requires.
        """
        cache_key = (
            request.prompt,
            request.ledger_text,
            request.media.clip_path if request.media is not None else None,
            request.media.t_start if request.media is not None else None,
            request.media.t_end if request.media is not None else None,
        )
        cached = self._input_cache.get(cache_key)
        if cached is not None:
            inputs, offset = cached
            return dict(inputs), offset

        model = self._require_model()
        prompt_text = request.prompt
        ledger_text = request.ledger_text or ""

        content = self._media_content(request.media)
        combined = f"{prompt_text}\n\n{ledger_text}" if ledger_text else prompt_text
        content.append({"type": "text", "text": combined})
        messages = [{"role": "user", "content": content}]

        text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        kwargs: Dict[str, Any] = {"text": text, "return_tensors": "pt", "padding": True}

        if len(content) > 1:
            # The preprocessor's reader is selected by an environment variable. `decord` is the
            # default because the torchvision reader omits the frame rate for some containers,
            # which surfaces as a KeyError deep inside the preprocessor. Set
            # OST_VIDEO_READER=torchvision if decord livelocks on a particular corpus.
            os.environ.setdefault(
                "FORCE_QWENVL_VIDEO_READER",
                os.environ.get("OST_VIDEO_READER", "decord"),
            )
            # The multimodal preprocessor decodes audio through audioread, which shells out
            # to a bare `ffmpeg` and ignores OST_FFMPEG. Put its directory on PATH so an
            # ffmpeg that is installed but not on PATH still works.
            _ensure_ffmpeg_on_path()
            from qwen_omni_utils import process_mm_info

            audios, images, videos = process_mm_info(
                messages, use_audio_in_video=self.config.use_audio_in_video
            )
            kwargs.update(
                {
                    "audio": audios,
                    "images": images,
                    "videos": videos,
                    "use_audio_in_video": self.config.use_audio_in_video,
                }
            )
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        inputs = self._processor(**kwargs)
        inputs = inputs.to(model.device)
        # The processor emits float32 features while the checkpoint is bf16.
        param_dtype = next(model.parameters()).dtype
        for key, value in list(inputs.items()):
            if torch.is_tensor(value) and value.is_floating_point():
                inputs[key] = value.to(param_dtype)

        # Where the ledger starts in the tokenised prompt, so span offsets line up.
        offset = 0
        if ledger_text:
            head = text.split(ledger_text)[0] if ledger_text in text else prompt_text
            offset = len(self.tokenize(head))

        if len(self._input_cache) >= _INPUT_CACHE_SIZE:
            # Only the two branches of the current chunk need to coexist; a small bound stops
            # cached media tensors accumulating across chunks.
            self._input_cache.clear()
        self._input_cache[cache_key] = (dict(inputs), offset)
        return dict(inputs), offset

    # -- generation ---------------------------------------------------------------------

    def next_token_logits(
        self, request: LaneRequest, generated: Sequence[int]
    ) -> torch.Tensor:
        """Logits for the next token, with the verdict bias in force."""
        model = self._require_model()
        inputs, offset = self._build_inputs(request)
        input_ids = inputs["input_ids"]
        if generated:
            extra = torch.tensor([list(generated)], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, extra], dim=1)
            inputs["input_ids"] = input_ids
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.ones_like(input_ids)

        kv_len = int(input_ids.shape[1])
        forward_kwargs = {
            key: value
            for key, value in inputs.items()
            if key not in ("use_audio_in_video",)
        }
        forward_kwargs.setdefault("attention_mask", torch.ones_like(input_ids))

        with self._lane_scope(request.stage):
            with self.verdict_bias(request.span_ranges, offset=offset, kv_len=kv_len):
                with torch.inference_mode():
                    outputs = self.thinker(
                        **forward_kwargs, use_cache=False, return_dict=True
                    )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        return logits[0, -1].float()

    def generate(self, request: LaneRequest) -> str:
        """Greedy generation for a lane, with the verdict bias in force."""
        model = self._require_model()
        inputs, offset = self._build_inputs(request)
        kv_len = int(inputs["input_ids"].shape[1])
        prompt_length = kv_len

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(request.max_new_tokens),
            "do_sample": False,
            "return_audio": False,
        }
        if self.decoding is not None and self.decoding.temperature > 0:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": float(self.decoding.temperature),
                    "top_p": float(self.decoding.top_p),
                }
            )

        # Grammar-constrained slot filling. Without it the state and forecast lanes emit
        # prose that the parser has to discard, which shows up as an empty forecast field and
        # a loop that never registers a claim.
        processor = self._grammar_processor(request.json_schema)
        if processor is not None:
            gen_kwargs["logits_processor"] = [processor]

        with self._lane_scope(request.stage):
            with self.verdict_bias(request.span_ranges, offset=offset, kv_len=kv_len):
                with torch.inference_mode():
                    out = model.generate(**inputs, **gen_kwargs)
        if isinstance(out, (tuple, list)):
            out = out[0]
        trimmed = out[:, prompt_length:]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def _grammar_processor(self, schema: Optional[Mapping[str, Any]]):
        """Build a logits processor that constrains generation to ``schema``.

        Returns ``None`` when grammar is disabled or no schema was supplied. A missing or
        broken grammar backend is reported rather than silently skipped, because an
        unconstrained lane produces output the parser discards, which surfaces as an empty
        forecast field and a loop that never registers a claim.
        """
        if schema is None or (self.decoding is not None and not self.decoding.grammar):
            return None
        if self._grammar_unavailable:
            return None
        try:
            import xgrammar as xgr
        except ImportError:
            LOGGER.warning(
                "xgrammar is not installed, so the state and forecast lanes will be "
                "unconstrained and their output may not parse. Install it, or set "
                "decoding.grammar=false to accept unconstrained lanes."
            )
            self._grammar_unavailable = True
            return None

        try:
            if self._grammar_compiler is None:
                tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                    self._tokenizer, vocab_size=self._vocab_size()
                )
                self._grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
            key = json.dumps(dict(schema), sort_keys=True)
            compiled = self._grammar_cache.get(key)
            if compiled is None:
                compiled = self._grammar_compiler.compile_json_schema(key)
                self._grammar_cache[key] = compiled
            return xgr.contrib.hf.LogitsProcessor(compiled)
        except Exception as exc:  # noqa: BLE001 - a grammar backend failure must be visible
            LOGGER.warning(
                "grammar-constrained decoding is unavailable (%s: %s); lanes will be "
                "unconstrained for the rest of this run",
                type(exc).__name__,
                exc,
            )
            self._grammar_unavailable = True
            return None

    def _vocab_size(self) -> int:
        """Vocabulary width the grammar bitmask must cover.

        On this backbone the width lives on the nested text config, not the top-level one, and
        the padded embedding table can be wider than the tokenizer. The larger value is taken
        so the bitmask covers every logit position.
        """
        candidates = []
        config = self.thinker.config
        for holder in (config, getattr(config, "text_config", None)):
            value = getattr(holder, "vocab_size", None)
            if isinstance(value, int) and value > 0:
                candidates.append(value)
        embedding = self.thinker.get_input_embeddings()
        weight = getattr(embedding, "weight", None)
        if weight is not None:
            candidates.append(int(weight.shape[0]))
        if not candidates:
            raise BackboneError("could not determine the vocabulary size for the grammar")
        return max(candidates)

    # -- scoring -------------------------------------------------------------------------

    def pooled_representation(
        self,
        text: str,
        media: Optional[MediaWindow] = None,
        *,
        modality: Optional[Modality] = None,
    ) -> torch.Tensor:
        """Mean-pooled final hidden state, for the verifier and gate heads.

        App. A.3 runs the verifier with the policy LoRA disabled, so pooled
        representations are always taken outside the policy adapter.
        """
        request = LaneRequest(
            stage=GenerationStage.FORECAST,
            prompt=text,
            ledger_text="",
            media=media,
            max_new_tokens=1,
        )
        inputs, _ = self._build_inputs(request)
        forward_kwargs = {
            key: value for key, value in inputs.items() if key not in ("use_audio_in_video",)
        }
        forward_kwargs.setdefault("attention_mask", torch.ones_like(inputs["input_ids"]))

        with self.adapter_scope(policy_adapter=False):
            with torch.inference_mode():
                outputs = self.thinker(
                    **forward_kwargs,
                    use_cache=False,
                    return_dict=True,
                    output_hidden_states=True,
                )
        hidden = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else None
        if hidden is None:
            raise BackboneError("the decoder did not return hidden states")
        mask = forward_kwargs["attention_mask"].to(hidden.dtype)
        pooled = (hidden[0] * mask[0].unsqueeze(-1)).sum(dim=0) / mask[0].sum().clamp(min=1)
        return pooled.float()

    def sequence_log_prob(
        self,
        request: LaneRequest,
        target_text: str,
        *,
        length_normalized: bool = False,
    ) -> torch.Tensor:
        """Log probability of ``target_text``, optionally length-normalised.

        With ``length_normalized`` the result is ``log p_bar``, the geometric mean token
        log probability, which is what makes continuations of different lengths
        comparable.
        """
        model = self._require_model()
        inputs, offset = self._build_inputs(request)
        prompt_ids = inputs["input_ids"]
        target_ids = torch.tensor(
            [self.tokenize(target_text)], device=prompt_ids.device, dtype=prompt_ids.dtype
        )
        if target_ids.shape[1] == 0:
            return torch.zeros((), device=prompt_ids.device)

        full = torch.cat([prompt_ids, target_ids], dim=1)
        forward_kwargs = {
            key: value
            for key, value in inputs.items()
            if key not in ("use_audio_in_video", "input_ids", "attention_mask")
        }

        # A learned prefix is prepended in embedding space, so its length has to be added
        # to the key-value length the verdict bias is built against.
        soft_prompt = self._soft_prompts.get(request.stage)
        prefix_length = 0
        if soft_prompt is not None:
            embedding = self.thinker.get_input_embeddings()
            token_embeds = embedding(full)
            prefix = soft_prompt.to(device=token_embeds.device, dtype=token_embeds.dtype)
            if prefix.dim() == 2:
                prefix = prefix.unsqueeze(0)
            prefix = prefix.expand(full.shape[0], -1, -1)
            prefix_length = int(prefix.shape[1])
            forward_kwargs["inputs_embeds"] = torch.cat([prefix, token_embeds], dim=1)
            forward_kwargs["attention_mask"] = torch.ones(
                forward_kwargs["inputs_embeds"].shape[:2],
                device=full.device,
                dtype=torch.long,
            )
        else:
            forward_kwargs["input_ids"] = full
            forward_kwargs["attention_mask"] = torch.ones_like(full)

        kv_len = int(full.shape[1]) + prefix_length
        with self._lane_scope(request.stage):
            with self.verdict_bias(
                request.span_ranges, offset=offset + prefix_length, kv_len=kv_len
            ):
                outputs = self.thinker(**forward_kwargs, use_cache=False, return_dict=True)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # Predict target token t from position t-1, shifted past any learned prefix.
        start = prefix_length + int(prompt_ids.shape[1]) - 1
        window = logits[0, start : start + int(target_ids.shape[1])].float()
        log_probs = torch.log_softmax(window, dim=-1)
        gathered = log_probs.gather(-1, target_ids[0].unsqueeze(-1)).squeeze(-1)
        total = gathered.sum()
        if length_normalized:
            return total / float(target_ids.shape[1])
        return total

    # -- adapters and soft prompts ---------------------------------------------------------

    def _apply_adapter(self, path: Path) -> None:
        """Attach the policy adapter to the thinker, frozen."""
        from peft import PeftModel

        candidates = [path / "thinker_lora", path / "adapter", path]
        for candidate in candidates:
            if (candidate / "adapter_config.json").exists():
                target = getattr(self._model, "thinker", self._model)
                patched = PeftModel.from_pretrained(target, str(candidate), is_trainable=False)
                if target is not self._model:
                    self._model.thinker = patched
                else:
                    self._model = patched
                self.load_info["adapter_applied"] = True
                self.load_info["adapter_path"] = str(candidate)
                LOGGER.info("attached policy adapter from %s", candidate)
                return
        raise BackboneError(
            f"no adapter_config.json under {path} (looked in ./thinker_lora, ./adapter, .)"
        )

    def _load_soft_prompt(self, stage: GenerationStage, path: Path) -> None:
        from ost.models.soft_prompt import SoftPromptError, load_soft_prompt

        try:
            embeddings, _, hidden_size = load_soft_prompt(path)
        except SoftPromptError as exc:
            raise BackboneError(str(exc)) from exc
        if hidden_size != self.hidden_size:
            raise BackboneError(
                f"soft prompt width {hidden_size} does not match the decoder hidden "
                f"size {self.hidden_size}"
            )
        self._soft_prompts[stage] = embeddings
        if stage is GenerationStage.FORECAST:
            self.load_info["forecast_prompt_loaded"] = True

    def set_soft_prompt(
        self, stage: GenerationStage, embeddings: Optional[torch.Tensor]
    ) -> None:
        self._soft_prompts[stage] = embeddings

    @contextlib.contextmanager
    def adapter_scope(self, *, policy_adapter: bool) -> Iterator[None]:
        """Enable or disable the policy LoRA for the duration of the block."""
        thinker = getattr(self._require_model(), "thinker", None)
        previous = self._policy_adapter_enabled
        self._policy_adapter_enabled = bool(policy_adapter)
        if policy_adapter or thinker is None or not hasattr(thinker, "disable_adapter"):
            try:
                yield
            finally:
                self._policy_adapter_enabled = previous
            return
        try:
            with thinker.disable_adapter():
                yield
        finally:
            self._policy_adapter_enabled = previous

    @contextlib.contextmanager
    def _lane_scope(self, stage: GenerationStage) -> Iterator[None]:
        """Apply the lane's adapter policy.

        App. A.5: the state policy uses its LoRA, while the forecaster uses its learned
        prompt with that LoRA disabled.
        """
        use_policy = stage is not GenerationStage.FORECAST
        with self.adapter_scope(policy_adapter=use_policy):
            yield

    # -- diagnostics -------------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        out = {"backend": "qwen_omni"}
        out.update(self.load_info)
        return out


def _ensure_ffmpeg_on_path() -> None:
    """Make ``ffmpeg`` discoverable on ``PATH``.

    The multimodal preprocessor decodes audio via ``audioread``, which invokes a bare
    ``ffmpeg`` and does not consult ``OST_FFMPEG``. Without this, an ffmpeg that exists but is
    not on ``PATH`` produces an opaque decoder error deep inside a third-party package.
    """
    import shutil

    if shutil.which("ffmpeg"):
        return
    override = os.environ.get("OST_FFMPEG")
    if not override:
        raise BackboneError(
            "ffmpeg is not on PATH and OST_FFMPEG is unset, so the multimodal preprocessor "
            "cannot decode audio. Install ffmpeg or set OST_FFMPEG to its absolute path."
        )
    directory = str(Path(override).expanduser().resolve().parent)
    os.environ["PATH"] = f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"
    if not shutil.which("ffmpeg"):
        raise BackboneError(
            f"OST_FFMPEG points at {override}, but no ffmpeg is executable in {directory}"
        )


def _past_length(*caches: Any) -> int:
    """Number of cached key positions preceding the current query block."""
    for cache in caches:
        if cache is None:
            continue
        getter = getattr(cache, "get_seq_length", None)
        if callable(getter):
            try:
                return int(getter())
            except (TypeError, ValueError):
                continue
        if isinstance(cache, (tuple, list)) and cache:
            first = cache[0]
            if isinstance(first, (tuple, list)) and first and torch.is_tensor(first[0]):
                return int(first[0].shape[-2])
    return 0


def _causal_additive_mask(
    q_len: int,
    kv_len: int,
    past_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive causal mask of shape ``(1, 1, q_len, kv_len)``."""
    q_index = torch.arange(q_len, device=device).view(q_len, 1) + past_len
    k_index = torch.arange(kv_len, device=device).view(1, kv_len)
    mask = torch.zeros(q_len, kv_len, device=device, dtype=dtype)
    mask.masked_fill_(k_index > q_index, torch.finfo(dtype).min)
    return mask.view(1, 1, q_len, kv_len)


@register_backbone("qwen_omni")
def _build_qwen_omni(config, **kwargs) -> QwenOmniBackbone:  # noqa: ANN001
    return QwenOmniBackbone(config, **kwargs)


__all__ = ["SUPPORTED_ATTENTION", "QwenOmniBackbone"]
