# Omni-Streaming Thinking (OST)

Inference implementation of **Omni-Streaming Thinking**, a streaming audio–visual
reasoning system.

A streaming model must decide *what* and *when* to answer from only the video chunks and
audio observed so far. Visual cues often support an interpretation before an utterance or
sound event is complete. If that interpretation enters memory as a fact, later reasoning
can keep repeating it even after audio contradicts it. OST addresses this by recording an
unresolved interpretation as a **claim** carrying four things: what is expected to occur,
which modality can verify it, when that evidence is due, and which states depend on it.
When the interval closes, the claim is checked against evidence from the specified
modality. A refutation reduces the influence of the claim *and its dependent states*, then
guides a corrected state update. Separate audio and visual retention preserves the evidence
needed for these checks, and an answer gate delays responses while answer-critical claims
await review.

- [1. Scope of this repository](#1-scope-of-this-repository)
- [2. How the loop runs](#2-how-the-loop-runs)
- [3. Repository structure](#3-repository-structure)
- [4. Installation](#4-installation)
- [5. The backbone](#5-the-backbone)
- [6. Checkpoints](#6-checkpoints)
- [7. Running inference](#7-running-inference)
- [8. Reading the run record](#8-reading-the-run-record)
- [9. Long videos](#9-long-videos)
- [10. Configuration](#10-configuration)
- [11. Troubleshooting](#11-troubleshooting)
- [12. Citation](#12-citation)

---

## 1. Scope of this repository

This release is **inference only**. It reads a configuration, loads whatever checkpoints
that configuration names, and runs the streaming loop over a video.

Not included: the training stages, the offline data construction they consume, the
benchmark evaluation harness, and the diagnostic benchmark. The method's inference-time
mechanism — claim admission, typed verification, retraction, verdict-conditioned decoding
and the answer gate — is here in full, and every quantity it depends on is a configurable
field rather than a constant at a call site.

The backbone is **frozen throughout**: no part of this code modifies a backbone parameter.
Everything OST adds is an adapter, a soft prompt, or a small head loaded on top of it.

## 2. How the loop runs

One decision chunk, in the order the code executes it
(`ost/policy/orchestrator.py`):

1. **Perceive.** The stream is read one perception step at a time and never past the
   current boundary (`ost/streaming/source.py`). Audio and visual slots land in separate
   retention buffers with separate budgets, so dense visual tokens can never displace the
   audio a pending claim needs (`ost/streaming/retention.py`).
2. **Review what is due.** Claims whose evidence interval has closed are verified against
   evidence projected from their *declared* modality only
   (`ost/verification/verifier.py`). A score is read against that modality's verdict band
   to give confirmed, refuted, or undecided; an interval with too little coverage is capped
   so confirmation stays unavailable where the evidence could not have been assessed.
3. **Retract.** A refutation lowers the refuted span's reliability and propagates to every
   span that depends on it, never amplifying anything (`ost/retraction/algebra.py`). The
   result is applied at decoding time as an additive log-reliability bias on the
   reasoning-token keys, leaving perceptual attention untouched
   (`ost/retraction/attention_bias.py`).
4. **Rewrite the state, guided.** When something was refuted this chunk, the state update
   is decoded from a contrast between a corrected context and a matched comparison context
   that omits this chunk's refutations (`ost/guidance/verdict_conditioned.py`). The state
   itself is six fields emitted under a grammar constraint (`ost/state/schema.py`).
5. **Forecast.** The forecast lane proposes new claims, which are admitted subject to the
   window, capacity and depth limits (`ost/forecasting/`).
6. **Gate.** The gate answers only if the represented evidence is sufficient *and* no
   answer-critical claim is still queued for review. The second check is a hard registry
   rule the learned head cannot override (`ost/gating/answer_gate.py`).
7. **Fold.** Older states are merged up a four-way pyramid, sized so a claim's state
   survives at full resolution until the claim settles (`ost/state/pyramid.py`).

Before any run, the runtime probes whether the verdict-reliability attention bias actually
changes the backbone's logits, and **refuses to proceed if it does not**. Retraction is the
mechanism most able to fail invisibly: if the additive mask were dropped, everything would
still run and only the numbers would be wrong.

## 3. Repository structure

```text
OST/
├── ost/
│   ├── config.py                 all configurable quantities, including the paper defaults
│   ├── types.py                  Omni-State, claim, span and verdict types
│   ├── runtime.py                assembles an orchestrator from a config plus its checkpoints
│   ├── cli.py                    the inference entry point
│   ├── streaming/                decision grid, causal stream source, separate A/V retention
│   ├── state/                    six-field Omni-State, output schemas, memory pyramid
│   ├── forecasting/              claim generation and the admission rule
│   ├── verification/             typed due-window verifier, scoring head, verdict bands
│   ├── retraction/               retraction algebra, span registry, attention bias
│   ├── guidance/                 verdict-conditioned two-branch decoding
│   ├── gating/                   the answer gate
│   ├── memory/                   the reasoning ledger and verdict log
│   ├── models/                   backbone interface, Qwen3-Omni wrapper, soft-prompt loader
│   ├── policy/                   the online inference loop and answer decoding
│   └── utils/                    seeding, logging, path resolution
├── configs/ost/                  runtime YAML
└── scripts/run_ost.sh            one-clip inference
```

## 4. Installation

Python 3.11 or newer, and a CUDA-capable GPU.

```bash
git clone <repository-url> OST
cd OST
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`ffmpeg` and `ffprobe` must be on `PATH`, or point at them explicitly:

```bash
export OST_FFMPEG=/path/to/ffmpeg
export OST_FFPROBE=/path/to/ffprobe
```

## 5. The backbone

No weights are included here. The backbone is a public release:

| Item | Value |
|---|---|
| Model | `Qwen3-Omni-30B-A3B-Instruct` |
| Family | omni-modal (text, image, audio, video), mixture-of-experts |
| Model class | `Qwen3OmniMoeForConditionalGeneration` |
| Precision | `bfloat16` |
| Pooled hidden width | 2048 |
| Attention implementation | `sdpa` or `eager` |
| Approximate size on disk | ~66 GB |

```bash
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir ../models/Qwen3-Omni-30B-A3B-Instruct
```

Point the code at it with `model.path` in the YAML, `--model_path`, or `OST_MODEL_PATH`.

Load in `bfloat16` on a single device. Automatic device sharding breaks this checkpoint's
mixture-of-experts residuals and produces cross-device errors part-way through generation,
which is why `model.device_map` is unset by default.

## 6. Checkpoints

Four components can be loaded, and each is optional:

| Config field | Directory contents | Without it |
|---|---|---|
| `verifier.head_path` | `verifier_head.pt`, and `verifier_head.json` if the checkpoint carries its geometry | a zero-shot verifier is used, and the run record says so |
| `verifier.thresholds_path` | a JSON file with per-modality verdict bands and `kappa` | the configured default bands apply |
| `gate.head_path` | `gate_head.pt` | only the hard registry rule of Eq. (16) applies |
| `forecaster.prompt_path` | `soft_prompt.pt` | the forecast lane runs without a learned prefix |
| `policy.adapter_path` | `adapter_config.json`, either directly or under `./thinker_lora` | the state policy runs on the unmodified backbone |

Name them in one place and a run needs no flags beyond the clip and the question.
[`configs/ost/with_checkpoints.yaml`](configs/ost/with_checkpoints.yaml) is the template:

```yaml
_base_: default.yaml

model:
  path: ../models/Qwen3-Omni-30B-A3B-Instruct

verifier:
  head_path: ./checkpoints/verifier
  thresholds_path: ./checkpoints/verifier/calibration.json
gate:
  head_path: ./checkpoints/gate
forecaster:
  prompt_path: ./checkpoints/forecaster
policy:
  adapter_path: ./checkpoints/policy
```

Relative paths resolve against the working directory. Every one of these may also be given
on the command line (`--verifier_head`, `--thresholds`, `--gate_head`,
`--forecast_prompt`, `--adapter`), and the command line wins.

A path that is **set but absent is an error**, not a silent fallback to the unmodified
component. A missing checkpoint that degrades quietly is the failure most likely to be
mistaken for a weak method.

## 7. Running inference

```bash
export OST_OUTPUT_DIR=../output          # must be outside this repository

bash scripts/run_ost.sh \
    --media clips/example.mp4 \
    --question "Which gate does the flight depart from?"
```

Or call the module directly, which is the same thing without the environment indirection:

```bash
python -m ost.cli infer \
    --config configs/ost/with_checkpoints.yaml \
    --media clips/example.mp4 \
    --question "Which gate does the flight depart from?" \
    --output_dir ../output
```

Useful flags:

| Flag | Effect |
|---|---|
| `--max_duration_seconds S` | truncate the clip, for a bounded first run |
| `--set KEY=VALUE` | override any config key, e.g. `--set guidance.lambda_0=0` |
| `--cache_dir DIR` | where extracted window clips go; defaults to the OS temp area |
| `--device` | backbone device, e.g. `cuda:1` |

To run several clips over several GPUs, give each its own device and output directory:

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i bash scripts/run_ost.sh \
      --media "clips/part${i}.mp4" --question "..." &
done
wait
```

## 8. Reading the run record

Each run writes `<clip name>.inference.json` into the output directory, named after the
clip so a sweep does not overwrite itself. Alongside the answer it carries a per-chunk
trace, which is the fastest way to see whether the loop is doing what you expect:

```json
{
  "answer": "Gate 12.",
  "stop_chunk": 3,
  "claims": 2,
  "refutations": 1,
  "runtime": {
    "verifier_head_loaded": true,
    "zero_shot_verifier": false,
    "calibrated_thresholds": true,
    "verdict_bias_probe": {"applied": true, "max_logit_delta": 0.47}
  },
  "trace": [
    {
      "chunk": 2,
      "t_end": 8.0,
      "verdicts": [{"claim": 1, "verdict": "refuted", "gamma": 0.71, "score": 0.10}],
      "refuted": [1],
      "guidance_scale": 0.71,
      "guidance_applied": true,
      "gate": "wait",
      "hard_wait": true
    }
  ]
}
```

Two fields are worth checking on any new clip. `runtime.zero_shot_verifier` tells you
whether the verifier was a fitted head or the fallback. `claims` and `refutations` at zero
mean the loop ran but never exercised the mechanism, so nothing about that run tests the
method: check that the forecast lane produces parseable output and that
`guidance.lambda_0` is not zero unless the no-guidance control was intended.

## 9. Long videos

The loop is designed for streams longer than a context window, and its cost is bounded by
configuration rather than by clip length:

- **Retention** keeps `retention.floor_seconds` at full density per modality and summarises
  beyond it, so the perceptual budget does not grow with duration.
- **The pyramid** folds older states four-way, so the reasoning context stays bounded.
  `memory.keep_archive: false` additionally drops the append-only archive, which is the
  setting to use if resident memory grows on a very long stream.
- **Guided decoding is the expensive part.** It evaluates two branches per generated token
  and this implementation recomputes the sequence rather than keeping a key-value cache, so
  a guided chunk costs `O(n^2)` forward passes. On a 30B backbone a long clip with many
  refutations may not finish. Reduce `decoding.max_new_tokens_state`, or run
  `configs/ost/no_guidance.yaml` for the matched control with `guidance.lambda_0 = 0`.

Start a new corpus with `--max_duration_seconds 60` to confirm decoding and media handling
work before committing to a full-length run.

## 10. Configuration

Configuration is layered: dataclass defaults, then a YAML file, then `--set` overrides.
A YAML file may name a parent with `_base_`.

```bash
python -m ost.cli infer --config configs/ost/default.yaml \
    --set guidance.lambda_0=0.5 \
    --set claims.max_per_chunk=3 \
    --media clips/example.mp4 --question "..." --output_dir ../output
```

`configs/ost/default.yaml` reproduces the paper's default configuration table. Key groups:

| Group | Controls |
|---|---|
| `streaming` | perception interval, decision interval, frame rate |
| `claims` | evidence-window durations, retry slack, per-chunk and active capacities, depth |
| `retention` | separate audio and visual budgets, dense floor, dense windows, retrieval slots |
| `memory.pyramid` | branching and the level capacities |
| `verifier` | head geometry, the calibration cap, per-modality verdict bands, checkpoints |
| `retraction` | the reliability floor and the retraction scale |
| `guidance` | the base guidance scale |
| `decoding` | attention backend, grammar constraints, sampling, token budgets, seed |
| `gate`, `forecaster`, `policy` | checkpoints and the gate threshold |

The configuration validates cross-section invariants and refuses inconsistent settings
rather than failing later. The level-1 pyramid capacity must be large enough to hold a
claim's Omni-State until the claim settles; both retention buffers must retain the longest
evidence window plus the retry slack plus one decision interval at full density; and the
attention backend must be one that accepts a floating-point additive mask.

## 11. Troubleshooting

**`the verdict-reliability attention bias of Eq. (12) has no effect on this backbone`**
The runtime probed the bias and found the logits unchanged, so retraction would be a no-op.
Check that `decoding.attention_backend` is `sdpa` or `eager`. Fused kernels such as
`flash_attention_2` reject the four-dimensional floating-point mask the bias needs. If a
new `transformers` release changed how the decoder builds its causal mask, the probe is
telling you the hook no longer lands; fix the hook rather than disabling the probe.

**`No backbone path configured`**
Pass `--model_path`, set `model.path` in the YAML, or export `OST_MODEL_PATH`.

**`no verifier scoring head at ...`**
The path was named in the config or on the command line but nothing is there. Correct it,
or remove the setting to run with the zero-shot verifier deliberately.

**`the verifier head expects a N-wide pooled representation but this backbone is M`**
The head belongs to a different backbone. The same check applies to the forecast soft
prompt.

**`ffmpeg was not found`**
Install it, or set `OST_FFMPEG` to its absolute path.

**`pyramid level-1 capacity is too small`**
The configuration would let a claim's Omni-State be folded away before the claim settles.
Either raise `memory.pyramid.capacities[0]` or shorten the longest evidence window.

**`retention.floor_seconds is below the perceptual retention floor`**
Both perceptual buffers must retain the longest evidence window plus the retry slack plus
one decision interval at full density, otherwise a pending claim's evidence could be
evicted before it is verified.

**Out of memory when loading the backbone**
Load in `bfloat16` on a single device; see §5 on device sharding.

**A run with refutations makes almost no progress**
See §9: guided decoding is quadratic in the generated length. Reduce
`decoding.max_new_tokens_state`, or use `configs/ost/no_guidance.yaml`.

## 12. Citation

```bibtex
@inproceedings{ost_anonymous,
  title     = {Omni-Streaming Thinking},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2026},
  note      = {Code: this repository}
}
```

## License

See [`LICENSE`](LICENSE).
