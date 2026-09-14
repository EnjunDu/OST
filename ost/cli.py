"""Command-line entry point.

Streaming inference on one media file:

    python -m ost.cli infer --config configs/ost/default.yaml \
        --media path/to/video.mp4 --question "..." --output_dir path/to/output

Every path is an argument or a configuration field. Nothing defaults into the source tree,
and relative paths resolve against the working directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------------------


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model_path",
        default=None,
        help="path to the backbone weights (or set model.path, or OST_MODEL_PATH)",
    )
    parser.add_argument("--config", default=None, help="YAML configuration file")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config key, e.g. --set guidance.lambda_0=0",
    )
    parser.add_argument("--device", default=None, help="device for the backbone")

    checkpoints = parser.add_argument_group(
        "checkpoints",
        "Each may also be set in the configuration; the command line wins. Any that is "
        "omitted leaves that component unmodified, which the run record reports.",
    )
    checkpoints.add_argument(
        "--verifier_head", default=None, help="verifier scoring head directory"
    )
    checkpoints.add_argument("--gate_head", default=None, help="gate head directory")
    checkpoints.add_argument(
        "--forecast_prompt", default=None, help="forecast soft-prompt directory"
    )
    checkpoints.add_argument("--adapter", default=None, help="policy adapter directory")
    checkpoints.add_argument(
        "--thresholds",
        default=None,
        help="calibration JSON with per-modality verdict bands and the retraction scale",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="directory for the run record; must be outside this repository",
    )
    parser.add_argument(
        "--cache_dir", default=None, help="scratch directory for clip extraction"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_level", default="INFO")

    parser.add_argument("--media", required=True, help="path to a video file")
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--max_duration_seconds",
        type=float,
        default=None,
        help="truncate the stream, for a bounded run on a long clip",
    )


def _coerce(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _overrides(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for entry in getattr(args, "set", []) or []:
        if "=" not in entry:
            raise SystemExit(f"--set expects KEY=VALUE, got {entry!r}")
        key, raw = entry.split("=", 1)
        out[key.strip()] = _coerce(raw.strip())
    if args.model_path:
        out["model.path"] = args.model_path
    if args.device:
        out["model.device"] = args.device
    if args.seed is not None:
        out["decoding.seed"] = int(args.seed)
    return out


# --------------------------------------------------------------------------------------
# infer
# --------------------------------------------------------------------------------------


def command_infer(args: argparse.Namespace) -> int:
    from ost.config import OSTConfig
    from ost.runtime import build_runtime
    from ost.streaming.source import MediaStreamSource
    from ost.utils.paths import resolve_cache_dir, resolve_output_dir
    from ost.utils.seed import set_seed

    set_seed(int(args.seed))
    config = OSTConfig.load(args.config, overrides=_overrides(args))
    runtime = build_runtime(
        config,
        verifier_head_path=args.verifier_head,
        gate_head_path=args.gate_head,
        forecast_prompt_path=args.forecast_prompt,
        adapter_path=args.adapter,
        thresholds_path=args.thresholds,
    )

    output_dir = resolve_output_dir(args.output_dir)
    cache_dir = resolve_cache_dir(args.cache_dir)
    media = Path(args.media).expanduser()

    source = MediaStreamSource(
        media,
        cache_dir=cache_dir / media.stem,
        perception_interval_seconds=config.streaming.perception_interval_seconds,
        frames_per_second=config.streaming.frames_per_second,
        audio_sample_rate=config.streaming.audio_sample_rate,
        max_duration_seconds=args.max_duration_seconds,
    )
    try:
        result = runtime.orchestrator.run(question=args.question, source=source)
    finally:
        source.close()

    payload = {
        "question": args.question,
        "media": media.name,
        "answer": result.answer.text,
        "low_confidence": result.answer.low_confidence,
        "expired_support": result.answer.expired_support,
        "unresolved_support": result.answer.unresolved_support,
        "stop_chunk": result.stop_chunk,
        "stop_time": result.stop_time,
        "stopped_early": result.stopped_early,
        "chunks": result.num_chunks,
        "claims": result.num_claims,
        "refutations": result.num_refutations,
        "guided_chunks": result.num_guided_chunks,
        "runtime": runtime.describe(),
        "trace": [
            {
                "chunk": t.chunk,
                "t_end": t.t_end,
                "admitted_claims": t.admitted_claim_ids,
                "verdicts": [
                    {
                        "claim": v.claim_id,
                        "verdict": v.verdict.value,
                        "gamma": v.contradiction_margin,
                        "score": v.score,
                    }
                    for v in t.verdicts
                ],
                "refuted": t.refuted_claim_ids,
                "expired": t.expired_claim_ids,
                "guidance_scale": t.guidance_scale,
                "guidance_applied": t.guidance_applied,
                "gate": t.gate_action.value,
                "gate_score": t.gate_score,
                "hard_wait": t.hard_wait,
                "folds": t.folds,
            }
            for t in result.traces
        ],
        "states": [state.as_dict() for state in result.states],
    }

    # Named after the clip so a sweep over several clips does not overwrite itself.
    target = output_dir / f"{media.stem}.inference.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("wrote %s", target)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "answer",
                    "stop_chunk",
                    "stop_time",
                    "chunks",
                    "claims",
                    "refutations",
                    "low_confidence",
                )
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ost.cli",
        description="Omni-Streaming Thinking: streaming audio-visual inference.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser(
        "infer", help="run streaming inference on one media file"
    )
    _add_arguments(infer)
    infer.set_defaults(handler=command_infer)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    from ost.utils.logging import configure_logging

    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
