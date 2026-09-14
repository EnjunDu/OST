#!/usr/bin/env bash
# Streaming inference on one media file.
#
# The backbone and the checkpoints come from the config named in OST_CONFIG. Set
# OST_MODEL_PATH to override the backbone path without editing the config.
#
#   export OST_OUTPUT_DIR=../output
#   bash scripts/run_ost.sh --media clips/example.mp4 --question "Which gate?"
#
# Any extra flag is passed through, so a checkpoint can also be named on the command line:
#
#   bash scripts/run_ost.sh --media clips/example.mp4 --question "..." \
#       --verifier_head ./checkpoints/verifier
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_output_dir
cd "${REPO_ROOT}"

exec "${PYTHON}" -m ost.cli infer \
  --config "${OST_CONFIG:-configs/ost/default.yaml}" \
  ${OST_MODEL_PATH:+--model_path "${OST_MODEL_PATH}"} \
  --output_dir "${OST_OUTPUT_DIR}/inference" \
  "$@"
