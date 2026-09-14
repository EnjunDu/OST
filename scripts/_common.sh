#!/usr/bin/env bash
# Shared setup for the OST scripts.
#
# Every path comes from the environment, a flag or the YAML config. Nothing defaults into
# the repository, so a run cannot silently write results into the source tree.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

: "${PYTHON:=python}"

require_env() {
  local name="$1" hint="$2"
  if [[ -z "${!name:-}" ]]; then
    echo "error: ${name} is not set. ${hint}" >&2
    exit 2
  fi
}

require_output_dir() {
  require_env OST_OUTPUT_DIR "Point it at a directory OUTSIDE this repository, e.g. export OST_OUTPUT_DIR=../output"
  case "$(cd "${OST_OUTPUT_DIR}" 2>/dev/null && pwd || echo "${OST_OUTPUT_DIR}")" in
    "${REPO_ROOT}"|"${REPO_ROOT}"/*)
      echo "error: OST_OUTPUT_DIR must not be inside the repository (${REPO_ROOT})." >&2
      exit 2
      ;;
  esac
  mkdir -p "${OST_OUTPUT_DIR}"
}
