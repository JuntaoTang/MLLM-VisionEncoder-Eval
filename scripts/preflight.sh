#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <experiment.yaml> --local <local.yaml> [--set key=value ...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_CONFIG="$1"
shift

PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  exec "${PYTHON_BIN}" -m vision_encoder_eval preflight \
  --experiment "${EXPERIMENT_CONFIG}" \
  "$@"
