#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <suite.yaml> --local <local.yaml> [--dry-run] [--set key=value ...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SUITE_CONFIG="$1"
shift

PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  exec "${PYTHON_BIN}" -m vision_encoder_eval experiment run \
  --experiment "${SUITE_CONFIG}" \
  "$@"
