#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}[methods,knn]"

echo "Installed vision-encoder-eval with method dependencies."
echo "Next: bash ${REPO_ROOT}/scripts/preflight.sh <experiment.yaml> --local <local.yaml>"
