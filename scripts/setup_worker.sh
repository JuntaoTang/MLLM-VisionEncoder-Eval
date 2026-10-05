#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <mllm|probing|knn|ckax|tokbench|zero_shot> <absolute-python-path>" >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKER_GROUP="$1"
WORKER_PYTHON="$2"
case "${WORKER_GROUP}" in
  mllm) REQUIREMENTS="${REPO_ROOT}/requirements/mllm-original.txt" ;;
  probing) REQUIREMENTS="${REPO_ROOT}/src/workers/linear/requirements.txt" ;;
  knn) REQUIREMENTS="${REPO_ROOT}/src/workers/knn/requirements.txt" ;;
  ckax) REQUIREMENTS="${REPO_ROOT}/src/workers/ckax/requirements.txt" ;;
  tokbench) REQUIREMENTS="${REPO_ROOT}/src/workers/tokbench/requirements.txt" ;;
  zero_shot) REQUIREMENTS="${REPO_ROOT}/src/workers/zero_shot/requirements.txt" ;;
  *) echo "Unknown worker group: ${WORKER_GROUP}" >&2; exit 2 ;;
esac
if [[ "${WORKER_PYTHON}" != /* || ! -x "${WORKER_PYTHON}" ]]; then
  echo "Supply an executable absolute Python path for a separate environment." >&2
  exit 2
fi
"${WORKER_PYTHON}" -m pip install -r "${REQUIREMENTS}"
"${WORKER_PYTHON}" -m pip install --no-deps -e "${REPO_ROOT}"
echo "Installed historical worker requirements; GPU/model smoke tests are still required."
