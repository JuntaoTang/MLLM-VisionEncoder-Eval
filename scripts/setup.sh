#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BOOTSTRAP_PYTHON="${PYTHON_BIN:-python}"
if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  "${BOOTSTRAP_PYTHON}" -m venv --system-site-packages "${REPO_ROOT}/.venv"
fi
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

PACKAGE_INDEX="${PIP_INDEX_URL:-https://pypi.org/simple}"
"${PYTHON_BIN}" -m pip install --index-url "${PACKAGE_INDEX}" 'setuptools>=68' wheel
"${PYTHON_BIN}" -m pip install --index-url "${PACKAGE_INDEX}" --no-build-isolation -e "${REPO_ROOT}[methods,knn]"

echo "Installed vision-encoder-eval with method dependencies."
echo "Next: bash ${REPO_ROOT}/scripts/reproduce.sh --init-local --data-root /path/to/data"
echo "Then: bash ${REPO_ROOT}/scripts/reproduce.sh ravel"
