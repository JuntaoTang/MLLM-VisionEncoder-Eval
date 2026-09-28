#!/usr/bin/env bash
# One-click VTB environment setup.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME=VTB
PYTHON=3.10
TORCH_INDEX=https://download.pytorch.org/whl/cu124

if [ -n "${CONDA_EXE:-}" ]; then
  CONDA_BASE="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="${HOME}/miniconda3"
else
  echo "ERROR: conda not found." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" "python=${PYTHON}" -y
fi
conda activate "${ENV_NAME}"

pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url "${TORCH_INDEX}"
pip install -r "${ROOT}/requirements.txt"

LLAVA_DIR="${ROOT}/third_party/LLaVA-NeXT"
if [ ! -f "${LLAVA_DIR}/llava/train/train_mem.py" ]; then
  mkdir -p "${ROOT}/third_party"
  git clone --depth 1 https://github.com/LLaVA-VL/LLaVA-NeXT.git "${LLAVA_DIR}"
fi

VLMEVAL_DIR="${ROOT}/third_party/VLMEvalKit"
if [ ! -f "${VLMEVAL_DIR}/run.py" ]; then
  git clone --depth 1 https://github.com/open-compass/VLMEvalKit.git "${VLMEVAL_DIR}"
fi

pip install -e "${VLMEVAL_DIR}"
pip install opencv-python-headless
pip uninstall -y opencv-python 2>/dev/null || true
pip install --force-reinstall opencv-python-headless

echo "VTB ready. Run: conda activate ${ENV_NAME} && cd ${ROOT} && python run.py"
