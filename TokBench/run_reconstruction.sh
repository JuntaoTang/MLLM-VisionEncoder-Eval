#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_SCRIPTS="$SCRIPT_DIR/tokenzier_vae_scripts/image_scripts"

if [[ $# -ne 1 ]]; then
    echo "usage: $0 {toklip_s|toklip_l|unitok|vilau_256|uniar_bsq}" >&2
    exit 2
fi

export DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/tokbench_data}"
export RECON_ROOT="${RECON_ROOT:-$SCRIPT_DIR/image_reconstruction_results}"
export MODEL_ZOO="${MODEL_ZOO:-$SCRIPT_DIR/tokenizer_modelzoo}"
export PADDING_SIZES=256

case "$1" in
    toklip_s) bash "$IMAGE_SCRIPTS/toklip_s.sh" ;;
    toklip_l) bash "$IMAGE_SCRIPTS/toklip_l.sh" ;;
    unitok) bash "$IMAGE_SCRIPTS/unitok.sh" ;;
    vilau_256) bash "$IMAGE_SCRIPTS/vilau_7b_256.sh" ;;
    uniar_bsq)
        echo "UniAR-BSQ has no released reconstruction decoder; TokBench is not applicable." >&2
        exit 3
        ;;
    *) echo "unknown tokenizer: $1" >&2; exit 2 ;;
esac
