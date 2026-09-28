#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/tokbench_data}"
export RECON_ROOT="${RECON_ROOT:-$SCRIPT_DIR/image_reconstruction_results}"
export OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/image_outputs}"
export RES=256

for tokenizer in toklip_s toklip_l unitok vilau_7b_256; do
    echo ">> evaluating $tokenizer at 256"
    TOKENIZER_NAME="$tokenizer" bash "$SCRIPT_DIR/image_eval.sh"
done

python "$SCRIPT_DIR/summarize_paper_results.py" \
    --input-dir "$OUT_DIR" \
    --output-dir "${SUMMARY_DIR:-$SCRIPT_DIR/../output}"
