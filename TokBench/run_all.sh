#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for tokenizer in toklip_s toklip_l unitok vilau_256; do
    bash "$SCRIPT_DIR/run_reconstruction.sh" "$tokenizer"
done
bash "$SCRIPT_DIR/run_eval_all.sh"
