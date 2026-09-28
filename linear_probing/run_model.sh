#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MINI_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 MODEL_OR_RANK [linear_probe.py arguments...]" >&2
    exit 2
fi

requested="$1"
shift
resolved=""
while IFS=$'\t' read -r rank label model _head; do
    [[ -n "$rank" && "$rank" != \#* ]] || continue
    if [[ "$requested" == "$rank" || "$requested" == "$label" || "$requested" == "$model" ]]; then
        resolved="$model"
        break
    fi
done < "$SCRIPT_DIR/tokenizers.tsv"
[[ -n "$resolved" ]] || { echo "unknown model or rank: $requested" >&2; exit 2; }

PYTHON_BIN="${PROBE_PYTHON:-python}"
DATA_ROOT="${IMAGENET_ROOT:-$MINI_ROOT/data/imagenet1k}"
EXTRA_ROOT="${IMAGENET_EXTRA_ROOT:-$DATA_ROOT/extra}"
OUTPUT_ROOT="${PROBE_OUTPUT_ROOT:-$MINI_ROOT/output/linear_probing_runs}"
CACHE_ROOT="${PROBE_CACHE_ROOT:-$OUTPUT_ROOT/_feature_cache}"
NUM_WORKERS="${PROBE_NUM_WORKERS:-8}"

PYTHONPATH="$SCRIPT_DIR/vendor/dinov2:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$SCRIPT_DIR/linear_probe.py" \
    --model "$resolved" \
    --data-root "$DATA_ROOT" \
    --extra-root "$EXTRA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --cache-root "$CACHE_ROOT" \
    --five-shot-cache-root "$CACHE_ROOT" \
    --num-workers "$NUM_WORKERS" \
    --cap-shots 5 \
    --validation-samples 5000 \
    --validation-seed 42 \
    "$@"
