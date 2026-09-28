#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROBE_GPUS="${PROBE_GPUS:-0}"

labels=()
models=()
while IFS=$'\t' read -r rank label model _head; do
    [[ -n "$rank" && "$rank" != \#* ]] || continue
    labels+=("$label")
    models+=("$model")
done < "$SCRIPT_DIR/tokenizers.tsv"
[[ ${#models[@]} -eq 70 ]] || { echo "manifest must contain 70 models" >&2; exit 2; }

if [[ $# -gt 0 ]]; then
    selected=()
    for requested in "$@"; do
        found=""
        for ((i = 0; i < ${#models[@]}; i += 1)); do
            if [[ "$requested" == "$((i + 1))" || "$requested" == "${labels[$i]}" || "$requested" == "${models[$i]}" ]]; then
                selected+=("${models[$i]}")
                found=1
                break
            fi
        done
        [[ -n "$found" ]] || { echo "unknown model or rank: $requested" >&2; exit 2; }
    done
    models=("${selected[@]}")
fi

IFS=',' read -r -a gpu_ids <<< "$PROBE_GPUS"
for gpu in "${gpu_ids[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "PROBE_GPUS must be comma-separated GPU ids" >&2; exit 2; }
done

pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap 'cleanup; exit 130' INT TERM

workers=${#gpu_ids[@]}
for ((worker = 0; worker < workers; worker += 1)); do
    gpu="${gpu_ids[$worker]}"
    (
        for ((i = worker; i < ${#models[@]}; i += workers)); do
            echo ">> model $((i + 1))/${#models[@]} ${models[$i]} on GPU $gpu"
            CUDA_VISIBLE_DEVICES="$gpu" \
            SWEEP_TOKENIZER_INDEX="$((i + 1))" \
            SWEEP_TOKENIZER_TOTAL="${#models[@]}" \
                bash "$SCRIPT_DIR/run_model.sh" "${models[$i]}"
        done
    ) &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
pids=()
((failed == 0)) || exit 1
echo ">> all requested 5-shot probes completed"
