#!/usr/bin/env bash
# Training-budget sweep for the SAIL alignment-probing baseline.
# Train on the first N CC3M pairs (nested subsets of cc3m10k), evaluate on the
# unchanged COCO-2K gallery. Everything else is the SAIL-verbatim config.
set -uo pipefail
cd "$(dirname "$0")"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BUDGETS="${BUDGETS:-500 1000 2000 3000 5000 10000}"
ALIGN_ARGS="${ALIGN_ARGS:---train-dataset cc3m10k --test-dataset coco2k --extra-positive \
--optimizer lion --linear-type linear --target-dimension 2048 --loss siglip \
--loss-reduction mean --logit-scale 20 --logit-bias -10 --lr-grid 1e-5 --wd-grid 1e-7 \
--beta1 0.9 --beta2 0.99 --steps 2000 --batch-size 32768 --val-frac 0 --select final --tf32}"
IFS=',' read -r -a G <<< "$GPUS"; NG=${#G[@]}
mkdir -p logs
for N in $BUDGETS; do
  echo "=== budget N=$N ==="
  pids=(); i=0
  for llm in qwen25 qwen3 smollm2; do
    for ((s=0; s<NG/3; s++)); do
      gpu=${G[$((i % NG))]}
      CUDA_VISIBLE_DEVICES="$gpu" python train_align.py --llm "$llm" --tag "budget_n${N}" \
        --train-budget "$N" --shard "$s" --num-shards $((NG/3)) --device cuda:0 $ALIGN_ARGS \
        > "logs/budget_${N}_${llm}_${s}.log" 2>&1 &
      pids+=($!); i=$((i+1))
    done
  done
  for p in "${pids[@]}"; do wait "$p"; done
  grep -hE "\[FAIL\]|\[miss\]" logs/budget_${N}_*.log || echo "  N=$N: 210 combos done"
done
