#!/usr/bin/env bash
# Robustness sweep: same 2000 COCO-2K pairs, different train/test partitions.
# Vision/text embeddings are already cached, so only the alignment stage reruns.
#
#   bash run_sweep.sh                     # default variant list
#   VARIANTS="t1200_s-1:1200:-1" bash run_sweep.sh
#
# A variant is  <tag>:<n_train>:<split_seed>   (split_seed -1 = coco2k.json order)
set -uo pipefail
cd "$(dirname "$0")"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
ASHARDS="${ASHARDS:-2}"
EXTRA_ALIGN="${EXTRA_ALIGN:-}"

# ratio sweep (fixed order) + repartition sweep (fixed 80/20 ratio, new seeds)
DEFAULT_VARIANTS="\
r400:400:-1 \
r800:800:-1 \
r1000:1000:-1 \
r1200:1200:-1 \
r1400:1400:-1 \
r1800:1800:-1 \
s1:1600:1 \
s2:1600:2 \
s3:1600:3"
VARIANTS="${VARIANTS:-$DEFAULT_VARIANTS}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
mkdir -p logs

for variant in $VARIANTS; do
  IFS=':' read -r tag ntrain sseed <<< "$variant"
  echo "=== variant $tag  (n_train=$ntrain, split_seed=$sseed) ==="
  pids=()
  i=0
  for llm in qwen25 qwen3 smollm2; do
    for ((s = 0; s < ASHARDS; s++)); do
      gpu=${GPU_ARR[$((i % NGPU))]}
      CUDA_VISIBLE_DEVICES="$gpu" python train_align.py --llm "$llm" \
        --tag "$tag" --n-train "$ntrain" --split-seed "$sseed" \
        --shard "$s" --num-shards "$ASHARDS" --device cuda:0 $EXTRA_ALIGN \
        > "logs/sweep_${tag}_${llm}_${s}.log" 2>&1 &
      pids+=($!)
      i=$((i + 1))
    done
  done
  python watch_progress.py --stage sweep --tag "$tag" --pids "${pids[*]}"
  for p in "${pids[@]}"; do wait "$p"; done
  grep -hE "\[FAIL\]|\[miss\]" logs/sweep_${tag}_*.log || echo "  $tag: all 210 combos scored"
  python summarize.py --protocol 1cap --tag "$tag" 2>/dev/null | grep -E "combos scored|mean="
done

echo
echo "=== all variants done ==="
ls -d results/sweeps/*/ 2>/dev/null
