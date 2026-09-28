#!/usr/bin/env bash
# End-to-end COCO-2K alignment probing: 70 tokenizers x 3 LLMs = 210 combos.
#
#   bash run_all.sh              # everything, 8 GPUs
#   GPUS="0,1,2,3" bash run_all.sh
#   STAGES="vision" bash run_all.sh
#
set -uo pipefail
cd "$(dirname "$0")"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
STAGES="${STAGES:-data text vision align summary}"
VISION_BATCH="${VISION_BATCH:-32}"
WORKERS="${WORKERS:-6}"
EXTRA_ALIGN="${EXTRA_ALIGN:-}"
EXTRA_VISION="${EXTRA_VISION:-}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
mkdir -p logs

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

# ---------------------------------------------------------------- 0. data ----
if has_stage data; then
  echo "=== [1/5] building COCO-2K split ==="
  python build_data.py || exit 1
fi

# ------------------------------------------------------- 1. text embeddings ---
if has_stage text; then
  echo "=== [2/5] pre-encoding captions with the 3 LLMs ==="
  i=0
  pids=()
  for llm in qwen25 qwen3 smollm2; do
    gpu=${GPU_ARR[$((i % NGPU))]}
    echo "  -> $llm on cuda:$gpu (log: logs/text_${llm}.log)"
    CUDA_VISIBLE_DEVICES="$gpu" python encode_text.py --llm "$llm" --device cuda:0 \
      > "logs/text_${llm}.log" 2>&1 &
    pids+=($!)
    i=$((i + 1))
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ $fail -ne 0 ] && { echo "text encoding failed, see logs/text_*.log"; exit 1; }
  tail -n 2 logs/text_*.log
fi

# ----------------------------------------------------- 2. vision embeddings ---
if has_stage vision; then
  echo "=== [3/5] pre-encoding COCO-2K images with 70 tokenizers on $NGPU GPU(s) ==="
  pids=()
  for ((s = 0; s < NGPU; s++)); do
    gpu=${GPU_ARR[$s]}
    echo "  -> shard $s/$NGPU on cuda:$gpu (log: logs/vision_shard${s}.log)"
    CUDA_VISIBLE_DEVICES="$gpu" python encode_vision.py \
      --shard "$s" --num-shards "$NGPU" --device cuda:0 \
      --batch "$VISION_BATCH" --workers "$WORKERS" $EXTRA_VISION \
      > "logs/vision_shard${s}.log" 2>&1 &
    pids+=($!)
  done
  python watch_progress.py --stage vision --pids "${pids[*]}"
  for p in "${pids[@]}"; do wait "$p"; done
  grep -h "\[FAIL\]" logs/vision_shard*.log || echo "  all tokenizers encoded"
fi

# ------------------------------------------------- 3. alignment layer + eval ---
if has_stage align; then
  echo "=== [4/5] training alignment layers + retrieval (210 combos) ==="
  pids=()
  i=0
  # one job per (llm, shard); alignment training is tiny, 2 shards/LLM is plenty
  ASHARDS="${ASHARDS:-2}"
  for llm in qwen25 qwen3 smollm2; do
    for ((s = 0; s < ASHARDS; s++)); do
      gpu=${GPU_ARR[$((i % NGPU))]}
      echo "  -> $llm shard $s on cuda:$gpu (log: logs/align_${llm}_${s}.log)"
      CUDA_VISIBLE_DEVICES="$gpu" python train_align.py --llm "$llm" \
        --shard "$s" --num-shards "$ASHARDS" --device cuda:0 $EXTRA_ALIGN \
        > "logs/align_${llm}_${s}.log" 2>&1 &
      pids+=($!)
      i=$((i + 1))
    done
  done
  python watch_progress.py --stage align --pids "${pids[*]}"
  for p in "${pids[@]}"; do wait "$p"; done
  grep -hE "\[FAIL\]|\[miss\]" logs/align_*.log || echo "  all combos scored"
fi

# ------------------------------------------------------------- 4. summary -----
if has_stage summary; then
  echo "=== [5/5] summary ==="
  python summarize.py --protocol 1cap
  python summarize.py --protocol 5cap
fi
