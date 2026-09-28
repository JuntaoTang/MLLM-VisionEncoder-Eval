#!/usr/bin/env bash
# SAIL-protocol alignment probing: train on CC3M-2K, evaluate retrieval on COCO-2K.
#
#   bash run_cc3m.sh                  # everything, 8 GPUs
#   STAGES="align summary" bash run_cc3m.sh
#
# Budget: ~1.77 PFLOPs per tokenizer (python budget.py --extra-positive \
#   --grid 1 --steps 300 --target-dimension 2048).
set -uo pipefail
cd "$(dirname "$0")"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
STAGES="${STAGES:-data text vision align summary}"
TAG="${TAG:-cc3m2k}"
ASHARDS="${ASHARDS:-2}"
VISION_BATCH="${VISION_BATCH:-32}"
WORKERS="${WORKERS:-6}"

# --- SAIL scripts/alignment_probing.sh, verbatim where it can be copied ------
#  linear head, d=2048, SigLIP loss + longSV extra positive, logit_scale 20 /
#  bias -10, Lion (params.py default), lr 1e-5, wd 1e-7, betas (0.9, 0.99),
#  cosine schedule with ceil(0.1*total) warmup, no gradient clipping, batch
#  32768 (> our 2000 pairs, so full batch), final checkpoint, no val split and
#  no hyper-parameter search - SAIL does none of those.
#
#  The ONE thing that cannot be copied is training duration: SAIL's 100 epochs
#  over 23M pairs is ~70,000 optimizer steps, whereas 100 epochs over 2000
#  pairs at full batch is 100 steps and lands at chance level. We instead train
#  to convergence under SAIL's own optimizer and lr: the train loss plateaus
#  around 2000-4000 full-batch steps (logs/lion_convergence_probe.txt), so the
#  step count is set by convergence, never by the COCO test score.
#  Cost: ~2.90 PFLOPs/tokenizer (budget.py --extra-positive --grid 1
#  --steps 2000 --target-dimension 2048 --val-frac 0).
ALIGN_ARGS="${ALIGN_ARGS:---train-dataset cc3m2k --test-dataset coco2k \
--extra-positive --optimizer lion --linear-type linear --target-dimension 2048 \
--loss siglip --loss-reduction mean --logit-scale 20 --logit-bias -10 \
--lr-grid 1e-5 --wd-grid 1e-7 --beta1 0.9 --beta2 0.99 \
--steps 2000 --batch-size 32768 --val-frac 0 --select final}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
mkdir -p logs
has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

if has_stage data; then
  echo "=== [1/5] sampling CC3M-2K ==="
  [ -f data/cc3m2k.json ] && echo "  data/cc3m2k.json exists, skipping" \
    || python build_cc3m2k.py || exit 1
fi

if has_stage text; then
  echo "=== [2/5] encoding CC3M captions (raw + longSV) with the 3 LLMs ==="
  pids=(); i=0
  for llm in qwen25 qwen3 smollm2; do
    gpu=${GPU_ARR[$((i % NGPU))]}
    CUDA_VISIBLE_DEVICES="$gpu" python encode_text.py --dataset cc3m2k --llm "$llm" \
      --device cuda:0 --max-length 128 > "logs/text_cc3m2k_${llm}.log" 2>&1 &
    pids+=($!); i=$((i + 1))
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ $fail -ne 0 ] && { echo "text encoding failed"; exit 1; }
fi

if has_stage vision; then
  echo "=== [3/5] encoding the 2000 CC3M images with 70 tokenizers ==="
  pids=()
  for ((s = 0; s < NGPU; s++)); do
    gpu=${GPU_ARR[$s]}
    CUDA_VISIBLE_DEVICES="$gpu" python encode_vision.py --dataset cc3m2k \
      --shard "$s" --num-shards "$NGPU" --device cuda:0 \
      --batch "$VISION_BATCH" --workers "$WORKERS" \
      > "logs/vision_cc3m2k_shard${s}.log" 2>&1 &
    pids+=($!)
  done
  python watch_progress.py --stage vision --dataset cc3m2k --pids "${pids[*]}"
  for p in "${pids[@]}"; do wait "$p"; done
  grep -h "\[FAIL\]" logs/vision_cc3m2k_shard*.log || echo "  all 70 encoded"
fi

if has_stage align; then
  echo "=== [4/5] training alignment layers on CC3M-2K, scoring on COCO-2K ==="
  pids=(); i=0
  for llm in qwen25 qwen3 smollm2; do
    for ((s = 0; s < ASHARDS; s++)); do
      gpu=${GPU_ARR[$((i % NGPU))]}
      CUDA_VISIBLE_DEVICES="$gpu" python train_align.py --llm "$llm" --tag "$TAG" \
        --shard "$s" --num-shards "$ASHARDS" --device cuda:0 $ALIGN_ARGS \
        > "logs/align_${TAG}_${llm}_${s}.log" 2>&1 &
      pids+=($!); i=$((i + 1))
    done
  done
  python watch_progress.py --stage sweep --tag "$TAG" --pids "${pids[*]}"
  for p in "${pids[@]}"; do wait "$p"; done
  grep -hE "\[FAIL\]|\[miss\]" logs/align_${TAG}_*.log || echo "  all 210 combos scored"
fi

if has_stage summary; then
  echo "=== [5/5] summary + correlation with the main table ==="
  python summarize.py --protocol 1cap --tag "$TAG"
  python summarize.py --protocol 5cap --tag "$TAG"
  python correlate.py --protocol 5cap --tag "$TAG"
fi
