#!/usr/bin/env bash
# One GPU handles shard $1 of all three LLMs for budget $2 on dataset $3.
set -uo pipefail
cd "$(dirname "$0")"
G=$1; N=$2; DS=$3
A="--train-dataset $DS --test-dataset coco2k --extra-positive --optimizer lion --linear-type linear \
--target-dimension 2048 --loss siglip --loss-reduction mean --logit-scale 20 --logit-bias -10 \
--lr-grid 1e-5 --wd-grid 1e-7 --beta1 0.9 --beta2 0.99 --steps 2000 --batch-size 32768 \
--val-frac 0 --select final --tf32"
for llm in qwen3 qwen25 smollm2; do
  CUDA_VISIBLE_DEVICES=$G python train_align.py --llm "$llm" --tag "budget_n${N}" \
    --train-budget "$N" --shard "$G" --num-shards 8 --device cuda:0 $A \
    >> "logs/budget_${N}_gpu${G}.log" 2>&1
done
