#!/usr/bin/env bash
set -euo pipefail

# Reproducible upstream checkout. Override this with a reviewed commit when needed.
CLIP_BENCHMARK_REF="${CLIP_BENCHMARK_REF:-main}"
WORKDIR="${WORKDIR:-third_party}"
DATASET_ROOT="${IMAGENET_ROOT:?Set IMAGENET_ROOT to the official ImageNet validation directory}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_FILE="${MODEL_FILE:-$SCRIPT_DIR/models.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-results/zero_shot}"

mkdir -p "$WORKDIR"
if [[ ! -d "$WORKDIR/CLIP_benchmark/.git" ]]; then
  git clone https://github.com/LAION-AI/CLIP_benchmark.git "$WORKDIR/CLIP_benchmark"
fi
git -C "$WORKDIR/CLIP_benchmark" fetch --tags origin
git -C "$WORKDIR/CLIP_benchmark" checkout "$CLIP_BENCHMARK_REF"
python -m pip install -e "$WORKDIR/CLIP_benchmark"

# One model/pretrained pair per line prevents an accidental Cartesian product.
# The checked-in list was validated against open_clip_torch 3.3.0.
mkdir -p "$OUTPUT_DIR"
clip_benchmark eval \
  --dataset imagenet1k \
  --dataset_root "$DATASET_ROOT" \
  --task zeroshot_classification \
  --model_type open_clip \
  --pretrained_model "$MODEL_FILE" \
  --output "$OUTPUT_DIR/{dataset}_{model}_{pretrained}_{language}_{task}.json"

