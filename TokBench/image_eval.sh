#!/bin/bash
# Image evaluation adapted for a single A100.
#  - one GPU (device 0)
#  - text datasets run SEQUENTIALLY (upstream launched them with `&` but left the
#    `wait` commented out, so compute_all_metrics.py raced half-written JSON)
set -e

# ---- config (override via env) ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../VisualTokenizer/TokBench
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/tokbench_data}"          # images/ + annotations/
RECON_ROOT="${RECON_ROOT:-$REPO_ROOT/image_reconstruction_results}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/image_outputs}"
TOKENIZER_NAME="${TOKENIZER_NAME:-${MODEL_NAME:-resize}}"
RES="${RES:-256}"

cd "$REPO_ROOT"

python check_eval_requirements.py

# ---- text recognition (T-ACC / T-NED) ----
dataset_names=("ic13" "ic15" "tt" "textocr" "cord" "sroie" "infograph" "docvqa")
for dataset_name in "${dataset_names[@]}"; do
  python eval_text.py \
    --img_folder "${RECON_ROOT}/${TOKENIZER_NAME}/text_data/${dataset_name}_${RES}/" \
    --gt_path    "${DATA_ROOT}/annotations/text_${dataset_name}.json" \
    --dataset "${dataset_name}" \
    --data_type "image" \
    --batch_size 64 \
    --method_name "$TOKENIZER_NAME" \
    --setting "$RES" \
    --save_dir "$OUT_DIR"
done

# ---- face similarity (F-Sim) ----
python eval_face.py \
    --original_image_path "${DATA_ROOT}/images/face_data/wflw" \
    --reconstruction_image_path "${RECON_ROOT}/${TOKENIZER_NAME}/face_data/wflw_${RES}/" \
    --tokenizer "$TOKENIZER_NAME" \
    --data_type "image" \
    --meta_path "${DATA_ROOT}/annotations/face_meta.json" \
    --setting "$RES" \
    --save_dir "$OUT_DIR"

# ---- aggregate by scale bucket ----
python compute_all_metrics.py --setting "$RES" --data_type image --output_path "$OUT_DIR"
