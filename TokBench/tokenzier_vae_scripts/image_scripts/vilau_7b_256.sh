#!/bin/bash
# Reconstruction with the VILA-U 7B-256 vision tokenizer.
# Usage:
#   CUDA_VISIBLE_DEVICES=0 PADDING_SIZES="256" bash vilau_7b_256.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/tokbench_data}"
RECON_ROOT="${RECON_ROOT:-$REPO_ROOT/image_reconstruction_results}"
MODEL_ZOO="${MODEL_ZOO:-$REPO_ROOT/tokenizer_modelzoo}"
MODEL_NAME="${MODEL_NAME:-vilau_7b_256}"
VILAU_PATH="${VILAU_PATH:-$SCRIPT_DIR/vila-u}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ZOO/VILA-U/vila-u-7b-256}"
SIGLIP_CONFIG_PATH="${SIGLIP_CONFIG_PATH:-$MODEL_ZOO/VILA-U/siglip-large-patch16-256}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
if [ "${PADDING_SIZES+x}" ]; then PADDING_SIZES=($PADDING_SIZES); else PADDING_SIZES=(256); fi
if [ "${TEXT_DATAS+x}" ]; then TEXT_DATAS=($TEXT_DATAS); else TEXT_DATAS=(ic13 ic15 textocr tt cord docvqa infograph sroie); fi
if [ "${FACE_DATAS+x}" ]; then FACE_DATAS=($FACE_DATAS); else FACE_DATAS=(wflw); fi

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS="${CHUNKS:-${#GPULIST[@]}}"

VISION_TOWER_PATH="$MODEL_PATH"
if [ -d "$MODEL_PATH/vision_tower" ]; then
    VISION_TOWER_PATH="$MODEL_PATH/vision_tower"
fi

require_dir() {
    if [ ! -d "$1" ]; then
        echo "Missing $2: $1"
        echo "Download weights to $MODEL_ZOO/VILA-U or pass MODEL_PATH/SIGLIP_CONFIG_PATH."
        exit 1
    fi
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "Missing $2: $1"
        echo "Download weights to $MODEL_ZOO/VILA-U or pass MODEL_PATH/SIGLIP_CONFIG_PATH."
        exit 1
    fi
}

require_dir "$VILAU_PATH/vila_u" "VILA-U source directory"
require_dir "$VISION_TOWER_PATH" "VILA-U vision tower checkpoint"
require_file "$SIGLIP_CONFIG_PATH/config.json" "SigLIP base config"

cd "$SCRIPT_DIR"

for DATA in "${TEXT_DATAS[@]}"; do
    for PADDING_SIZE in "${PADDING_SIZES[@]}"; do
        echo "[$MODEL_NAME] padding=$PADDING_SIZE dataset=$DATA (text)"
        for IDX in $(seq 0 $((CHUNKS - 1))); do
            GPU_IDX=$((IDX % ${#GPULIST[@]}))
            CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX]} python vilau_rec.py \
                --image_path "$DATA_ROOT/images/text_data/$DATA" \
                --model_name "$MODEL_NAME" \
                --save_path "$RECON_ROOT/$MODEL_NAME/text_data/$DATA" \
                --vilau_path "$VILAU_PATH" \
                --model_path "$MODEL_PATH" \
                --siglip_config_path "$SIGLIP_CONFIG_PATH" \
                --padding_size "$PADDING_SIZE" \
                --batch_size "$BATCH_SIZE" \
                --dtype "$DTYPE" \
                --num_chunks "$CHUNKS" \
                --chunk_idx "$IDX" &
        done
        wait
    done
done

for DATA in "${FACE_DATAS[@]}"; do
    for PADDING_SIZE in "${PADDING_SIZES[@]}"; do
        echo "[$MODEL_NAME] padding=$PADDING_SIZE dataset=$DATA (face)"
        for IDX in $(seq 0 $((CHUNKS - 1))); do
            GPU_IDX=$((IDX % ${#GPULIST[@]}))
            CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX]} python vilau_rec.py \
                --image_path "$DATA_ROOT/images/face_data/$DATA" \
                --model_name "$MODEL_NAME" \
                --save_path "$RECON_ROOT/$MODEL_NAME/face_data/$DATA" \
                --vilau_path "$VILAU_PATH" \
                --model_path "$MODEL_PATH" \
                --siglip_config_path "$SIGLIP_CONFIG_PATH" \
                --padding_size "$PADDING_SIZE" \
                --batch_size "$BATCH_SIZE" \
                --dtype "$DTYPE" \
                --num_chunks "$CHUNKS" \
                --chunk_idx "$IDX" &
        done
        wait
    done
done

echo "Image reconstruction done -> $RECON_ROOT/$MODEL_NAME"
