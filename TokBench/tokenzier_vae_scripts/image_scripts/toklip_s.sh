#!/bin/bash
# Reconstruction with the TokLIP-S VQ image tokenizer.
# Usage:
#   CUDA_VISIBLE_DEVICES=0 PADDING_SIZES="256" bash toklip_s.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/tokbench_data}"
RECON_ROOT="${RECON_ROOT:-$REPO_ROOT/image_reconstruction_results}"
MODEL_ZOO="${MODEL_ZOO:-$REPO_ROOT/tokenizer_modelzoo}"
MODEL_NAME="${MODEL_NAME:-toklip_s}"
TOKLIP_PATH="${TOKLIP_PATH:-$SCRIPT_DIR/TokLIP}"
TOKLIP_DIR="${TOKLIP_DIR:-$MODEL_ZOO/TokLIP}"
TOKLIP_CKPT_PATH="${TOKLIP_CKPT_PATH:-$TOKLIP_DIR/TokLIP_S_256.pt}"
VQ_CKPT_PATH="${VQ_CKPT_PATH:-$TOKLIP_DIR/vq_ds16_t2i.pt}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LATENT_SOURCE="${LATENT_SOURCE:-quantized}"
TOKLIP_MODEL_CONFIG="${TOKLIP_MODEL_CONFIG:-ViT-SO400M-16-SigLIP2-256-toklip}"
if [ "${PADDING_SIZES+x}" ]; then PADDING_SIZES=($PADDING_SIZES); else PADDING_SIZES=(256 512 1024); fi
if [ "${TEXT_DATAS+x}" ]; then TEXT_DATAS=($TEXT_DATAS); else TEXT_DATAS=(ic13 ic15 textocr tt cord docvqa infograph sroie); fi
if [ "${FACE_DATAS+x}" ]; then FACE_DATAS=($FACE_DATAS); else FACE_DATAS=(wflw); fi

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS="${CHUNKS:-${#GPULIST[@]}}"

require_dir() {
    if [ ! -d "$1" ]; then
        echo "Missing $2: $1"
        echo "Download TokLIP first, or pass TOKLIP_PATH=/path/to/TokLIP."
        exit 1
    fi
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "Missing $2: $1"
        echo "Download weights to $TOKLIP_DIR or pass TOKLIP_CKPT_PATH/VQ_CKPT_PATH."
        exit 1
    fi
}

require_dir "$TOKLIP_PATH/src/tokenizer" "TokLIP tokenizer source directory"
require_file "$VQ_CKPT_PATH" "TokLIP LlamaGen VQ checkpoint"

cd "$SCRIPT_DIR"

for DATA in "${TEXT_DATAS[@]}"; do
    for PADDING_SIZE in "${PADDING_SIZES[@]}"; do
        echo "[$MODEL_NAME] padding=$PADDING_SIZE dataset=$DATA (text)"
        for IDX in $(seq 0 $((CHUNKS - 1))); do
            GPU_IDX=$((IDX % ${#GPULIST[@]}))
            CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX]} python toklip_s_rec.py \
                --image_path "$DATA_ROOT/images/text_data/$DATA" \
                --model_name "$MODEL_NAME" \
                --save_path "$RECON_ROOT/$MODEL_NAME/text_data/$DATA" \
                --toklip_path "$TOKLIP_PATH" \
                --toklip_ckpt_path "$TOKLIP_CKPT_PATH" \
                --vq_ckpt_path "$VQ_CKPT_PATH" \
                --toklip_model_config "$TOKLIP_MODEL_CONFIG" \
                --latent_source "$LATENT_SOURCE" \
                --padding_size "$PADDING_SIZE" \
                --batch_size "$BATCH_SIZE" \
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
            CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX]} python toklip_s_rec.py \
                --image_path "$DATA_ROOT/images/face_data/$DATA" \
                --model_name "$MODEL_NAME" \
                --save_path "$RECON_ROOT/$MODEL_NAME/face_data/$DATA" \
                --toklip_path "$TOKLIP_PATH" \
                --toklip_ckpt_path "$TOKLIP_CKPT_PATH" \
                --vq_ckpt_path "$VQ_CKPT_PATH" \
                --toklip_model_config "$TOKLIP_MODEL_CONFIG" \
                --latent_source "$LATENT_SOURCE" \
                --padding_size "$PADDING_SIZE" \
                --batch_size "$BATCH_SIZE" \
                --num_chunks "$CHUNKS" \
                --chunk_idx "$IDX" &
        done
        wait
    done
done

echo "Image reconstruction done -> $RECON_ROOT/$MODEL_NAME"
