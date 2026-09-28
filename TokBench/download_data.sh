#!/bin/bash
# Download the TokBench benchmark.
#   Default: IMAGE only (annotations/ + images.zip, skips the 1.35GB videos.zip).
#   WITH_VIDEO=1 bash download_data.sh   -> also fetch video_annotations/ + videos.zip
# Run inside the TokBench conda env (needs huggingface_hub >= 0.20 for `download`).
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/tokbench_data}"
WITH_VIDEO="${WITH_VIDEO:-0}"


mkdir -p "$DATA_ROOT"
echo "Using Hugging Face endpoint: ${HF_ENDPOINT:-https://huggingface.co}"

# image: annotations/ (text_*.json + face_meta.json) + image archive
huggingface-cli download Junfeng5/TokBench --repo-type dataset \
    --local-dir "$DATA_ROOT" \
    --include "annotations/*" "images.zip"

if [ ! -d "$DATA_ROOT/images" ]; then
    echo "Unzipping images.zip ..."
    unzip -q "$DATA_ROOT/images.zip" -d "$DATA_ROOT"
fi

# video (optional): video_annotations/ + video archive
if [ "$WITH_VIDEO" = "1" ]; then
    huggingface-cli download Junfeng5/TokBench --repo-type dataset \
        --local-dir "$DATA_ROOT" \
        --include "video_annotations/*" "videos.zip"
    if [ ! -d "$DATA_ROOT/videos" ]; then
        echo "Unzipping videos.zip ..."
        unzip -q "$DATA_ROOT/videos.zip" -d "$DATA_ROOT"
    fi
fi

echo "Data ready under: $DATA_ROOT"
echo "  annotations: $(ls "$DATA_ROOT/annotations" 2>/dev/null | wc -l) files"
echo "  images dir : $(ls "$DATA_ROOT/images" 2>/dev/null)"
[ "$WITH_VIDEO" = "1" ] && echo "  videos dir : $(ls "$DATA_ROOT/videos" 2>/dev/null)"
