#!/usr/bin/env bash
# Fill missing A-scores (25 failed Qwen2.5 + 2 new Pixio) and C-scores (2 new Pixio).
set -u
ROOT="/home/ma-user/work_space/Law_of_Vision_Representation_in_MLLMs/reproduce"
VTB="/home/ma-user/work_space/VTB"
PY="${PY:-/home/ma-user/miniconda3/bin/python}"
export VTB_ROOT="$VTB"
export PYTHONPATH="${VTB}/third_party/LLaVA-NeXT:${VTB}:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME="${VTB}/scripts/cuda_stub"
export DS_SKIP_CUDA_CHECK=1
mkdir -p "$ROOT/logs" "$ROOT/results" /cache/VTB/law_ac/features
cd "$VTB"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$ROOT/logs/fill_missing.log"; }

log "=== C-feature extract for pixio_vith16 / pixio_vitl16 ==="
CUDA_VISIBLE_DEVICES=3 "$PY" "$ROOT/extract_c_features.py" --vision-id pixio_vith16 --shard 91 --batch-size 8 \
  >"$ROOT/logs/c_extract_pixio_vith16.log" 2>&1 &
pid_ch=$!
CUDA_VISIBLE_DEVICES=4 "$PY" "$ROOT/extract_c_features.py" --vision-id pixio_vitl16 --shard 90 --batch-size 16 \
  >"$ROOT/logs/c_extract_pixio_vitl16.log" 2>&1 &
pid_cl=$!

log "=== A-score missing-only on GPUs 0,1,2,5,7 ==="
a_pids=()
gpus=(0 1 2 5 7)
for i in 0 1 2 3 4; do
  gpu=${gpus[$i]}
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/compute_a_score.py" --missing-only --shard $i --nshards 5 --batch-size 8 \
    >"$ROOT/logs/a_score_fill${i}.log" 2>&1 &
  a_pids+=($!)
done

fail=0
for pid in "${a_pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
    log "A-score PID $pid failed"
  fi
done
log "A-score workers finished fail=$fail"

if ! wait "$pid_ch"; then log "C-extract pixio_vith16 failed"; fail=1; fi
if ! wait "$pid_cl"; then log "C-extract pixio_vitl16 failed"; fail=1; fi
log "C-extract finished"

log "=== C-score PCK for two new Pixio encoders ==="
CUDA_VISIBLE_DEVICES=3 "$PY" "$ROOT/compute_c_score.py" --vision-id pixio_vith16 --shard 91 --device cuda:0 \
  >"$ROOT/logs/c_score_pixio_vith16.log" 2>&1 &
pid_cs1=$!
CUDA_VISIBLE_DEVICES=4 "$PY" "$ROOT/compute_c_score.py" --vision-id pixio_vitl16 --shard 90 --device cuda:0 \
  >"$ROOT/logs/c_score_pixio_vitl16.log" 2>&1 &
pid_cs2=$!
if ! wait "$pid_cs1"; then log "C-score pixio_vith16 failed"; fail=1; fi
if ! wait "$pid_cs2"; then log "C-score pixio_vitl16 failed"; fail=1; fi

log "=== re-fit AC ==="
"$PY" "$ROOT/fit_ac.py" 2>&1 | tee "$ROOT/logs/fit_ac.log"
log "=== fill_missing done fail=$fail ==="
exit $fail
