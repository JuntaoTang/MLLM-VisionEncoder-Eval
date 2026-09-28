#!/usr/bin/env bash
# Fill missing discrete A/C, then evaluate AC Policy k'=8 on 70 tokenizers × 3 LLMs.
set -u
ROOT="/home/ma-user/work_space/Law_of_Vision_Representation_in_MLLMs/reproduce"
VTB="/home/ma-user/work_space/VTB"
PY="${PY:-/home/ma-user/miniconda3/bin/python}"
export VTB_ROOT="$VTB"
export PYTHONPATH="${VTB}/third_party/LLaVA-NeXT:${VTB}:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME="${VTB}/scripts/cuda_stub" DS_SKIP_CUDA_CHECK=1
mkdir -p "$ROOT/logs" "$ROOT/results"
cd "$VTB"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$ROOT/logs/gt70_orchestrate.log"; }

IDS=(toklip_l_384 toklip_s_256 unitok_attn vilau_256 uniar_bsq)

log "=== discrete C-score feature extract (5 tokenizers) ==="
pids=()
for i in 0 1 2 3 4; do
  vid="${IDS[$i]}"
  CUDA_VISIBLE_DEVICES=$i "$PY" "$ROOT/extract_c_features_discrete.py" --vision-id "$vid" --batch-size 2 \
    >"$ROOT/logs/c_extract_${vid}.log" 2>&1 &
  pids+=($!)
  log "extract $vid gpu=$i pid=${pids[-1]}"
done
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    log "ERROR extract pid $pid failed"
    fail=1
  fi
done
if [[ "$fail" -ne 0 ]]; then
  log "C extract had failures; continuing to score whatever exists"
fi

log "=== discrete C-score PCK ==="
pids=()
for i in 0 1 2 3 4; do
  vid="${IDS[$i]}"
  CUDA_VISIBLE_DEVICES=$i "$PY" "$ROOT/compute_c_score.py" --vision-id "$vid" --shard $((80 + i)) \
    >"$ROOT/logs/c_score_${vid}.log" 2>&1 &
  pids+=($!)
  log "pck $vid gpu=$i pid=${pids[-1]}"
done
for pid in "${pids[@]}"; do
  wait "$pid" || log "ERROR pck pid $pid failed"
done

log "=== discrete A-score 15 models on 8 GPUs ==="
pids=()
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i "$PY" "$ROOT/compute_a_score_discrete.py" --shard $i --nshards 8 \
    >"$ROOT/logs/a_score_disc_${i}.log" 2>&1 &
  pids+=($!)
  log "a-disc shard $i pid=${pids[-1]}"
done
for pid in "${pids[@]}"; do
  wait "$pid" || log "ERROR a-disc pid $pid failed"
done

log "=== AC Policy k'=8 protocol on 70×3 ==="
"$PY" "$ROOT/eval_k8_protocol.py" | tee -a "$ROOT/logs/gt70_orchestrate.log"
log "done"
