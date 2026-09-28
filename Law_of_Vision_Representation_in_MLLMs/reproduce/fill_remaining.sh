#!/usr/bin/env bash
# Fill remaining Law-of-Vision-Representation A/C scores for finish.json models.
# Shares leftover A100 memory with the ongoing VTB train.
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

LOG="$ROOT/logs/fill_remaining.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

wait_pids() {
  local fail=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      fail=1
      log "PID $pid failed"
    fi
  done
  return $fail
}

# vision_id batch_size
C_JOBS=(
  "pe_lang_g14_448_tiling 1"
  "pe_lang_l14_448 2"
  "webssl_dino1b_full2b_224 4"
  "eupe_convnext_b 8"
  "webssl_dino300m_full2b_224 8"
  "dinov3_vitb16 8"
  "dinov3_vits16 16"
  "eupe_vit_s 16"
  "eupe_convnext_s 16"
  "eupe_convnext_t 16"
  "eupe_vit_t 16"
)

log "=== inventory ==="
"$PY" "$ROOT/inventory.py" 2>&1 | tee -a "$ROOT/logs/inventory.log" | tee -a "$LOG"

# Most leftover memory first (training currently holds ~60-71GB/GPU).
GPUS=(1 4 3 6 7 2 5 0)

log "=== C-feature extract for missing tokenizers ==="
c_pids=()
idx=0
for spec in "${C_JOBS[@]}"; do
  vid=${spec%% *}
  bs=${spec##* }
  gpu=${GPUS[$((idx % 8))]}
  shard=$((110 + idx))
  log "extract $vid gpu=$gpu shard=$shard bs=$bs"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/extract_c_features.py" \
    --vision-id "$vid" --shard "$shard" --batch-size "$bs" \
    >"$ROOT/logs/c_extract_${vid}.log" 2>&1 &
  c_pids+=($!)
  idx=$((idx + 1))
  if (( idx % 8 == 0 )); then
    wait_pids "${c_pids[@]}" || log "WARN: some C-extract jobs failed"
    c_pids=()
  fi
done
if ((${#c_pids[@]})); then
  wait_pids "${c_pids[@]}" || log "WARN: some C-extract jobs failed"
fi

log "=== C-score PCK for missing tokenizers ==="
cs_pids=()
for i in 0 1 2 3 4 5 6 7; do
  gpu=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/compute_c_score.py" \
    --missing-only --shard $i --nshards 8 --device cuda:0 \
    >"$ROOT/logs/c_score_fill${i}.log" 2>&1 &
  cs_pids+=($!)
done
wait_pids "${cs_pids[@]}" || log "WARN: some C-score jobs failed"

log "=== A-score missing-only on 8 GPUs ==="
a_pids=()
for i in 0 1 2 3 4 5 6 7; do
  gpu=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/compute_a_score.py" \
    --missing-only --shard $i --nshards 8 --batch-size 2 \
    >"$ROOT/logs/a_score_remain${i}.log" 2>&1 &
  a_pids+=($!)
done
wait_pids "${a_pids[@]}" || log "WARN: some A-score jobs failed"

log "=== re-fit AC ==="
"$PY" "$ROOT/fit_ac.py" 2>&1 | tee "$ROOT/logs/fit_ac.log" | tee -a "$LOG"
log "=== fill_remaining done ==="
ls -l "$ROOT/results" | tee -a "$LOG"
