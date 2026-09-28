#!/usr/bin/env bash
# Reproduce Law of Vision Representation AC scores on VTB finish.json models.
# Uses 8 GPUs; designed to share leftover A100 memory with an ongoing VTB train.
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
mkdir -p "$ROOT/logs" "$ROOT/results" /cache/data /cache/VTB/law_ac/features
cd "$VTB"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$ROOT/logs/run.log"; }

download_spair() {
  local dst="/cache/data/SPair-71k"
  if [[ -d "$dst/JPEGImages" && -d "$dst/PairAnnotation" ]]; then
    log "SPair-71k already present at $dst"
    return 0
  fi
  mkdir -p /cache/data
  local tar="/cache/data/SPair-71k.tar.gz"
  local old="$ROOT/data/SPair-71k.tar.gz"
  if [[ -s "$old" && ! -s "$tar" ]]; then
    log "Moving partial SPair tarball $old -> $tar"
    mv "$old" "$tar"
  fi
  log "Downloading SPair-71k to $tar (resume ok)"
  curl -L --retry 8 -C - --connect-timeout 30 -o "$tar" "http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz" \
    || curl -L --retry 8 -C - --connect-timeout 30 -o "$tar" "https://huggingface.co/datasets/kmjung/SPair-71k/resolve/main/SPair-71k.tar.gz"
  log "Extracting SPair-71k into /cache/data"
  tar -xf "$tar" -C /cache/data
  if [[ ! -d "$dst/JPEGImages" ]]; then
    local found
    found=$(find /cache/data -type d -name JPEGImages | head -1)
    if [[ -n "$found" ]]; then
      ln -sfn "$(dirname "$found")" "$dst"
    fi
  fi
  [[ -d "$dst/JPEGImages" ]]
}

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

log "=== inventory ==="
"$PY" "$ROOT/inventory.py" 2>&1 | tee "$ROOT/logs/inventory.log"
log "=== download SPair ==="
if ! download_spair; then
  log "FATAL: SPair-71k download/extract failed"
  exit 1
fi

log "=== C-feature extract on 8 GPUs x 2 workers, batch=32 ==="
pids=()
for i in $(seq 0 15); do
  gpu=$((i % 8))
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/extract_c_features.py" --shard $i --nshards 16 --batch-size 32 \
    >"$ROOT/logs/c_extract_shard${i}.log" 2>&1 &
  pids+=($!)
done
wait_pids "${pids[@]}" || log "WARN: some C-extract shards failed (will continue)"

log "=== C-score PCK on 8 GPUs ==="
pids=()
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i "$PY" "$ROOT/compute_c_score.py" --shard $i --nshards 8 --device cuda:0 \
    >"$ROOT/logs/c_score_shard${i}.log" 2>&1 &
  pids+=($!)
done
wait_pids "${pids[@]}" || log "WARN: some C-score shards failed"

log "=== A-score on 8 GPUs x 2 workers, batch=16 (Stage-1, 100 samples) ==="
pids=()
for i in $(seq 0 15); do
  gpu=$((i % 8))
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/compute_a_score.py" --shard $i --nshards 16 --batch-size 16 \
    >"$ROOT/logs/a_score_shard${i}.log" 2>&1 &
  pids+=($!)
done
wait_pids "${pids[@]}" || log "WARN: some A-score shards failed"

log "=== fit AC polynomial vs finish.json ==="
"$PY" "$ROOT/fit_ac.py" 2>&1 | tee "$ROOT/logs/fit_ac.log"
log "=== done ==="
ls -l "$ROOT/results" | tee -a "$ROOT/logs/run.log"
