#!/usr/bin/env bash
# =============================================================================
# CKA-X — full Linux pipeline
# =============================================================================
# One entry point for the whole chain: calibration images -> per-encoder visual
# features -> frozen-LLM text features -> descriptor (C: pairwise CKA kernel,
# X: cross-modal alignment) -> gradient-boosted prediction -> budget figure.
#
# --- check / reproduce (no GPU, works from the frozen artifacts/) ---
#   bash run.sh preflight          resolve + check every path, print the gaps
#   bash run.sh verify             recompute the descriptor + prediction +
#                                  budget study and diff against results/reference/
#   bash run.sh figure             budget figure (pooled)   -> figures/
#   bash run.sh sweep              budget-study data         -> results/derived/
#   bash run.sh main               GBDT, shared 5-fold       -> results/derived/
#   bash run.sh learners           learner comparison        -> results/derived/
#
# --- descriptor: rebuild C and X from feature matrices (GPU) ---
#   bash run.sh cka                C: full pairwise linear-CKA kernel
#   bash run.sh crossmodal         X: cross-modal statistics -> the 2 columns
#
# --- upstream: rebuild the feature matrices themselves (GPU) ---
#   bash run.sh text               frozen-LLM text features (text_features_qwen25.pt)
#   bash run.sh features           visual features, the continuous encoders
#   bash run.sh features-discrete  visual features, the discrete encoders
#   bash run.sh images             build the calibration image set
#
# --- whole pipelines ---
#   bash run.sh pipeline           evaluation on the FROZEN artifacts
#                                  (main -> learners -> sweep -> figure -> verify)
#   bash run.sh pipeline-descriptor  rebuild C and X from existing features,
#                                  evaluate on them, then verify vs the archive
#   bash run.sh pipeline-all       everything, from the calibration images up
#   bash run.sh help
#
# Extra arguments after a command are appended to the underlying command, e.g.
#   bash run.sh sweep --n_splits 20          (quick smoke run)
#   bash run.sh verify --skip-sweep
#   bash run.sh figure --caliber fold_mean
#
# Environment (all optional; the defaults live inside this package,
# and `bash run.sh preflight` prints every resolved value):
#   labels     CKA_X_GT
#   data       LMU_DATA, OCR_VQA_CACHE, IMG_DIR, FEAT_DIR, SAMPLE_DIR, TXT_PT,
#              N_IMAGES, OCR_RATIO
#   weights    TOKENIZER_WEIGHTS_ROOT, VTB_CONFIGS_ROOT, VTB_ROOT, DISC_CKPT,
#              VQGAN_PATH, CLIP_DIR, LLM_ROOT
#   plumbing   PY, DEVICE, ART, REBUILD, SWEEP, REF, CKA, LOG_DIR
#
# Rebuilds NEVER overwrite the shipped inputs: cka/crossmodal write
# into results/rebuild/ instead, so `verify` can still diff against the
# archive; pipeline-descriptor/pipeline-all end by verifying the rebuilt
# descriptor against it.
# =============================================================================
set -uo pipefail

PROJ="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ"

PY="${PY:-python}"
DEVICE="${DEVICE:-cuda}"
N_IMAGES="${N_IMAGES:-}"        # size of the calibration set: an input, so it
                                # has no default (see require_n_images)
OCR_RATIO="${OCR_RATIO:-0.3}"
LOG_DIR="${LOG_DIR:-logs}"
ART="${ART:-artifacts}"                       # descriptor inputs for eval stages
REBUILD="${REBUILD:-results/rebuild}"         # where rebuild stages write
GT="${CKA_X_GT:-}"


require_n_images() {
  # The image/feature stages rebuild the calibration set from raw images, so
  # they need its size; nothing in this package assumes a particular one.
  if [ -z "$N_IMAGES" ]; then
    echo "[ERROR] N_IMAGES is not set." >&2
    echo "        export N_IMAGES=<size of your calibration set>" >&2
    return 2
  fi
}


mkdir -p "$LOG_DIR" results/derived "$REBUILD"
FAILED=""

# ---------------------------------------------------------------------------
# data-dir discovery: same rule as ckax_common.resolve_data_path()
# (everything lives inside this package unless the env vars point elsewhere)
# ---------------------------------------------------------------------------
detect() {
  echo "$PROJ/$1"          # package-local by design; override with the env vars
}
FEAT_DIR="${FEAT_DIR:-$(detect features_diverse)}"
SAMPLE_DIR="${SAMPLE_DIR:-$(detect sample_data)}"
IMG_DIR="${IMG_DIR:-$(detect images_diverse)}"
TXT_PT="${TXT_PT:-$SAMPLE_DIR/text_features/text_features_qwen25.pt}"

# these three are re-pointed at $REBUILD by the descriptor pipelines
CM="${CM:-$ART/crossmodal_stats_final_qwen25_full.csv}"
# The pairwise CKA kernel (C) is not included in the package; build it once
# with `bash run.sh cka` (needs FEAT_DIR) — it lands in results/rebuild/.
# An explicit --cka_pt / $CKA always wins.
if [ -z "${CKA:-}" ]; then
  if [ -f "$ART/cka_diverse.pt" ]; then CKA="$ART/cka_diverse.pt"
  else CKA="$REBUILD/cka_diverse.pt"; fi
fi
SWEEP="${SWEEP:-results/reference/budget_sweep.json}"
REF="${REF:-results/reference/fullinfo_gbdt.json}"

# ---------------------------------------------------------------------------
# stage runner: logs to logs/<name>.log, never aborts the pipeline
# ---------------------------------------------------------------------------
run_stage() {
  local name="$1"; shift
  local log="$LOG_DIR/$name.log"
  echo ""
  echo "===================== $(date '+%F %T')  $name  ====================="
  if "$@" 2>&1 | tee "$log"; then
    echo "-- OK: $name   (log: $log)"
  else
    echo "-- FAILED: $name   (log: $log)"
    FAILED="$FAILED $name"
  fi
}

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
stage_preflight() {
  local miss=0
  echo "== CKA-X preflight =="
  echo "  project : $PROJ"
  echo "  python  : $("$PY" -V 2>&1)   ($(command -v "$PY" || echo 'not found'))"
  echo "  device  : $DEVICE"
  echo ""
  echo "-- Python packages --"
  local m
  for m in numpy scipy sklearn matplotlib; do
    if "$PY" -c "import $m" 2>/dev/null; then echo "  [ OK ] $m"
    else echo "  [MISS] $m   (pip install -r requirements.txt)"; miss=$((miss+1)); fi
  done
  if "$PY" -c "import torch" 2>/dev/null; then echo "  [ OK ] torch"
  else echo "  [warn] torch missing: artifacts/*.pt unreadable, the"
       " .npz mirror is used instead"; fi
  for m in open_clip transformers datasets PIL yaml; do
    if "$PY" -c "import $m" 2>/dev/null; then echo "  [ OK ] $m"
    else echo "  [ -- ] $m   (only the upstream stages need it)"; fi
  done

  echo ""
  echo "-- Labels (the target scores y_m) --"
  local gtp="$PROJ/ground_truth/ground_truth.json"
  [ -n "$GT" ] && gtp="$GT"
  if [ -f "$gtp" ]; then
    echo "  [ OK ] $gtp"
    "$PY" - "$gtp" <<'PYEOF' || true
import json, sys
raw = json.load(open(sys.argv[1], encoding="utf-8"))
gt = {t: v for t, v in raw.items() if isinstance(v, dict)}
if len(gt) != len(raw):
    print('         (ignored %d non-encoder key(s), e.g. "_comment")'
          % (len(raw) - len(gt)))
print("         %d encoders in the label file; coverage per backbone:" % len(gt))
for bb in ("qwen3", "qwen25", "smollm2"):
    n = sum(1 for t, v in gt.items()
            if (v.get("scores") or {}).get(bb) is not None)
    print("           %-8s %d" % (bb, n))
PYEOF
  else
    echo "  [MISS] $gtp   (supply it as described in README §7)"
    echo "         set CKA_X_GT=/path/to/ground_truth.json"; miss=$((miss+1))
  fi

  echo ""
  echo "-- Inputs the rebuild stages need (not required for evaluation) --"
  if [ -n "$N_IMAGES" ]; then
    echo "  [ OK ] N_IMAGES=$N_IMAGES"
  else
    echo "  [ -- ] N_IMAGES not set"
    echo "         the images/features stages rebuild the calibration set, so"
    echo "         they need its size:  export N_IMAGES=<number of images>"
  fi

  echo ""
  echo "-- Descriptor inputs --"
  if [ -f "$CKA" ]; then
    echo "  [ OK ] C (kernel): $CKA"
  else
    echo "  [ -- ] C (kernel): not built yet"
    echo "         the pairwise CKA kernel is not included in this package."
    echo "         Build it with:  bash run.sh cka"
    echo "         (needs $FEAT_DIR).  Every evaluation stage needs it;"
    echo "         'bash run.sh figure' does not."
  fi
  local f
  for f in "$CM" "results/reference/budget_sweep.json"; do
    if [ -f "$f" ]; then echo "  [ OK ] $f"; else echo "  [MISS] $f"; miss=$((miss+1)); fi
  done

  echo ""
  echo "-- Rebuild assets (only the upstream/descriptor stages need these) --"
  local p
  for p in "$FEAT_DIR" "$IMG_DIR" "$SAMPLE_DIR" "$TXT_PT" \
           "${TOKENIZER_WEIGHTS_ROOT:-$PROJ/tokenizer/continuous}" \
           "${VTB_CONFIGS_ROOT:-$PROJ/configs/continuous/vision_encoder}" \
           "${DISC_CKPT:-$PROJ/tokenizer/discrete}" \
           "${VQGAN_PATH:-${DISC_CKPT:-$PROJ/tokenizer/discrete}/toklip/vq_ds16_t2i.pt}" \
           "${VTB_ROOT:-$PROJ/UniTok}" \
           "${LLM_ROOT:-$PROJ/llm}" \
           "${LMU_DATA:-$PROJ/LMUData}" \
           "${OCR_VQA_CACHE:-$PROJ/ocr-vqa}"; do
    if [ -e "$p" ]; then echo "  [ OK ] $p"; else echo "  [ -- ] $p"; fi
  done

  echo ""
  if [ "$miss" -gt 0 ]; then
    echo "  preflight: $miss blocking item(s) -- fix the [MISS] lines above"
    return 2
  fi
  echo "  preflight: all blocking items present"
}

# ---------------------------------------------------------------------------
# upstream stages
# ---------------------------------------------------------------------------
stage_images() {
  require_n_images || return 2
  "$PY" scripts/prepare_images.py --source diverse --num_images "$N_IMAGES" \
    --output_dir "$IMG_DIR" --ocr_ratio "$OCR_RATIO" "$@"
}
stage_features() {
  require_n_images || return 2
  # no --tokenizers: extract every encoder the config directory defines; the
  # evaluation pool is then derived from the labels (see README, "the pool")
  "$PY" scripts/extract_features.py --image_dir "$IMG_DIR" \
    --num_images "$N_IMAGES" --output_dir "$FEAT_DIR" \
    --device "$DEVICE" --batch_size 64 --skip_text "$@"
}
stage_features_discrete() {
  require_n_images || return 2
  local ref="$FEAT_DIR/clip_openai__l14/image_paths.txt"
  [ -f "$ref" ] || ref=""
  "$PY" scripts/extract_discrete_features.py --image_dir "$IMG_DIR" \
    --num_images "$N_IMAGES" --output_dir "$FEAT_DIR" \
    ${ref:+--align_to "$ref"} \
    --device "$DEVICE" "$@"
}
stage_text() {
  "$PY" scripts/extract_qwen_text_features.py --diverse_dir "$FEAT_DIR" \
    --out_dir "$SAMPLE_DIR" --encoder qwen25 --device "$DEVICE" "$@"
}

# ---------------------------------------------------------------------------
# descriptor stages (write into $REBUILD via CKA_X_RESULTS_DIR)
# ---------------------------------------------------------------------------
stage_cka() {
  "$PY" scripts/compute_cka_kernel.py --feature_dir "$FEAT_DIR" --full \
    --out "$REBUILD/cka_diverse.pt" --device "$DEVICE" "$@"
}
stage_crossmodal() {
  CKA_X_RESULTS_DIR="$REBUILD" "$PY" scripts/compute_crossmodal_stats.py \
    --diverse_dir "$FEAT_DIR" --text_features_pt "$TXT_PT" --full_cka \
    --out_name crossmodal_stats_qwen25_full --device "$DEVICE" "$@" \
  && CKA_X_RESULTS_DIR="$REBUILD" "$PY" scripts/make_cm_final.py \
    --src "$REBUILD/crossmodal_stats_qwen25_full.csv" \
    --out_name crossmodal_stats_final_qwen25_full
}

# ---------------------------------------------------------------------------
# evaluation stages
# ---------------------------------------------------------------------------
stage_verify() {
  "$PY" scripts/verify_ckax.py ${GT:+--gt "$GT"} \
    --cm_csv "$CM" --cka_pt "$CKA" "$@"
}
stage_main() {
  "$PY" scripts/run_ckax.py --preset main --learner gbdt \
    --cm_csv "$CM" --cka_pt "$CKA" \
    --out results/derived/ckax_main_gbdt.json "$@"
}
stage_learners() {
  "$PY" scripts/run_ckax.py --preset learners \
    --cm_csv "$CM" --cka_pt "$CKA" \
    --out results/derived/learner_comparison.json "$@"
}
stage_sweep() {
  "$PY" scripts/run_budget_sweep.py --cm_csv "$CM" --cka_pt "$CKA" \
    --out_json results/derived/budget_sweep.json \
    --out_csv  results/derived/budget_sweep.csv \
    --out_md   results/derived/budget_sweep.md "$@"
}
stage_figure() {
  "$PY" scripts/plot_budget_sweep.py --caliber pooled \
    --outdir figures --basename budget_sweep \
    --sweep "$SWEEP" --ref "$REF" "$@"
}

summary() {
  echo ""
  echo "===================== $(date '+%F %T')  DONE  ====================="
  if [ -n "$FAILED" ]; then
    echo "  FAILED stages:$FAILED"
    echo "  logs: $LOG_DIR/"
    return 1
  fi
  echo "  all requested stages OK"
}

# ---------------------------------------------------------------------------
CMD="${1:-help}"
[ $# -gt 0 ] && shift

case "$CMD" in
  preflight)          stage_preflight ;;
  images)             run_stage images            stage_images "$@" ;;
  features)           run_stage features          stage_features "$@" ;;
  features-discrete)  run_stage features_discrete stage_features_discrete "$@" ;;
  text)               run_stage text              stage_text "$@" ;;
  cka)                run_stage cka               stage_cka "$@" ;;
  crossmodal)         run_stage crossmodal        stage_crossmodal "$@" ;;
  verify)             stage_verify "$@" ;;
  main)               run_stage main              stage_main "$@" ;;
  learners)           run_stage learners          stage_learners "$@" ;;
  sweep)              run_stage sweep             stage_sweep "$@" ;;
  figure)             stage_figure "$@" ;;

  # ---- evaluation on the frozen artifacts (no GPU, ~2 min) ----------------
  pipeline)
    stage_preflight || true
    run_stage main      stage_main
    run_stage learners  stage_learners
    run_stage sweep     stage_sweep
    SWEEP=results/derived/budget_sweep.json
    REF=results/derived/ckax_main_gbdt.json
    stage_figure
    stage_verify
    summary
    exit $?
    ;;

  # ---- rebuild C and X from existing features, then evaluate --------
  pipeline-descriptor)
    stage_preflight || true
    run_stage cka        stage_cka
    run_stage crossmodal stage_crossmodal
    ART="$REBUILD"
    CM="$REBUILD/crossmodal_stats_final_qwen25_full.csv"
    CKA="$REBUILD/cka_diverse.pt"
    run_stage main      stage_main
    run_stage learners  stage_learners
    run_stage sweep     stage_sweep
    SWEEP=results/derived/budget_sweep.json
    REF=results/derived/ckax_main_gbdt.json
    stage_figure
    echo ""
    echo "== verify the REBUILT descriptor against the archive (results/reference/) =="
    stage_verify
    summary
    exit $?
    ;;

  # ---- everything, from the calibration images upward ---------------------
  pipeline-all)
    stage_preflight || true
    run_stage images            stage_images
    run_stage features          stage_features
    run_stage features_discrete stage_features_discrete
    run_stage text              stage_text
    "$0" pipeline-descriptor
    exit $?
    ;;

  help|*)
    sed -n '2,54p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
