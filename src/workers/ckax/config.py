# -*- coding: utf-8 -*-
"""
CKA-X — configuration
=================================================================
Configuration for the CKA-X package:

  * ``PATHS``            — project layout + ground-truth location
  * ``BENCHMARKS``       — the eleven downstream benchmarks whose unweighted
                           mean over the MLLM built from ``E_m`` + backbone
                           ``L`` defines the label ``y_m``
  * ``load_ground_truth`` — the label loader used by every script

Ground truth
------------
The paper's 210 labels are bundled in ``resources/ground_truth.json`` and
configured by ``local.yaml``. Custom label files remain supported through
``CKA_X_GT``, e.g.::

         # bash
         CKA_X_GT=/data/benchmarks/ground_truth.json \
             python scripts/run_budget_sweep.py
         # PowerShell
         $env:CKA_X_GT = "D:\\data\\ground_truth.json"

Any of the three backbones' columns may be missing for individual encoders;
the scripts intersect the pool with the backbones that have a score.
"""
import os
import json
from typing import Dict
from vision_encoder_eval.core.runtime import asset_path


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Every path is env-overridable and defaults to a location *inside this
# package*, so nothing here points outside the folder.  `bash run.sh
# preflight` resolves and prints all of them; README section "Labels and
# paths" lists every variable.
PATHS = {
    "project_root": PROJECT_ROOT,

    # --- labels (user-supplied; see the module docstring) ------------------
    "ground_truth_dir": os.path.join(PROJECT_ROOT, "ground_truth"),
    "ground_truth_json": (os.environ.get("CKA_X_GT")
                          or asset_path('ground_truth')),

    # --- outputs ---------------------------------------------------------
    "results": os.environ.get('VEE_CKAX_RESULTS') or asset_path('runtime', 'ckax/results'),

    # --- upstream assets, needed only by the full pipeline ---------------
    # calibration set sources (scripts/prepare_images.py); both default to a
    # location inside this package, override to use your own copies
    "lmu_data": os.environ.get("LMU_DATA",
                               asset_path('lmudata')),
    "ocr_vqa_cache": os.environ.get(
        "OCR_VQA_CACHE", asset_path('ocr_vqa_cache')),
}


# ============================================================
# The eleven downstream benchmarks
# ============================================================
# y_m = unweighted mean of these eleven scores for the MLLM built from E_m
# with language backbone L.  Only these averages enter as labels; the
# per-benchmark numbers are not needed here.
BENCHMARKS = [
    "MMMU_TEST",
    "MMBench_TEST_EN_V11",
    "VQAv2_VAL",
    "ScienceQA_VAL",
    "ChartQA_TEST",
    "DocVQA_VAL",
    "TextVQA_VAL",
    "POPE",
    "GQA_TestDev_Balanced",
    "MSCOCO_KARPATHY_TEST",
    "FLICKR30K_KARPATHY_TEST",
]


# ============================================================
# Ground truth
# ============================================================
def load_ground_truth() -> Dict[str, Dict]:
    """Load the label table ``{tokenizer: {"family":..., "scores":{bb: y}}}``.

    Location = ``$CKA_X_GT`` if set, else the configured ground-truth asset.
    Returns ``{}`` (with a warning) when the file is absent, which is the
    expected state right after unpacking this package — see the module
    docstring for how to supply the real labels.
    """
    gt_path = PATHS["ground_truth_json"]
    if os.path.exists(gt_path):
        with open(gt_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if raw.get('schema_version') == 1 and 'encoders' in raw:
            from vision_encoder_eval.data.ground_truth import validate_ground_truth
            validate_ground_truth(raw)
            return {name: {'family': row['family'],
                           'scores': {llm: item['average'] for llm, item in row['llms'].items()}}
                    for name, row in raw['encoders'].items()}
        # Only dict entries are encoders: the example file (and users' own
        # copies of it) may carry a top-level "_comment" documentation key.
        gt = {t: v for t, v in raw.items() if isinstance(v, dict)}
        if len(gt) != len(raw):
            print("[INFO] ignored %d non-encoder key(s) in %s"
                  % (len(raw) - len(gt), gt_path))
        return gt
    print("[WARN] Ground truth not found at %s" % gt_path)
    print("       Drop the real ground_truth.json there, or set CKA_X_GT "
          "to its path.")
    print("       Format reference: ground_truth/ground_truth.example.json")
    return {}
