# -*- coding: utf-8 -*-
"""
ckax_common.py
============
Shared helpers used by every CKA-X script in this package.

    RESULTS_DIR / DERIVED_DIR      where named artifacts are read from and
                                   written to (results/reference, results/derived)
    ensure_results_dir / ensure_derived_dir
    resolve_data_path(name)        locate features_diverse/ etc. next to the
                                   package or one level up
    list_feature_tokenizers(dir)   the encoders present in a feature directory
    spearman(a, b)                 rank correlation used by the LOO selection
    load_ground_truth()            re-exported from config (the labels y_m)

Paths are env-overridable; see README ("Labels and paths").
"""


import os
import sys
from pathlib import Path


import numpy as np
from scipy import stats as sps


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)


from config import load_ground_truth, PATHS  # noqa: E402

# --- where the scripts read and write their named artifacts.
#     RESULTS_DIR   = the archived reference outputs consumed as inputs and
#                     diffed by verify_ckax.py (results/reference/).
#     DERIVED_DIR   = anything this package computes on the fly
#                     (results/derived/), so a run never overwrites the
#                     reference copies.
#     CKA_X_RESULTS_DIR overrides RESULTS_DIR (the rebuild stages point it at
#     results/rebuild/).  Defaults unchanged on the original layout.


RESULTS_DIR = (os.environ.get("CKA_X_RESULTS_DIR")
               or os.path.join(PATHS["results"], "reference"))


DERIVED_DIR = os.path.join(PATHS["results"], "derived")


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def ensure_derived_dir():
    os.makedirs(DERIVED_DIR, exist_ok=True)
    return DERIVED_DIR


def resolve_data_path(name):
    """Locate a data directory inside this package (nothing outside it).

    Returns PROJECT_ROOT/<name>; pass an absolute path or set the matching
    env var (FEAT_DIR, IMG_DIR, SAMPLE_DIR, ...) to use data stored elsewhere.
    """
    cand = os.path.join(os.path.dirname(SCRIPT_DIR), name)
    return cand

# ------------------------------------------------------------------
# Series mapping
# ------------------------------------------------------------------


def list_feature_tokenizers(feature_dir):
    """Lightweight enumeration of a features_diverse-style dir
    (no tensor loading)."""
    feature_dir = Path(feature_dir)
    toks = []
    if not feature_dir.exists():
        return toks
    for td in sorted(feature_dir.iterdir()):
        if td.is_dir() and (td / "visual_features.pt").exists():
            toks.append(td.name)
    return sorted(toks)


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(sps.spearmanr(a, b)[0])


# ------------------------------------------------------------------
# Backbones (the label columns)
# ------------------------------------------------------------------
# The names the reference figure was drawn for; a label file may define
# others, and nothing here requires this particular set.
KNOWN_BACKBONES = ("qwen3", "qwen25", "smollm2")


def label_backbones(gt, preferred=KNOWN_BACKBONES):
    """Backbone names defined by a label table, in a stable order.

    The `preferred` names come first (when the labels carry them), then any
    other backbone the labels define, so an evaluation is never tied to one
    experiment's backbone set."""
    present = []
    for v in gt.values():
        for b in ((v or {}).get("scores") or {}):
            if b not in present:
                present.append(b)
    return ([b for b in preferred if b in present]
            + [b for b in present if b not in preferred])
