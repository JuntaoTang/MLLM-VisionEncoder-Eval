# -*- coding: utf-8 -*-
"""
ckax_core.py
============
Shared core of CKA-X: descriptor loading, the kernel-ridge
predictor and the evaluation metrics.

    load_cm(csv_path)        -> ({tokenizer: cross-modal row}, columns)
    load_cka(pt_path)        -> (tokens, pairwise CKA kernel)
    predict_te(...)          -> kernel-ridge prediction (K = K_lin(X) + w*K_CKA)
    _spearman/_pearson/_top1_gt   the three reported metrics
    ALPHAS / KERNEL_WEIGHTS / kernel_loo_preds / select_alpha_by_loo
                             hyper-parameter grids + leave-one-out selection

Used by run_ckax.py (the method driver), run_budget_sweep.py
(the budget study), ckax_folds.py (the fold protocol) and verify_ckax.py.
"""


import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ckax_common import spearman  # noqa: E402


# ---------------------------------------------------------------------------
# Kernel-ridge hyper-parameter grids and leave-one-out selection.
# ---------------------------------------------------------------------------
ALPHAS = np.logspace(-2, 5, 15)
KERNEL_WEIGHTS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]


def loo_score(P):
    """Collapse a LOO prediction matrix to a per-encoder score."""
    return P.mean(axis=1) if P.shape[1] > 1 else P[:, 0]


def kernel_loo_preds(K, Y, alphas):
    """Leave-one-out predictions for every alpha (hat-matrix shortcut)."""
    Y = Y[:, None] if Y.ndim == 1 else Y
    s, U = np.linalg.eigh(K)
    s = np.clip(s, 0, None)
    C = U.T @ Y
    out = {}
    for a in alphas:
        w = s / (s + a)
        Yhat = (U * w) @ C
        h = np.clip((U ** 2) @ w, 0, 0.9999)
        out[a] = (Yhat - h[:, None] * Y) / (1 - h)[:, None]
    return out


def select_alpha_by_loo(loo_preds, y_target):
    """Pick the alpha whose LOO predictions correlate best with the labels."""
    best_a, best_sp = None, -2.0
    for a, P in loo_preds.items():
        sp = spearman(loo_score(P), y_target)
        if not np.isnan(sp) and sp > best_sp:
            best_a, best_sp = a, sp
    return best_a, best_sp


def _spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    from scipy import stats as sps
    return float(sps.spearmanr(a, b)[0])


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _top1_gt(pred, y_te):
    if len(pred) == 0 or np.isnan(pred).any():
        return float("nan")
    return float(y_te[int(np.argmax(pred))])


def load_cm(csv_path):
    """6 cross-modal columns per tokenizer from the final cm csv."""
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = None
    data = {}
    for row in rows:
        t = (row.get("tokenizer") or "").strip()
        if not t:
            continue
        if cols is None:
            cols = [k for k in row.keys()
                    if k != "tokenizer" and _num(row[k]) is not None]
        vec = np.array([_num(row[k]) for k in cols], dtype=np.float64)
        if np.isnan(vec).any():
            continue
        data[t] = vec
    if cols is None:
        raise ValueError(f"no numeric columns found in {csv_path}")
    print(f"  CM csv: {len(data)} tokenizers x {len(cols)} cols: {cols}")
    return data, cols


def _num(v):
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "-", "None", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_cka(pt_path):
    import torch
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    ck = d.get("cka") or d
    toks = list(ck["toks"])
    K = np.asarray(ck["kernel"], dtype=np.float64)
    print(f"  CKA kernel: {len(toks)} tokenizers, shape {K.shape}")
    return toks, K


def _zscore(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd, mu, sd


def predict_te(Xtr_raw, Xte_raw, ytr, Kcka_tr, Kcka_te,
               w_fixed=None, alpha_fixed=None):
    """Fit CKA-X on labeled rows, predict test rows.
    Returns (pred, alpha, w). Standardization inside the labeled set.
    When w_fixed & alpha_fixed are given, skip the per-split LOO selection."""
    Xtr, mu, sd = _zscore(Xtr_raw)
    Xte = (Xte_raw - mu) / sd
    Klin_tr = Xtr @ Xtr.T
    Klin_te = Xte @ Xtr.T
    scale = np.trace(Klin_tr) / max(np.trace(Kcka_tr), 1e-9)
    Kcka_tr = Kcka_tr * scale
    Kcka_te = Kcka_te * scale
    if w_fixed is not None and alpha_fixed is not None:
        a, wgt = alpha_fixed, w_fixed
    else:
        best = (None, None, -2.0)
        for wgt in KERNEL_WEIGHTS:
            Ktr = Klin_tr + wgt * Kcka_tr
            loo = kernel_loo_preds(Ktr, ytr, ALPHAS)
            a, sp = select_alpha_by_loo(loo, ytr)
            if a is not None and (best[2] is None or sp > best[2]):
                best = (a, wgt, sp)
        a, wgt, _ = best
    if a is None:
        return np.full(Xte.shape[0], np.nan), a, wgt
    Ktr = Klin_tr + wgt * Kcka_tr
    W = np.linalg.solve(Ktr + a * np.eye(len(ytr)), ytr)
    pred = (Klin_te + wgt * Kcka_te) @ W
    return pred, a, wgt
