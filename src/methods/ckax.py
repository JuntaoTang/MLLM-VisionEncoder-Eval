from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr

ALPHAS = np.logspace(-2, 5, 15)
KERNEL_WEIGHTS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]

def spearman(a, b):
    a = np.asarray(a,dtype=np.float64)
    b = np.asarray(b,dtype=np.float64)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(spearmanr(a, b)[0])

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


def _gbdt(Xb, tr_i, te_i, ytr, yte):
    """Conservative GBDT for the tiny-tokenizer-count regime: shallow trees,
    subsampling, few boosting rounds. Hyperparameters are FIXED across folds,
    pools and backbones (no per-fold tuning)."""
    from sklearn.ensemble import GradientBoostingRegressor
    est = GradientBoostingRegressor(
        n_estimators=50, learning_rate=0.05, max_depth=2,
        subsample=0.8, min_samples_leaf=1, random_state=0)
    est.fit(Xb[tr_i], ytr)
    return est.predict(Xb[te_i])
