# -*- coding: utf-8 -*-
"""
ckax_folds.py
=============
Shared fold protocol and comparison learners for the learner study.

    kfold_folds(pool, n_folds, seed)   the shared k-fold split; with seed 42
                                       every encoder is held out exactly once
    run_ridge_fold / run_knn_fold / run_gbdt_fold / run_mlp_fold
                                       the comparison learners; each one scales
                                       and regularizes on the training fold only

Imported by run_ckax.py (--preset learners) and by
run_budget_sweep.py.  The CKA-X learner itself is
run_ckax.py::_gbdt (gradient boosting, depth-2 trees, T=50 rounds).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision_encoder_eval.workers.ckax.scripts.ckax_core import _spearman  # noqa: E402



RIDGE_ALPHAS = np.logspace(-2, 5, 15)


def _zscore_cols(X):
    mu = X.mean(0)
    sd = X.std(0) + 1e-8
    return (X - mu) / sd, mu, sd


def kfold_folds(pool, n_folds, seed):
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(pool))
    return [[int(i) for i in fi] for fi in np.array_split(order, n_folds)]


def run_knn_fold(Xm, tr_i, te_i, ytr, yte, k):
    Xtr, mu, sd = _zscore_cols(Xm[tr_i])
    A = Xtr
    B = (Xm[te_i] - mu) / sd
    d2 = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(d2, axis=1)[:, :k]
    w = 1.0 / (d2[np.arange(d2.shape[0])[:, None], nn] + 1e-8)
    return (w * ytr[nn]).sum(1) / w.sum(1)


def run_ridge_fold(Xm, tr_i, te_i, ytr, yte):
    Xtr, mu, sd = _zscore_cols(Xm[tr_i])
    Xte = (Xm[te_i] - mu) / sd
    lam, V = np.linalg.eigh(Xtr.T @ Xtr)
    lam = np.clip(lam, 0, None)
    B = Xtr @ V
    C = V.T @ (Xtr.T @ ytr)
    best = (None, -2.0)
    for a in RIDGE_ALPHAS:
        w = 1.0 / (lam + a)
        yh = (B * w) @ C
        h = np.clip((B ** 2) @ w, 0, 0.9999)
        loo = (yh - h * ytr) / (1 - h)
        sp = _spearman(loo, ytr)
        if not np.isnan(sp) and sp > best[1]:
            best = (a, sp)
    a = best[0]
    if a is None:
        return None
    wcoef = np.linalg.solve(Xtr.T @ Xtr + a * np.eye(Xtr.shape[1]),
                            Xtr.T @ ytr)
    return Xte @ wcoef


def _sklearn_est(name):
    try:
        if name == "gbdt":
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(n_estimators=200,
                                             learning_rate=0.05,
                                             max_depth=3, random_state=0)
        if name == "mlp":
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                early_stopping=True, random_state=0)
    except Exception as e:
        print("  [warn] sklearn unavailable: %s" % e)
    return None


def run_gbdt_fold(Xm, tr_i, te_i, ytr, yte):
    est = _sklearn_est("gbdt")
    if est is None:
        return None
    est.fit(Xm[tr_i], ytr)
    return est.predict(Xm[te_i])


def run_mlp_fold(Xm, tr_i, te_i, ytr, yte):
    est = _sklearn_est("mlp")
    if est is None:
        return None
    Xtr, mu, sd = _zscore_cols(Xm[tr_i])
    est.fit(Xtr, ytr)
    return est.predict((Xm[te_i] - mu) / sd)
