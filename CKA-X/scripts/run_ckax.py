# -*- coding: utf-8 -*-
"""
run_ckax.py
=================
CKA-X driver: the main method plus the descriptor / learner / cross-modal /
kernel-ridge variants, all under one shared k-fold (fold-mean + pooled
aggregations).

One driver, several presets:

    main      CKA-X (C + X) with X = {cm_cka, cm_r2}, fixed w=2 alpha=0.1
    combos    C only  /  X only  /  C+X  (descriptor ablation)
    xvariants for each X statistic s:  C + {s}   (cross-modal variants)
    learners  KRR / ridge-on-similarity-profile / kNN / GBDT / MLP, all on
              the same C+X information
    krr       w/alpha selection: fixed-fixed / LOO-LOO / fixed w + LOO alpha
              / LOO w + fixed alpha (kernel-ridge variants)

Descriptor inputs:
    C = --cka_pt  (the 80x80 pairwise CKA kernel)
    X = --cm_csv  (the cross-modal columns; the descriptor uses cm_cka + cm_r2)

The cross-modal table used here is
artifacts/crossmodal_stats_final_qwen25_full.csv for all three targets (README
§10).

Evaluation = shared k-fold, seed 42, every eligible encoder held out once.
Each cell is reported under both aggregations: fold-mean (rho / r / Top-1,
mean +- std) and pooled (all held-out predictions of all folds concatenated).

Usage (from the CKA-X/ root):
    python scripts/run_ckax.py --preset main --learner gbdt \
        --cm_csv artifacts/crossmodal_stats_final_qwen25_full.csv \
        --cka_pt results/rebuild/cka_diverse.pt \
        --targets qwen3 qwen25 smollm2 \
        --out results/derived/ckax_main_gbdt.json
NOTE: English output only.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ckax_common import (  # noqa: E402
    load_ground_truth, label_backbones, RESULTS_DIR, ensure_results_dir,
)
from ckax_core import (  # noqa: E402
    _spearman, _pearson, _top1_gt, predict_te,
    ALPHAS, kernel_loo_preds, select_alpha_by_loo,
)
from ckax_folds import (  # noqa: E402
    kfold_folds, run_ridge_fold, run_knn_fold, run_mlp_fold,
)

X_STATS = ["cm_cka", "cm_r2", "cm_cca", "cm_procrustes",
           "cm_mutual_knn", "cm_retrieval_r1"]
MAIN_COLS = ["cm_cka", "cm_r2"]


# ----------------------------------------------------------------------
def load_cm_cols(csv_path, cols):
    """{tok: (d,)-vector} for the given columns (None = all numeric).

    A column that the CSV does not carry is reported and dropped: the shipped
    artefact holds only cm_cka/cm_r2 (plus their LMU/OCR subsets), so a
    descriptor asking for e.g. cm_cca would otherwise be silently empty and the
    variant would quietly degenerate into C-only."""
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    allcols = [k for k in rows[0].keys() if k != "tokenizer"]
    if cols:
        absent = [c for c in cols if c not in allcols]
        if absent:
            print("  [WARN] %s has no column(s): %s"
                  % (os.path.basename(csv_path), ", ".join(absent)))
        if not [c for c in cols if c in allcols]:
            raise SystemExit("  [ERROR] none of the requested X columns exist "
                             "in %s\n          requested: %s\n"
                             "          available: %s"
                             % (csv_path, cols, allcols))
    keep = [c for c in (cols or allcols) if c in allcols]
    data = {}
    for r in rows:
        t = (r.get("tokenizer") or "").strip()
        if not t:
            continue
        vec = []
        ok = True
        for c in keep:
            v = r.get(c, "").strip()
            if v in ("", "-", "None", "nan"):
                ok = False
                break
            try:
                vec.append(float(v))
            except ValueError:
                ok = False
                break
        if ok:
            data[t] = np.array(vec, dtype=np.float64)
    return data, keep


def load_cka_kernel(pt_path, key=None):
    import torch
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    if key is not None:
        K = np.asarray(d["kernels"][key], dtype=np.float64)
        toks = list(d["toks"])
    else:
        ck = d.get("cka") or d
        K = np.asarray(ck["kernel"], dtype=np.float64)
        toks = list(ck["toks"])
    return toks, K


def predict_cka_only(Ktr, Kte, ytr, alpha):
    W = np.linalg.solve(Ktr + alpha * np.eye(len(ytr)), ytr)
    return Kte @ W


def loo_alpha(Ktr, ytr):
    loo = kernel_loo_preds(Ktr, ytr, ALPHAS)
    a, sp = select_alpha_by_loo(loo, ytr)
    return a if a is not None else 0.1


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


# ----------------------------------------------------------------------
def fold_eval(X, K, ymap, pool, n_folds, fold_seed, idx_tok, learner,
              w_fixed, alpha_fixed, use_cka):
    """One learner x config, evaluated on the shared k-fold. Returns
    {n_pool, fold_mean{rho,pearson,top1_gt}, pooled{...}, per_fold[...]}."""
    rows, pred_all, y_all = [], [], []
    for fi in kfold_folds(pool, n_folds, fold_seed):
        fi = list(fi)
        tr = [i for i in range(len(pool)) if i not in set(fi)]
        tr_i = np.array([idx_tok[pool[i]] for i in tr])
        te_i = np.array([idx_tok[pool[i]] for i in fi])
        ytr = np.array([ymap[pool[i]] for i in tr], dtype=np.float64)
        yte = np.array([ymap[pool[i]] for i in fi], dtype=np.float64)
        if np.std(ytr) < 1e-12 or len(fi) < 2:
            continue

        if learner == "krr":
            if X.shape[1] == 0:
                # C-only: no linear term; kernel only, alpha by LOO
                a = alpha_fixed if alpha_fixed is not None else loo_alpha(
                    K[np.ix_(tr_i, tr_i)], ytr)
                p = predict_cka_only(K[np.ix_(tr_i, tr_i)],
                                     K[np.ix_(te_i, tr_i)], ytr, a)
                w = float("nan")
            else:
                p, a, w = predict_te(X[tr_i], X[te_i], ytr,
                                     K[np.ix_(tr_i, tr_i)],
                                     K[np.ix_(te_i, tr_i)],
                                     w_fixed, alpha_fixed)
        else:
            # learner over [X columns, CKA-similarity-to-train profile]
            Xb = np.hstack([X, K[:, tr_i]]) if use_cka else X
            if learner == "ridge_profile":
                p = run_ridge_fold(Xb, tr_i, te_i, ytr, yte)
            elif learner == "knn":
                p = run_knn_fold(Xb, tr_i, te_i, ytr, yte, 3)
            elif learner == "gbdt":
                p = _gbdt(Xb, tr_i, te_i, ytr, yte)
            elif learner == "mlp":
                p = run_mlp_fold(Xb, tr_i, te_i, ytr, yte)
            else:
                raise ValueError(learner)
            a = w = float("nan")

        if p is None or np.isnan(np.asarray(p, dtype=np.float64)).any():
            continue
        p = np.asarray(p, dtype=np.float64)
        rows.append({"fold": len(rows) + 1, "n_eval": len(fi),
                     "rho": _spearman(p, yte), "pearson": _pearson(p, yte),
                     "top1_gt": _top1_gt(p, yte),
                     "alpha": float(a), "w": float(w)})
        pred_all.extend(list(p))
        y_all.extend(list(yte))
    if not rows:
        return None
    out = {"n_pool": len(pool)}
    for m in ("rho", "pearson", "top1_gt"):
        v = np.array([r[m] for r in rows if not np.isnan(r[m])])
        out["fold_mean_%s" % m] = [float(np.mean(v)),
                                   float(np.std(v)) if len(v) > 1 else 0.0]
    P = np.array(pred_all)
    Y = np.array(y_all)
    out["pooled_rho"] = _spearman(P, Y)
    out["pooled_pearson"] = _pearson(P, Y)
    out["pooled_top1"] = _top1_gt(P, Y)
    return out


# ----------------------------------------------------------------------
def run_configs(cfgs, cm, cka_toks, K, gt, pools, targets, n_folds, fold_seed):
    common = sorted(set(cm) & set(cka_toks) & set(gt))
    pos = {t: i for i, t in enumerate(cka_toks)}
    keep = [pos[t] for t in common]
    Kc = K[np.ix_(keep, keep)]
    idx_tok = {t: i for i, t in enumerate(common)}

    results = {}
    for cfg in cfgs:
        name = cfg["name"]
        cols = cfg.get("x_cols")           # list ([] = C-only) or None
        learner = cfg.get("learner", "krr")
        w_fixed = cfg.get("w_fixed")
        alpha_fixed = cfg.get("alpha_fixed")
        use_cka = cfg.get("use_cka", True)
        if cols == []:                      # C-only: no linear covariates
            Xfull = {t: np.zeros(0, dtype=np.float64) for t in common}
            used_cols = []
        else:
            Xfull, used_cols = load_cm_cols(cfg.get("cm_csv"), cols)
        X = np.stack([Xfull.get(t, np.full(len(used_cols), np.nan))
                      for t in common]).astype(np.float64)
        if np.isnan(X).any():
            okc = ~np.isnan(X).any(axis=0)
            X = X[:, okc]
            used_cols = [c for c, k in zip(used_cols, okc) if k]

        for ptag, pool_toks in pools.items():
            for bb in targets:
                ymap = {}
                for t in pool_toks:
                    v = (gt.get(t, {}).get("scores") or {}).get(bb)
                    if v is not None and t in common:
                        ymap[t] = float(v)
                pool = sorted(t for t in pool_toks if t in ymap)
                if len(pool) < n_folds + 3:
                    continue
                r = fold_eval(X, Kc, ymap, pool, n_folds, fold_seed, idx_tok,
                              learner, w_fixed, alpha_fixed, use_cka)
                if r is None:
                    continue
                results["%s|%s|%s" % (name, ptag, bb)] = {
                    "x_cols": used_cols, "learner": learner,
                    "w_fixed": w_fixed, "alpha_fixed": alpha_fixed, **r}
    return results


def build_configs(args, cm_for_target):
    L = args.learner
    cfgs = []
    main_cols = MAIN_COLS
    if args.preset == "main":
        cfgs.append({"name": "C+X", "cm_csv": cm_for_target,
                     "x_cols": main_cols, "w_fixed": args.w_fixed,
                     "alpha_fixed": args.alpha_fixed, "learner": L,
                     "use_cka": True})
    elif args.preset == "combos":
        cfgs += [
            {"name": "C_only", "cm_csv": cm_for_target, "x_cols": [],
             "w_fixed": None, "alpha_fixed": None, "learner": L,
             "use_cka": True},
            {"name": "X_only", "cm_csv": cm_for_target, "x_cols": main_cols,
             "w_fixed": 0.0, "alpha_fixed": 0.1, "learner": L,
             "use_cka": False},
            {"name": "C+X", "cm_csv": cm_for_target, "x_cols": main_cols,
             "w_fixed": args.w_fixed, "alpha_fixed": args.alpha_fixed,
             "learner": L, "use_cka": True},
        ]
    elif args.preset == "xvariants":
        cfgs.append({"name": "C+X", "cm_csv": cm_for_target,
                     "x_cols": main_cols, "w_fixed": args.w_fixed,
                     "alpha_fixed": args.alpha_fixed, "learner": L,
                     "use_cka": True})
        for s in X_STATS:
            cfgs.append({"name": "C+%s" % s, "cm_csv": cm_for_target,
                         "x_cols": [s], "w_fixed": args.w_fixed,
                         "alpha_fixed": args.alpha_fixed, "learner": L,
                         "use_cka": True})
    elif args.preset == "learners":
        for ln in ["krr", "ridge_profile", "knn", "gbdt", "mlp"]:
            cfgs.append({"name": ln, "cm_csv": cm_for_target,
                         "x_cols": main_cols,
                         "w_fixed": args.w_fixed if ln == "krr" else None,
                         "alpha_fixed": args.alpha_fixed if ln == "krr"
                         else None,
                         "learner": ln, "use_cka": True})
    elif args.preset == "krr":
        for tag, w, a in [("w2_a0.1", 2.0, 0.1), ("wLOO_aLOO", None, None),
                          ("w2_aLOO", 2.0, None), ("wLOO_a0.1", None, 0.1)]:
            cfgs.append({"name": tag, "cm_csv": cm_for_target,
                         "x_cols": main_cols, "w_fixed": w,
                         "alpha_fixed": a, "learner": "krr", "use_cka": True})
    else:
        raise SystemExit("unknown preset: %s" % args.preset)
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True,
                    choices=["main", "combos", "xvariants", "learners", "krr"])
    ap.add_argument("--cm_csv", type=str, required=True)
    ap.add_argument("--cka_pt", type=str, default=None,
                    help="the pairwise CKA kernel. Default: artifacts/, else "
                         "results/rebuild/cka_diverse.pt (build it with "
                         "`bash run.sh cka`)")
    ap.add_argument("--kernel_key", type=str, default=None,
                    help="if --cka_pt is a kernel-variants file, "
                         "pick which kernel to use (e.g. cos_gram, svcca)")
    ap.add_argument("--pool_file", nargs="+", default=None,
                    help="optional candidate lists, one name per line. "
                         "Default: derive the pool from the descriptor "
                         "coverage and the labels (the labeled pool of this "
                         "study), reported under an internal pool tag")
    ap.add_argument("--targets", nargs="+", default=None,
                    help="label columns to predict. Default: the backbones "
                         "the label file defines (the reference figure's "
                         "three come first when present)")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=42)
    ap.add_argument("--w_fixed", type=float, default=2.0)
    ap.add_argument("--alpha_fixed", type=float, default=0.1)
    ap.add_argument("--learner", type=str, default="krr",
                    choices=["krr", "ridge_profile", "knn", "gbdt", "mlp"],
                    help="regressor used by main/combos/xvariants presets "
                         "(default krr; set gbdt for the tree-ensemble main)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.cka_pt is None:
        for cand in (os.path.join(repo, "artifacts", "cka_diverse.pt"),
                     os.path.join(repo, "results", "rebuild", "cka_diverse.pt"),
                     os.path.join(RESULTS_DIR, "cka_diverse.pt")):
            if os.path.isfile(cand):
                args.cka_pt = cand
                break
        else:
            args.cka_pt = os.path.join(repo, "results", "rebuild",
                                       "cka_diverse.pt")
    pools = {}
    if args.pool_file:
        for pf in args.pool_file:
            p = pf if os.path.isabs(pf) else os.path.join(repo, pf)
            if not os.path.isfile(p):
                print("  [WARN] pool file missing: %s" % p)
                continue
            tag = os.path.splitext(os.path.basename(p))[0]
            pools[tag] = [l.strip() for l in open(p, encoding="utf-8")
                          if l.strip()]
    else:
        pools = None          # derive below, once the descriptors are loaded

    if args.out is None:
        args.out = os.path.join(ensure_results_dir(),
                                "ckax_%s%s.json" % (
                                    args.preset,
                                    ("_" + args.kernel_key)
                                    if args.kernel_key else ""))

    gt = load_ground_truth()
    if not args.targets:
        args.targets = label_backbones(gt)
        if not args.targets:
            sys.exit("[ERROR] the label file defines no backbone column; "
                     "pass --targets explicitly")
        print("  targets: %s (from the label file)" % "/".join(args.targets))
    cka_toks, K = load_cka_kernel(args.cka_pt, args.kernel_key)
    cm, _ = load_cm_cols(args.cm_csv, None)
    print("  cm tokens: %d   cka tokens: %d   gt: %d"
          % (len(cm), len(cka_toks), len(gt)))

    if pools is None:
        # No candidate list given: derive it (descriptor coverage + a label
        # for every backbone being evaluated), reported under a pool tag.
        pools = {"pool": sorted(
            t for t in (set(cm) & set(cka_toks) & set(gt))
            if all(((gt[t].get("scores") or {}).get(b)) is not None
                   for b in args.targets))}
        print("  [pool] derived %d candidates (descriptors + labels for %s)"
              % (len(pools["pool"]), "/".join(args.targets)))

    cfgs = build_configs(args, args.cm_csv)
    results = run_configs(cfgs, cm, cka_toks, K, gt, pools, args.targets,
                          args.n_folds, args.fold_seed)

    def _rel(path):
        try:
            # keep the exported paths OS-independent (this file may be written
            # on Windows and read anywhere)
            return os.path.relpath(path, repo).replace(os.sep, "/")
        except ValueError:          # different drive (Windows)
            return path

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"preset": args.preset, "cm_csv": _rel(args.cm_csv),
                   "cka_pt": _rel(args.cka_pt), "results": results},
                  f, indent=2)
    print("\n  saved: %s\n" % args.out)

    # table
    print("  rho (fold-mean | pooled)")
    print("  %-18s %-14s %s" % ("cfg", "pool",
                                "".join("%-21s" % b for b in args.targets)))
    for key, r in sorted(results.items()):
        name, ptag, bb = key.split("|")
        fm = r["fold_mean_rho"][0]
        pl = r["pooled_rho"]
        print("  %-18s %-14s %-21s" % (name, ptag, "%.3f | %+.3f" % (fm, pl)))


if __name__ == "__main__":
    main()
