# -*- coding: utf-8 -*-
"""
run_budget_sweep.py
===========================
Labeling-budget study of CKA-X: how the held-out correlation grows with
the number of labeled encoders.

Method:
    descriptor  [k_m, x_m] in R^{|R|+2}
        k_m : CKA similarities of the candidate to the reference pool R
              -> in code: the columns of the pairwise CKA kernel indexed by the
                 reference (labeled) rows, Kc[:, tr_i]
        x_m : cross-modal alignment, 2 scalars (cm_cka, cm_r2)
    learner     gradient-boosted shallow trees, hyperparameters FIXED across
                folds/pools/backbones (no per-fold tuning):
                n_estimators=50, learning_rate=0.05, max_depth=2,
                subsample=0.8, min_samples_leaf=1, random_state=0
                -> identical to run_ckax.py::_gbdt (the function the
                   study names as the reported GBDT), NOT
                   ckax_folds.py::run_gbdt_fold (200 trees, depth 3).

Artifacts = the configuration used throughout:
    C : results/reference/cka_diverse.pt              (all calibration images, every encoder)
    X : results/reference/crossmodal_stats_final_qwen25_full.csv
        unified frozen Qwen2.5-1.5B text encoder; columns cm_cka + cm_r2
    pool: derived from the descriptor coverage + the labels; that is the
          labeled encoders available here.  An
          explicit --pool_file overrides it.

Protocol:
    - N candidate encoders; per split randomly select k' as labeled
      (reference) set, fit on their observed Avg scores, predict the N-k'
      remaining encoders
    - random seed 0, 100 splits per k'
    - the SAME 100 splits for every language backbone, models fitted
      independently
    - metrics on the held-out encoders: Spearman, Pearson, Top-1
      (Top-1 = GT score of the highest-ranked held-out encoder)

Two calibers per metric:
    fold_mean : per-split metric averaged over the 100 splits (mean +- std)
    pooled    : all held-out predictions of all splits concatenated, then a
                single metric over n_splits * (N - k') points

Diagnostics:
    --self_mode keep (default: the descriptor as defined)
        training rows keep their own CKA diagonal entry K[m, m] = 1, exactly as
        the descriptor ("the same reference pool serves the candidates
        used to fit the predictor and the held-out candidate alike").
    --self_mode zero
        diagnostic only: the self entry of every *training* row is set to 0, so
        the trees cannot read the candidate's own identity off the profile.
        Diagnostic only; use it to size the self-similarity effect.

Usage (from the CKA-X/ root):
    python scripts/run_budget_sweep.py
    python scripts/run_budget_sweep.py --self_mode zero \
        --out_json results/derived/budget_sweep_selfzero.json \
        --out_csv  results/derived/budget_sweep_selfzero.csv \
        --out_md   results/derived/budget_sweep_selfzero.md
Console output is ASCII only.

Paths in this package: the inputs default to results/reference/<name> and fall
back to artifacts/<name>; the OUTPUTS default to results/derived/, so a run
never overwrites the reference copies that verify_ckax.py diffs against.
`bash run.sh sweep` issues exactly this.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ckax_common import (  # noqa: E402
    load_ground_truth, label_backbones, RESULTS_DIR, DERIVED_DIR,
)
from ckax_core import _spearman, _pearson, _top1_gt  # noqa: E402
from run_ckax import (  # noqa: E402
    load_cm_cols, load_cka_kernel, _gbdt, MAIN_COLS,
)
from ckax_folds import (  # noqa: E402
    run_ridge_fold, run_knn_fold, run_mlp_fold,
)

# Preferred column order for the reference tables; any other backbone the
# label file defines is appended after these.
BACKBONE_ORDER = ["qwen3", "qwen25", "smollm2"]
DEFAULT_K_SWEEP = "8,16,24,32,40,48,56,64"
METRICS = ("spearman", "pearson", "top1")


# ----------------------------------------------------------------------
def make_splits(pool, k_list, n_splits, seed, mode):
    """Splits: {k: [(labeled, held-out), ...]}, shared by all
    backbones. mode='nested' re-seeds per k, so split s uses the same
    permutation for every budget (labeled sets are nested across k')."""
    out = {}
    if mode == "independent":
        rng = np.random.RandomState(seed)
        for k in k_list:
            rows = []
            for _ in range(n_splits):
                perm = list(pool)
                rng.shuffle(perm)
                rows.append((perm[:k], perm[k:]))
            out[k] = rows
    else:
        for k in k_list:
            rng = np.random.RandomState(seed)
            rows = []
            for _ in range(n_splits):
                perm = list(pool)
                rng.shuffle(perm)
                rows.append((perm[:k], perm[k:]))
            out[k] = rows
    return out


def descriptor(X, Kc, tr_i, self_mode):
    """[x_m (2 cross-modal cols), k_m (CKA profile over the reference pool)]."""
    Xb = np.hstack([X, Kc[:, tr_i]])
    if self_mode == "zero":
        off = X.shape[1]
        for j, i in enumerate(tr_i):
            Xb[i, off + j] = 0.0
    return Xb


def predict_split(learner, Xb, tr_i, te_i, ytr, yte):
    if learner == "gbdt":
        return _gbdt(Xb, tr_i, te_i, ytr, yte)
    if learner == "knn":
        return run_knn_fold(Xb, tr_i, te_i, ytr, yte, 3)
    if learner == "ridge_profile":
        return run_ridge_fold(Xb, tr_i, te_i, ytr, yte)
    if learner == "mlp":
        return run_mlp_fold(Xb, tr_i, te_i, ytr, yte)
    raise ValueError(learner)


def eval_budget(bb, k, ymap, splits, X, Kc, idx_tok, learner, self_mode):
    rows, pred_all, y_all = [], [], []
    for s, (lab, te) in enumerate(splits):
        tr_i = np.array([idx_tok[t] for t in lab])
        te_i = np.array([idx_tok[t] for t in te])
        ytr = np.array([ymap[t] for t in lab], dtype=np.float64)
        yte = np.array([ymap[t] for t in te], dtype=np.float64)
        if len(te) < 2 or np.std(ytr) < 1e-12:
            continue
        Xb = descriptor(X, Kc, tr_i, self_mode)
        pred = predict_split(learner, Xb, tr_i, te_i, ytr, yte)
        if pred is None:
            continue
        pred = np.asarray(pred, dtype=np.float64)
        if np.isnan(pred).any():
            continue
        rows.append({"split": int(s),
                     "spearman": _spearman(pred, yte),
                     "pearson": _pearson(pred, yte),
                     "top1": _top1_gt(pred, yte)})
        pred_all.extend(pred.tolist())
        y_all.extend(yte.tolist())
    if not rows:
        return None

    fold_mean = {}
    for m in METRICS:
        v = np.array([r[m] for r in rows if not np.isnan(r[m])], dtype=np.float64)
        fold_mean[m] = {"mean": float(v.mean()) if len(v) else float("nan"),
                        "std": float(v.std()) if len(v) else float("nan"),
                        "n": int(len(v))}
    P = np.array(pred_all, dtype=np.float64)
    Y = np.array(y_all, dtype=np.float64)
    pooled = {"spearman": _spearman(P, Y), "pearson": _pearson(P, Y),
              "n_points": int(len(P))}
    return {"backbone": bb, "k": int(k), "fold_mean": fold_mean,
            "pooled": pooled, "n_splits_valid": len(rows)}


# ----------------------------------------------------------------------
def _fmt(mean, std=None, nd=3):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} +/- {std:.{nd}f}"


def md_table(k_list, backbones, values, title, nd=3, with_std=True):
    lines = [f"### {title}", "",
             "| k' | " + " | ".join(backbones) + " |",
             "|" + "---|" * (len(backbones) + 1)]
    for k in k_list:
        cells = []
        for bb in backbones:
            mean, std = values.get((bb, k), (None, None))
            cells.append(_fmt(mean, std if with_std else None, nd))
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _input_default(name, preferred_dir):
    """Locate a descriptor input.

    Order: the preferred (reference) directory, then the shipped copy under
    artifacts/, then results/rebuild/ where `bash run.sh cka` writes the CKA
    kernel.  Path-only convenience; on the original layout it is a no-op."""
    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in (os.path.join(preferred_dir, name),
                 os.path.join(proj, "artifacts", name),
                 os.path.join(proj, "results", "rebuild", name)):
        if os.path.isfile(cand):
            return cand
    return os.path.join(preferred_dir, name)


def main():
    ap = argparse.ArgumentParser(
        description="CKA-X (GBDT) labeling-budget sweep")
    ap.add_argument("--cm_csv", type=str, default=_input_default(
        "crossmodal_stats_final_qwen25_full.csv", RESULTS_DIR))
    ap.add_argument("--cka_pt", type=str, default=_input_default(
        "cka_diverse.pt", RESULTS_DIR))
    ap.add_argument("--kernel_key", type=str, default=None)
    ap.add_argument("--x_cols", nargs="*", default=MAIN_COLS,
                    help="cross-modal columns of the descriptor "
                         "(the descriptor's two cross-modal scalars)")
    ap.add_argument("--pool_file", type=str, default=None,
                    help="optional list of candidate encoders, one name per "
                         "line.  Default: derive the pool from the descriptor "
                         "coverage and the labels (the labeled pool of this "
                         "study).")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="label columns to evaluate. Default: the backbones "
                         "the label file defines")
    ap.add_argument("--k_sweep", type=str, default=DEFAULT_K_SWEEP)
    ap.add_argument("--n_splits", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_mode", choices=["nested", "independent"],
                    default="nested")
    ap.add_argument("--learner", choices=["gbdt", "knn", "ridge_profile", "mlp"],
                    default="gbdt")
    ap.add_argument("--self_mode", choices=["keep", "zero"], default="keep",
                    help="keep = descriptor as defined (training rows keep "
                         "CKA(m,m)=1); zero = diagnostic without the self entry")
    ap.add_argument("--out_json", type=str, default=os.path.join(
        DERIVED_DIR, "budget_sweep.json"))
    ap.add_argument("--out_csv", type=str, default=os.path.join(
        DERIVED_DIR, "budget_sweep.csv"))
    ap.add_argument("--out_md", type=str, default=os.path.join(
        DERIVED_DIR, "budget_sweep.md"))
    args = ap.parse_args()

    k_list = sorted({int(x) for x in args.k_sweep.split(",") if x.strip()})
    if not k_list:
        sys.exit("[ERROR] empty --k_sweep")
    gt = load_ground_truth()
    if not gt:
        sys.exit("[ERROR] ground truth empty")
    backbones = (list(args.targets) if args.targets
                 else label_backbones(gt, preferred=BACKBONE_ORDER))
    if not backbones:
        sys.exit("[ERROR] the label file defines no backbone column; pass "
                 "--targets explicitly")
    for p in (args.cm_csv, args.cka_pt):
        if not os.path.isfile(p):
            sys.exit(f"[ERROR] missing {p}")
    if args.pool_file and not os.path.isfile(args.pool_file):
        sys.exit(f"[ERROR] missing {args.pool_file}")

    cm, used_cols = load_cm_cols(args.cm_csv, args.x_cols)
    cka_toks, K = load_cka_kernel(args.cka_pt, args.kernel_key)
    if args.pool_file:
        with open(args.pool_file, "r", encoding="utf-8") as f:
            want = {ln.strip() for ln in f
                    if ln.strip() and not ln.startswith("#")}
    else:
        want = None          # derive: every encoder with all requested labels

    common = sorted(set(cm) & set(cka_toks) & set(gt))
    pool = [t for t in common
            if (want is None or t in want)
            and all(((gt[t].get("scores") or {}).get(bb)) is not None
                    for bb in backbones)]
    N = len(pool)
    pos = {t: i for i, t in enumerate(cka_toks)}
    Kc = K[np.ix_([pos[t] for t in common], [pos[t] for t in common])]
    idx_tok = {t: i for i, t in enumerate(common)}
    X = np.stack([cm[t] for t in common]).astype(np.float64)

    print(f"  learner        : {args.learner} "
          f"(gbdt = 50 trees/depth 2/lr 0.05/subsample 0.8, fixed)")
    print(f"  descriptor     : X cols {used_cols} + CKA profile over the "
          f"reference pool   [self_mode={args.self_mode}]")
    print(f"  C  : {os.path.basename(args.cka_pt)} "
          f"({len(cka_toks)} encoders, {tuple(K.shape)})")
    print(f"  X  : {os.path.basename(args.cm_csv)}")
    print(f"  pool: {N} encoders (cm {len(cm)} | cka {len(cka_toks)} | "
          f"gt {len(gt)} | pool_file "
        f"{len(want) if want is not None else 'derived'})")
    if N < 8:
        sys.exit("[ERROR] pool too small")
    for k in k_list:
        if k >= N - 1:
            sys.exit(f"[ERROR] k'={k} leaves <2 held-out encoders (N={N})")

    splits_by_k = make_splits(pool, k_list, args.n_splits, args.seed,
                              args.split_mode)
    print(f"  splits: {args.n_splits} per k', seed={args.seed}, "
          f"mode={args.split_mode}; shared across backbones")

    results = {}
    for bb in backbones:
        ymap = {t: float((gt[t].get("scores") or {})[bb]) for t in pool}
        results[bb] = {}
        print(f"\n  === backbone {bb} ===")
        for k in k_list:
            r = eval_budget(bb, k, ymap, splits_by_k[k], X, Kc, idx_tok,
                            args.learner, args.self_mode)
            if r is None:
                print(f"    k'={k:2d}: no valid split")
                results[bb][str(k)] = None
                continue
            results[bb][str(k)] = r
            fm, pl = r["fold_mean"], r["pooled"]
            print(f"    k'={k:2d}: valid={r['n_splits_valid']:3d}/"
                  f"{args.n_splits} | "
                  f"fold-mean rho={_fmt(fm['spearman']['mean'], fm['spearman']['std'])} "
                  f"pearson={_fmt(fm['pearson']['mean'], fm['pearson']['std'])} "
                  f"top1={_fmt(fm['top1']['mean'], fm['top1']['std'], 2)} | "
                  f"pooled rho={pl['spearman']:.3f} pearson={pl['pearson']:.3f} "
                  f"(n={pl['n_points']})")

    for p in (args.out_json, args.out_csv, args.out_md):
        d = os.path.dirname(os.path.abspath(p))
        if d:
            os.makedirs(d, exist_ok=True)

    def _rel(path):
        if not path:
            return path
        try:
            return os.path.relpath(
                path, os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))).replace(os.sep, "/")
        except ValueError:          # different drive (Windows)
            return path

    payload = {
        "protocol": {
            "name": "CKA-X (GBDT) labeling-budget sweep, 100 random labeled subsets per budget",
            "method": "descriptor [CKA profile to reference pool, cm_cka, "
                      "cm_r2] + gradient-boosted shallow trees",
            "learner": args.learner,
            "learner_hyperparams": {"n_estimators": 50, "learning_rate": 0.05,
                                    "max_depth": 2, "subsample": 0.8,
                                    "min_samples_leaf": 1, "random_state": 0},
            "self_mode": args.self_mode,
            "n_candidates_N": N,
            "k_sweep": k_list,
            "n_splits": args.n_splits,
            "seed": args.seed,
            "split_mode": args.split_mode,
            "shared_splits_across_backbones": True,
            "pipeline": "descriptor artifacts: pairwise CKA kernel + unified frozen "
                        "Qwen2.5-1.5B cross-modal",
            "cm_csv": _rel(args.cm_csv),
            "x_cols": used_cols,
            "cka_pt": _rel(args.cka_pt),
            "pool_file": args.pool_file or "derived (labels + descriptor coverage)",
            "calibers": {
                "fold_mean": "metric per split, then mean +/- std over splits",
                "pooled": "all held-out predictions of all splits "
                          "concatenated, one metric over n_splits*(N-k') points",
            },
        },
        "backbones": results,
    }
    with open(args.out_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["caliber", "backbone", "k", "metric", "mean", "std", "n"])
        for bb in backbones:
            for k in k_list:
                r = (results.get(bb) or {}).get(str(k))
                if r is None:
                    continue
                for m in METRICS:
                    c = r["fold_mean"][m]
                    wr.writerow(["fold_mean", bb, k, m, f"{c['mean']:.6f}",
                                 f"{c['std']:.6f}", c["n"]])
                for m in ("spearman", "pearson"):
                    wr.writerow(["pooled", bb, k, m,
                                 f"{r['pooled'][m]:.6f}", "",
                                 r["pooled"]["n_points"]])

    def collect(caliber, metric):
        vals = {}
        for bb in backbones:
            for k in k_list:
                r = (results.get(bb) or {}).get(str(k))
                if r is None:
                    continue
                if caliber == "fold_mean":
                    c = r["fold_mean"][metric]
                    vals[(bb, k)] = (c["mean"], c["std"])
                else:
                    vals[(bb, k)] = (r["pooled"][metric], None)
        return vals

    md = ["# CKA-X (GBDT) labeling-budget sweep", "",
          f"- method: descriptor [CKA profile to the reference pool, "
          f"{', '.join(used_cols)}] + GBDT "
          f"(50 trees, depth 2, lr 0.05, subsample 0.8)",
          f"- descriptor artifacts: `{os.path.basename(args.cka_pt)}` + "
          f"`{os.path.basename(args.cm_csv)}`",
          f"- N = {N} candidate encoders, k' in "
          f"{{{', '.join(str(k) for k in k_list)}}}",
          f"- {args.n_splits} random splits per k', seed {args.seed}, "
          f"split_mode `{args.split_mode}`, self_mode `{args.self_mode}`",
          "- same splits for every backbone; models fitted independently", ""]
    md += md_table(k_list, backbones, collect("fold_mean", "spearman"),
                   "Spearman - fold-mean caliber (mean +/- std over splits)")
    md += md_table(k_list, backbones, collect("pooled", "spearman"),
                   "Spearman - pooled caliber (all held-out points pooled)",
                   with_std=False)
    md += md_table(k_list, backbones, collect("fold_mean", "pearson"),
                   "Pearson - fold-mean caliber (mean +/- std over splits)")
    md += md_table(k_list, backbones, collect("pooled", "pearson"),
                   "Pearson - pooled caliber (all held-out points pooled)",
                   with_std=False)
    md += md_table(k_list, backbones, collect("fold_mean", "top1"),
                   "Top-1 - fold-mean caliber (GT score of the top-ranked "
                   "held-out encoder)", nd=2)
    with open(args.out_md, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")

    print("\n" + "=" * 78)
    for caliber in ("fold_mean", "pooled"):
        print(f"  Spearman - {caliber}")
        print("    k'".ljust(7) + "".join(f"{bb:>22s}" for bb in backbones))
        for k in k_list:
            line = f"    {k:<4d} "
            for bb in backbones:
                r = (results.get(bb) or {}).get(str(k))
                if r is None:
                    line += f"{'--':>22s}"
                elif caliber == "fold_mean":
                    c = r["fold_mean"]["spearman"]
                    line += f"{_fmt(c['mean'], c['std']):>22s}"
                else:
                    line += f"{r['pooled']['spearman']:>22.3f}"
            print(line)
        print()
    print("  Pearson - fold_mean")
    print("    k'".ljust(7) + "".join(f"{bb:>22s}" for bb in backbones))
    for k in k_list:
        line = f"    {k:<4d} "
        for bb in backbones:
            r = (results.get(bb) or {}).get(str(k))
            if r is None:
                line += f"{'--':>22s}"
            else:
                c = r["fold_mean"]["pearson"]
                line += f"{_fmt(c['mean'], c['std']):>22s}"
        print(line)
    print("\n  Top-1 (fold-mean, GT score of top-ranked held-out encoder)")
    print("    k'".ljust(7) + "".join(f"{bb:>22s}" for bb in backbones))
    for k in k_list:
        line = f"    {k:<4d} "
        for bb in backbones:
            r = (results.get(bb) or {}).get(str(k))
            if r is None:
                line += f"{'--':>22s}"
            else:
                c = r["fold_mean"]["top1"]
                line += f"{_fmt(c['mean'], c['std'], 2):>22s}"
        print(line)

    print("\n  Caliber comparison (Spearman averaged over k'):")
    for bb in backbones:
        fm, pl = [], []
        for k in k_list:
            r = (results.get(bb) or {}).get(str(k))
            if r is None:
                continue
            fm.append(r["fold_mean"]["spearman"]["mean"])
            pl.append(r["pooled"]["spearman"])
        if fm:
            print(f"    {bb:8s} fold_mean={np.mean(fm):.3f}  "
                  f"pooled={np.mean(pl):.3f}  "
                  f"delta={np.mean(pl) - np.mean(fm):+.3f}")
    print("=" * 78)
    for p in (args.out_json, args.out_csv, args.out_md):
        print(f"  Saved: {p}")


if __name__ == "__main__":
    main()
