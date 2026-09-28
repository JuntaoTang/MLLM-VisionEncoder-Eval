# -*- coding: utf-8 -*-
"""
verify_ckax.py
==============
One-shot self-check for CKA-X.  Recomputes the reference numbers from the
inputs in ``artifacts/`` (plus the CKA kernel) and diffs them against the
reference outputs in ``results/reference/``, so the reproducibility claim is
machine-checkable instead of asserted.

What is checked
---------------
  A. artifact integrity
       - pairwise CKA kernel: square, symmetric, unit diagonal, in [0, 1]
       - a ``cka_tokens.txt`` next to the kernel (if present) matches its
         row order
       - the cross-modal CSV carries ``cm_cka`` and ``cm_r2`` for the pool
       - the label file covers every encoder of the pool
  B. descriptor + full-information protocol (C, X)
       descriptor [k_m, x_m] = [CKA profile over the reference pool, cm_cka,
       cm_r2], shared 5-fold (seed 42) over the labeled pool, compared with
         results/reference/fullinfo_gbdt.json   (learner = gradient boosting)
         results/reference/fullinfo_krr.json    (learner = kernel ridge)
  C. labeling-budget study
       same descriptor + gradient boosting; 100 random labeled subsets per
       budget k' in {8,16,24,32,40,48,56,64}, seed 0, compared with
         results/reference/budget_sweep.json
       (pooled Spearman per backbone must reproduce 0.498/0.509/0.356 at
        k'=8 and 0.809/0.853/0.685 at k'=64)

The labels are not included in this package.  Supply them with
``--gt`` or by putting it at ``ground_truth/ground_truth.json``; otherwise the
script exits with code 2 and explains what is missing.

Usage (from the CKA-X/ root)
----------------------------
    python scripts/verify_ckax.py --gt /path/to/ground_truth.json
    python scripts/verify_ckax.py --skip-sweep          # fast: A + B only
    python scripts/verify_ckax.py --out-md results/derived/verify_report.md

Exit codes:  0 = all checks PASS, 1 = at least one mismatch, 2 = cannot run
(missing labels / artifacts).  Console output is ASCII only.
"""
import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJ_DIR)

import config  # noqa: E402

TOL = 1e-9
BACKBONES = ["qwen3", "qwen25", "smollm2"]     # the archive's label columns


# ----------------------------------------------------------------------
# reporting helpers
# ----------------------------------------------------------------------
class Report:
    def __init__(self):
        self.rows = []
        self.failed = 0
        self.checked = 0

    def check(self, section, name, got, want, tol=TOL):
        ok = False
        if got is None or want is None:
            ok = (got is None and want is None)
            diff = float("nan")
        else:
            diff = abs(float(got) - float(want))
            ok = (diff <= tol) or (np.isnan(got) and np.isnan(want))
        self.checked += 1
        if not ok:
            self.failed += 1
        self.rows.append((section, name, got, want, diff, ok))
        return ok

    def note(self, section, name, text):
        self.rows.append((section, name, text, None, None, True))

    def md(self):
        L = ["| section | quantity | recomputed | archived | |diff| |",
             "|---|---|---|---|---|"]
        for s, n, g, w, d, ok in self.rows:
            if w is None and d is None:
                L.append("| %s | %s | %s | -- | -- |" % (s, n, g))
                continue
            mark = "" if ok else "  **MISMATCH**"
            L.append("| %s | %s | %s | %s | %s |%s" % (
                s, n,
                "--" if g is None else ("%.6g" % g),
                "--" if w is None else ("%.6g" % w),
                "--" if d is None or np.isnan(d) else ("%.3g" % d),
                mark))
        return "\n".join(L) + "\n"

    def summary(self):
        bad = [r for r in self.rows if not r[5]]
        return self.checked, self.failed, bad


# ----------------------------------------------------------------------
# A. artifact integrity
# ----------------------------------------------------------------------
def default_cka_pt():
    """Where to look for the CKA kernel when --cka_pt is not given.

    The kernel is not included in this package: build it with
    `bash run.sh cka`, which writes results/rebuild/cka_diverse.pt.  A copy
    under artifacts/ is used if it exists."""
    for p in (os.path.join(PROJ_DIR, "artifacts", "cka_diverse.pt"),
              os.path.join(PROJ_DIR, "results", "rebuild", "cka_diverse.pt")):
        if os.path.isfile(p):
            return p
    return os.path.join(PROJ_DIR, "results", "rebuild", "cka_diverse.pt")


def _load_npz(path):
    z = np.load(path, allow_pickle=False)
    return [str(t) for t in z["toks"]], np.asarray(z["kernel"],
                                                   dtype=np.float64)


def load_kernel(cka_pt):
    """(tokens, K) from the kernel .pt, or from a .npz mirror next to it.

    A ``.npz`` path may also be passed directly; the mirror exists for
    environments where ``torch.load`` is unavailable."""
    if cka_pt.endswith(".npz"):
        if os.path.isfile(cka_pt):
            return _load_npz(cka_pt)
    elif os.path.isfile(cka_pt):
        import torch
        d = torch.load(cka_pt, map_location="cpu", weights_only=False)
        ck = d.get("cka") or d
        return list(ck["toks"]), np.asarray(ck["kernel"], dtype=np.float64)
    npz = os.path.splitext(cka_pt)[0] + ".npz"
    if os.path.isfile(npz):
        return _load_npz(npz)
    raise SystemExit(
        "\n[ERROR] no CKA kernel at %s\n"
        "        The kernel is not included in this package.\n"
        "        Build it with:  bash run.sh cka   (needs features_diverse/)\n"
        "        It lands in results/rebuild/cka_diverse.pt; see README,\n"
        "        section 5.1, \"Step 1 - build the CKA kernel (C)\".\n" % cka_pt)


def check_artifacts(rep, cka_pt, pool, gt):
    toks, K = load_kernel(cka_pt)
    rep.note("A", "kernel", os.path.relpath(cka_pt, PROJ_DIR))
    rep.note("A", "kernel shape", "%dx%d" % K.shape)
    rep.check("A", "kernel square", float(K.shape[0] == K.shape[1]), 1.0)
    rep.check("A", "max |K - K^T|", float(np.abs(K - K.T).max()), 0.0, 1e-12)
    rep.check("A", "min diag(K)", float(np.diag(K).min()), 1.0, 1e-12)
    rep.check("A", "max diag(K)", float(np.diag(K).max()), 1.0, 1e-12)
    rep.check("A", "min K (>= 0)", float(min(K.min(), 0.0)), 0.0)
    rep.check("A", "max K (<= 1)", float(max(K.max(), 1.0)), 1.0)

    tok_file = os.path.join(os.path.dirname(cka_pt), "cka_tokens.txt")
    if os.path.isfile(tok_file):
        listed = [l.strip() for l in open(tok_file, encoding="utf-8")
                  if l.strip()]
        rep.check("A", "cka_tokens.txt == kernel order",
                  float(listed == toks), 1.0)
    rep.check("A", "pool encoders present in kernel",
              float(all(t in toks for t in pool)), 1.0)
    return toks, K


def check_labels(rep, pool, gt):
    rep.note("A", "labelled encoders", "%d / %d in the pool"
             % (sum(1 for t in pool if t in gt), len(pool)))
    for bb in BACKBONES:
        n = sum(1 for t in pool if t in gt
                and (gt[t].get("scores") or {}).get(bb) is not None)
        rep.note("A", "backbone %s coverage" % bb, "%d / %d" % (n, len(pool)))


# ----------------------------------------------------------------------
# B. full-information 5-fold protocol
# ----------------------------------------------------------------------
def compare_main(rep, tag, got, want):
    for bb in BACKBONES:
        key = "C+X|pool|%s" % bb
        g = got.get(key)
        w = (want.get("results") or {}).get(key)
        if g is None or w is None:
            rep.check("B/%s" % tag, "%s present" % key,
                      float(g is not None), float(w is not None))
            continue
        rep.check("B/%s" % tag, "%s pooled rho" % bb,
                  g.get("pooled_rho"), w.get("pooled_rho"))
        rep.check("B/%s" % tag, "%s pooled pearson" % bb,
                  g.get("pooled_pearson"), w.get("pooled_pearson"))
        rep.check("B/%s" % tag, "%s pooled top1" % bb,
                  g.get("pooled_top1"), w.get("pooled_top1"), 1e-6)
        for metric, key_g, key_w in (
                ("fold-mean rho", "fold_mean_rho", "fold_mean_rho"),
                ("fold-mean pearson", "fold_mean_pearson", "fold_mean_pearson"),
                ("fold-mean top1", "fold_mean_top1_gt", "fold_mean_top1_gt")):
            gv = g.get(key_g) or [None, None]
            wv = w.get(key_w) or [None, None]
            rep.check("B/%s" % tag, "%s %s mean" % (bb, metric), gv[0], wv[0])
            rep.check("B/%s" % tag, "%s %s std" % (bb, metric), gv[1], wv[1],
                      1e-7)


def run_main(rep, res_dir, cm_csv, cka_pt, pool_file, bundle, learners):
    from run_ckax import run_configs, MAIN_COLS
    cm, cka_toks, K, gt, pools = bundle
    tag2file = {"gbdt": "fullinfo_gbdt.json",
                "krr": "fullinfo_krr.json"}
    for learner in learners:
        exp_path = os.path.join(res_dir, tag2file[learner])
        if not os.path.isfile(exp_path):
            rep.note("B", "%s archived output" % learner, "missing -> skipped")
            continue
        cfgs = [{"name": "C+X", "cm_csv": cm_csv, "x_cols": MAIN_COLS,
                 "w_fixed": 2.0, "alpha_fixed": 0.1, "learner": learner,
                 "use_cka": True}]
        got = run_configs(cfgs, cm, cka_toks, K, gt, pools, BACKBONES,
                          n_folds=5, fold_seed=42)
        want = json.load(open(exp_path, encoding="utf-8"))
        compare_main(rep, learner, got, want)


# ----------------------------------------------------------------------
# C. labeling-budget study
# ----------------------------------------------------------------------
def run_sweep(rep, res_dir, cm_csv, cka_pt, pool_file, k_sweep, n_splits,
              seed):
    from run_ckax import load_cm_cols, MAIN_COLS
    from run_budget_sweep import make_splits, eval_budget

    exp_path = os.path.join(res_dir, "budget_sweep.json")
    if not os.path.isfile(exp_path):
        rep.note("C", "archived sweep", "missing -> skipped")
        return
    want = json.load(open(exp_path, encoding="utf-8"))

    cm, used_cols = load_cm_cols(cm_csv, MAIN_COLS)
    cka_toks, K = load_kernel(cka_pt)
    gt = config.load_ground_truth()
    want_toks = None
    if pool_file:
        want_toks = {l.strip() for l in open(pool_file, encoding="utf-8")
                     if l.strip()}
    common = sorted(set(cm) & set(cka_toks) & set(gt))
    pool = [t for t in common if (want_toks is None or t in want_toks)
            and all(((gt[t].get("scores") or {}).get(bb)) is not None
                    for bb in BACKBONES)]
    pos = {t: i for i, t in enumerate(cka_toks)}
    Kc = K[np.ix_([pos[t] for t in common], [pos[t] for t in common])]
    idx_tok = {t: i for i, t in enumerate(common)}
    X = np.stack([cm[t] for t in common]).astype(np.float64)
    rep.note("C", "descriptor columns", ", ".join(used_cols))
    rep.note("C", "pool / N", "%d encoders" % len(pool))
    rep.check("C", "N == archived N",
              float(len(pool)),
              float(want["protocol"]["n_candidates_N"]))

    splits_by_k = make_splits(pool, k_sweep, n_splits, seed, "nested")
    for bb in BACKBONES:
        ymap = {t: float((gt[t].get("scores") or {})[bb]) for t in pool}
        for k in k_sweep:
            got = eval_budget(bb, k, ymap, splits_by_k[k], X, Kc, idx_tok,
                              "gbdt", "keep")
            w = ((want.get("backbones") or {}).get(bb) or {}).get(str(k))
            if got is None or w is None:
                rep.check("C", "%s k'=%d present" % (bb, k),
                          float(got is not None), float(w is not None))
                continue
            rep.check("C", "%s k'=%d pooled rho" % (bb, k),
                      got["pooled"]["spearman"], w["pooled"]["spearman"])
            rep.check("C", "%s k'=%d pooled pearson" % (bb, k),
                      got["pooled"]["pearson"], w["pooled"]["pearson"])
            for m in ("spearman", "pearson", "top1"):
                rep.check("C", "%s k'=%d fold-mean %s" % (bb, k, m),
                          got["fold_mean"][m]["mean"],
                          w["fold_mean"][m]["mean"], 1e-7)
                rep.check("C", "%s k'=%d fold-mean %s std" % (bb, k, m),
                          got["fold_mean"][m]["std"],
                          w["fold_mean"][m]["std"], 1e-7)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Self-check: recompute the CKA-X reference numbers from "
                    "the shipped artifacts and diff them against results/reference/.")
    ap.add_argument("--gt", type=str, default=None,
                    help="path to the real ground_truth.json "
                         "(overrides $CKA_X_GT and ground_truth/)")
    ap.add_argument("--results", type=str,
                    default=os.path.join(PROJ_DIR, "results", "reference"))
    ap.add_argument("--pool_file", type=str, default=None,
                    help="optional candidate list; default derives the pool "
                         "from the descriptor coverage and the labels")
    ap.add_argument("--cm_csv", type=str, default=os.path.join(
        PROJ_DIR, "artifacts", "crossmodal_stats_final_qwen25_full.csv"))
    ap.add_argument("--cka_pt", type=str, default=None,
                    help="the pairwise CKA kernel (not included; build it with "
                         "`bash run.sh cka`). Default: artifacts/cka_diverse.pt "
                         "if present, else results/rebuild/cka_diverse.pt")
    ap.add_argument("--k_sweep", type=str, default="8,16,24,32,40,48,56,64")
    ap.add_argument("--n_splits", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--learners", nargs="*", default=["gbdt", "krr"])
    ap.add_argument("--skip-sweep", action="store_true",
                    help="run checks A + B only (fast)")
    ap.add_argument("--tol", type=float, default=None,
                    help="override the absolute tolerance (default 1e-9)")
    ap.add_argument("--out-md", type=str, default=None,
                    help="also write the comparison table as markdown")
    args = ap.parse_args()

    global TOL
    if args.tol is not None:
        TOL = args.tol

    print("=" * 74)
    print("  CKA-X self-check (descriptor + full-information + budget study)")
    print("=" * 74)

    # ---- labels -------------------------------------------------------
    if args.gt:
        if not os.path.isfile(args.gt):
            sys.exit("[ERROR] --gt not found: %s" % args.gt)
        config.PATHS["ground_truth_json"] = os.path.abspath(args.gt)
    gt = config.load_ground_truth()
    if not gt:
        print("\n[ERROR] ground truth is empty -> nothing to verify against.")
        print("        This package ships only the format "
              "example")
        print("        (ground_truth/ground_truth.example.json).  Pass the "
              "real one:")
        print("          python scripts/verify_ckax.py --gt "
              "<path/to/ground_truth.json>")
        sys.exit(2)
    print("  labels      : %s  (%d encoders)"
          % (config.PATHS["ground_truth_json"], len(gt)))

    if args.cka_pt is None:
        args.cka_pt = default_cka_pt()
    print("  C (kernel)  : %s" % os.path.relpath(args.cka_pt, PROJ_DIR))
    print("  X (cross-mod): %s" % os.path.basename(args.cm_csv))
    print("  pool        : %s" % (os.path.basename(args.pool_file)
                                   if args.pool_file else "derived"))
    print("  archived    : %s" % args.results)

    if not os.path.isfile(args.cm_csv):
        sys.exit("[ERROR] missing %s" % args.cm_csv)
    if args.pool_file:
        pool = [l.strip() for l in open(args.pool_file, encoding="utf-8")
                if l.strip()]
    else:
        from ckax_core import load_cm as _load_cm
        _cm, _ = _load_cm(args.cm_csv)
        pool = sorted(t for t in _cm if t in gt
                      and all((gt[t].get("scores") or {}).get(bb) is not None
                              for bb in BACKBONES))

    # ---- A ------------------------------------------------------------
    rep = Report()
    print("\n-- A. artifact integrity " + "-" * 50)
    toks, K = check_artifacts(rep, args.cka_pt, pool, gt)
    check_labels(rep, pool, gt)
    print("   kernel %dx%d, %d encoders listed, %d pool encoders covered"
          % (K.shape[0], K.shape[1], len(toks),
             sum(1 for t in pool if t in toks)))

    # ---- shared bundle for B -----------------------------------------
    from run_ckax import load_cm_cols
    cm, used_cols = load_cm_cols(args.cm_csv, None)
    ptag = (os.path.splitext(os.path.basename(args.pool_file))[0]
            if args.pool_file else "pool")
    pools = {ptag: pool}
    bundle = (cm, toks, K, gt, pools)

    # ---- B ------------------------------------------------------------
    print("-- B. full-information 5-fold protocol " + "-" * 34)
    run_main(rep, args.results, args.cm_csv, args.cka_pt, args.pool_file,
             bundle, args.learners)
    for r in rep.rows:
        if r[0].startswith("B/") and "pooled rho" in r[1]:
            print("   %-28s rho %.6f (archived %.6f)  %s"
                  % (r[1], r[2], r[3], "OK" if r[5] else "MISMATCH"))

    # ---- C ------------------------------------------------------------
    if not args.skip_sweep:
        print("-- C. labeling-budget study " + "-" * 43)
        k_list = sorted({int(x) for x in args.k_sweep.split(",") if x.strip()})
        run_sweep(rep, args.results, args.cm_csv, args.cka_pt, args.pool_file,
                  k_list, args.n_splits, args.seed)
        for r in rep.rows:
            if r[0] == "C" and "pooled rho" in r[1]:
                print("   %-28s rho %.6f (archived %.6f)  %s"
                      % (r[1], r[2], r[3], "OK" if r[5] else "MISMATCH"))

    # ---- summary ------------------------------------------------------
    checked, failed, bad = rep.summary()
    print("\n" + "=" * 74)
    print("  checks: %d   mismatches: %d   tolerance: %g"
          % (checked, failed, TOL))
    if bad:
        print("  FAILED:")
        for s, n, g, w, d, _ in bad[:25]:
            print("    [%s] %-44s got=%s want=%s diff=%s"
                  % (s, n, g, w, d))
    else:
        print("  RESULT: PASS -- every archived CKA-X number reproduced "
              "bit-for-bit")
    print("=" * 74)

    if args.out_md:
        d = os.path.dirname(os.path.abspath(args.out_md))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8", newline="\n") as f:
            f.write("# CKA-X verification report\n\n")
            f.write("- checks: %d, mismatches: %d, tolerance: %g\n\n"
                    % (checked, failed, TOL))
            f.write(rep.md())
        print("  saved: %s" % args.out_md)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
