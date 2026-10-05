#!/usr/bin/env python3
"""AC-policy Spearman / Pearson / Top-1 on ground_truth.json vs k'."""

from __future__ import annotations

from vision_encoder_eval.core.runtime import asset_path

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_encoder_eval.workers.law.common import RESULTS_DIR, atomic_json  # noqa: E402
from vision_encoder_eval.workers.law.fit_ac import HEADLINE_LLMS, merge_shards, minmax, pearson, spearman  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GT_CANDIDATES = (
    ROOT / "ground_truth.json",
    Path(asset_path('mllm', 'results/ground_truth.json')),
    Path(asset_path('mllm', 'results/ground_truth.json')),
)
GT_PATH = next((p for p in GT_CANDIDATES if p.is_file() and p.stat().st_size > 0), GT_CANDIDATES[0])
N_REPEAT = 100
SEED = 0
K_GRID = (4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32)


def load_gt_frame() -> pd.DataFrame:
    gt = json.loads(GT_PATH.read_text())
    a_map = merge_shards("a_score")
    c_map = merge_shards("c_score")
    rows = []
    for vid in gt["order"]:
        enc = gt["encoders"][vid]
        for llm in gt["llms"]:
            rec = (enc.get("llms") or {}).get(llm) or {}
            avg = rec.get("average")
            finish_key = rec.get("finish_key")
            a_rec = a_map.get(finish_key) or a_map.get(f"continuous/{llm}/{vid}_mlp2x") or {}
            c_rec = c_map.get(vid) or {}
            rows.append(
                {
                    "vision_id": vid,
                    "display_name": enc.get("display_name"),
                    "type": enc.get("type"),
                    "llm_id": llm,
                    "average": avg,
                    "a_score": a_rec.get("a_score"),
                    "c_score": c_rec.get("c_score"),
                    "finish_key": finish_key,
                    "gt_mean": enc.get("mean"),
                }
            )
    return pd.DataFrame(rows)


def eval_policy(sub: pd.DataFrame, n_train: int, n_repeat: int = N_REPEAT, seed: int = SEED) -> dict:
    sub = sub.reset_index(drop=True)
    x = np.column_stack([minmax(sub["a_score"].to_numpy()), minmax(sub["c_score"].to_numpy())])
    xa = minmax(sub["a_score"].to_numpy()).reshape(-1, 1)
    xc = minmax(sub["c_score"].to_numpy()).reshape(-1, 1)
    y = sub["average"].to_numpy(dtype=float)
    enc = sub["vision_id"].astype(str).to_numpy()
    n = len(sub)
    rng = np.random.RandomState(seed)
    poly = PolynomialFeatures(degree=2, include_bias=True)
    acc = {
        m: {"rho": [], "r": [], "top1": [], "hit": [], "r3": [], "ids": []}
        for m in ("AC", "A", "C", "random")
    }
    xs = {"AC": x, "A": xa, "C": xc}
    for _ in range(n_repeat):
        idx = rng.permutation(n)
        tr, te = idx[:n_train], idx[n_train:]
        yte = y[te]
        true_order = np.argsort(-yte)
        for method, xm in xs.items():
            model = LinearRegression().fit(poly.fit_transform(xm[tr]), y[tr])
            pred = model.predict(poly.transform(xm[te]))
            pick = int(np.argmax(pred))
            acc[method]["rho"].append(spearman(pred, yte))
            acc[method]["r"].append(pearson(pred, yte))
            acc[method]["top1"].append(float(yte[pick]))
            acc[method]["hit"].append(int(pick == true_order[0]))
            acc[method]["r3"].append(int(pick in set(true_order[:3])))
            acc[method]["ids"].append(str(enc[te[pick]]))
        pred_rnd = rng.rand(len(te))
        pick = int(np.argmax(pred_rnd))
        acc["random"]["rho"].append(spearman(pred_rnd, yte))
        acc["random"]["r"].append(pearson(pred_rnd, yte))
        acc["random"]["top1"].append(float(yte[pick]))
        acc["random"]["hit"].append(int(pick == true_order[0]))
        acc["random"]["r3"].append(int(pick in set(true_order[:3])))
        acc["random"]["ids"].append(str(enc[te[pick]]))

    def pack(a: dict) -> dict:
        return {
            "spearman": float(np.mean(a["rho"])),
            "spearman_std": float(np.std(a["rho"])),
            "pearson": float(np.mean(a["r"])),
            "pearson_std": float(np.std(a["r"])),
            "top1": float(np.mean(a["top1"])),
            "top1_std": float(np.std(a["top1"])),
            "top1_hit": float(np.mean(a["hit"])),
            "recall_at_3": float(np.mean(a["r3"])),
            "most_picked": pd.Series(a["ids"]).value_counts().head(3).to_dict(),
        }

    return {m: pack(acc[m]) for m in acc}


def loo_fit(sub: pd.DataFrame) -> dict:
    x = np.column_stack([minmax(sub["a_score"].to_numpy()), minmax(sub["c_score"].to_numpy())])
    y = sub["average"].to_numpy(dtype=float)
    n = len(y)
    poly = PolynomialFeatures(degree=2, include_bias=True)
    pred_in = LinearRegression().fit(poly.fit_transform(x), y).predict(poly.transform(x))
    pred_loo = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model = LinearRegression().fit(poly.fit_transform(x[mask]), y[mask])
        pred_loo[i] = model.predict(poly.transform(x[i : i + 1]))[0]
    return {
        "in_spearman": spearman(pred_in, y),
        "in_pearson": pearson(pred_in, y),
        "loo_spearman": spearman(pred_loo, y),
        "loo_pearson": pearson(pred_loo, y),
        "oracle_best": float(np.max(y)),
        "pool_mean": float(np.mean(y)),
        "oracle_encoder": str(sub.iloc[int(np.argmax(y))]["vision_id"]),
        "n": int(n),
    }


def main() -> None:
    print(f"ground_truth: {GT_PATH}", flush=True)
    df = load_gt_frame()
    coverage = []
    for llm in HEADLINE_LLMS:
        sub = df[df["llm_id"] == llm]
        coverage.append(
            {
                "llm": llm,
                "n_gt": int(len(sub)),
                "n_with_a": int(sub["a_score"].notna().sum()),
                "n_with_c": int(sub["c_score"].notna().sum()),
                "n_complete": int(sub.dropna(subset=["average", "a_score", "c_score"]).shape[0]),
                "missing_a": sub[sub["a_score"].isna()]["vision_id"].tolist(),
                "missing_c": sub[sub["c_score"].isna()]["vision_id"].tolist(),
            }
        )
    out = {"n_repeat": N_REPEAT, "seed": SEED, "k_grid": list(K_GRID), "coverage": coverage, "llms": {}}
    print("coverage", json.dumps(coverage, indent=2), flush=True)
    for llm in HEADLINE_LLMS:
        sub = df[df["llm_id"] == llm].dropna(subset=["average", "a_score", "c_score"]).copy()
        stats = loo_fit(sub)
        policy = []
        n = len(sub)
        ks = [k for k in K_GRID if 4 <= k <= n // 2]
        print(f"\n=== {llm} n={n} oracle={stats['oracle_best']:.2f} ===", flush=True)
        print(f"{'k':>4}  {'rho':>7}  {'r':>7}  {'Top1':>7}  {'hit':>6}  {'R@3':>6}  {'A rho':>7}  {'rnd T1':>7}", flush=True)
        for k in ks:
            rec = eval_policy(sub, k)
            rec["k_prime"] = k
            rec["n_test"] = n - k
            policy.append(rec)
            ac, a, rnd = rec["AC"], rec["A"], rec["random"]
            print(
                f"{k:4d}  {ac['spearman']:7.3f}  {ac['pearson']:7.3f}  {ac['top1']:7.2f}  "
                f"{ac['top1_hit']:6.2f}  {ac['recall_at_3']:6.2f}  {a['spearman']:7.3f}  {rnd['top1']:7.2f}",
                flush=True,
            )
        out["llms"][llm] = {"fit": stats, "policy": policy}
    atomic_json(RESULTS_DIR / "ac_gt70_k_sweep.json", out)
    print(f"\nwrote {RESULTS_DIR / 'ac_gt70_k_sweep.json'}", flush=True)


if __name__ == "__main__":
    main()
