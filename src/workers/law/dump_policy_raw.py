#!/usr/bin/env python3
"""Dump 100 AC-policy splits at k'=8 and compute Pearson r + Top-1 true score."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_encoder_eval.workers.law.common import RESULTS_DIR, atomic_json  # noqa: E402
from vision_encoder_eval.workers.law.fit_ac import (  # noqa: E402
    BENCH_KEYS,
    HEADLINE_LLMS,
    K_PRIME_HEADLINE,
    PAPER_VISION_IDS,
    minmax,
    pearson,
    spearman,
)

try:
    from scipy.stats import pearsonr, spearmanr

    def _pearson(a, b):
        return float(pearsonr(a, b).statistic)

    def _spearman(a, b):
        return float(spearmanr(a, b).statistic)
except Exception:  # pragma: no cover
    _pearson, _spearman = pearson, spearman

K_PRIME = K_PRIME_HEADLINE
N_REPEAT = 100
SEED = 0
TARGET = "average"


def load_joined() -> pd.DataFrame:
    path = RESULTS_DIR / "ac_joined.csv"
    df = pd.read_csv(path)
    df = df[df["vision_id"].isin(PAPER_VISION_IDS)].copy()
    df = df.dropna(subset=["a_score", "c_score", TARGET, "llm_id"])
    return df.reset_index(drop=True)


def run_llm(df: pd.DataFrame, llm: str) -> tuple[pd.DataFrame, dict]:
    sub = df[df["llm_id"] == llm].copy().reset_index(drop=True)
    sub["norm_a"] = minmax(sub["a_score"].to_numpy())
    sub["norm_c"] = minmax(sub["c_score"].to_numpy())
    x = sub[["norm_a", "norm_c"]].to_numpy()
    y = sub[TARGET].to_numpy(dtype=float)
    enc = sub["vision_id"].astype(str).to_numpy()
    n = len(sub)
    # Replay the exact RNG of policy_sim_methods (permutation then one
    # rng.rand per benchmark for the random baseline). Otherwise k' splits
    # — and thus Spearman — do not match ac_paper_brief.txt.
    ys = {}
    for b in BENCH_KEYS:
        if b not in sub.columns:
            continue
        col = sub[b].to_numpy(dtype=float)
        if np.isfinite(col).sum() < K_PRIME + 2:
            continue
        ys[b] = col
    rng = np.random.RandomState(SEED)
    poly = PolynomialFeatures(degree=2, include_bias=True)
    rows = []
    rhos, rs, top1s, top1_ids = [], [], [], []
    for split_id in range(N_REPEAT):
        idx = rng.permutation(n)
        tr, te = idx[:K_PRIME], idx[K_PRIME:]
        # Same extra draws as the random baseline in policy_sim_methods.
        for yb in ys.values():
            mte = np.isfinite(yb[te])
            if mte.sum() < 3:
                continue
            rng.rand(int(mte.sum()))

        model = LinearRegression().fit(poly.fit_transform(x[tr]), y[tr])
        pred_tr = model.predict(poly.transform(x[tr]))
        pred_te = model.predict(poly.transform(x[te]))
        for i, p in zip(tr, pred_tr):
            rows.append(
                {
                    "split_id": split_id,
                    "llm": llm,
                    "encoder_id": enc[i],
                    "is_labeled": 1,
                    "y_true": float(y[i]),
                    "y_pred": float(p),
                }
            )
        for i, p in zip(te, pred_te):
            rows.append(
                {
                    "split_id": split_id,
                    "llm": llm,
                    "encoder_id": enc[i],
                    "is_labeled": 0,
                    "y_true": float(y[i]),
                    "y_pred": float(p),
                }
            )
        yte, pte = y[te], pred_te
        rhos.append(spearman(pte, yte))
        rs.append(_pearson(pte, yte))
        pick = int(np.argmax(pte))
        top1s.append(float(yte[pick]))
        top1_ids.append(str(enc[te[pick]]))
    raw = pd.DataFrame(rows)
    summary = {
        "llm": llm,
        "n": int(n),
        "k_prime": K_PRIME,
        "n_repeat": N_REPEAT,
        "seed": SEED,
        "target": TARGET,
        "spearman": float(np.mean(rhos)),
        "spearman_std": float(np.std(rhos)),
        "pearson": float(np.mean(rs)),
        "pearson_std": float(np.std(rs)),
        "top1": float(np.mean(top1s)),
        "top1_std": float(np.std(top1s)),
        "oracle_best": float(np.max(y)),
        "pool_mean": float(np.mean(y)),
        "most_picked": pd.Series(top1_ids).value_counts().head(5).to_dict(),
    }
    return raw, summary


def main():
    df = load_joined()
    parts, summaries = [], []
    for llm in HEADLINE_LLMS:
        raw, summary = run_llm(df, llm)
        parts.append(raw)
        summaries.append(summary)
        print(
            f"{llm}: rho={summary['spearman']:.3f}  r={summary['pearson']:.3f}  "
            f"Top1={summary['top1']:.2f}  oracle={summary['oracle_best']:.2f}",
            flush=True,
        )
    raw = pd.concat(parts, ignore_index=True)
    raw_path = RESULTS_DIR / "ac_policy_k8_raw.csv"
    raw.to_csv(raw_path, index=False)
    atomic_json(RESULTS_DIR / "ac_policy_k8_metrics.json", summaries)
    lines = [
        "AC Policy k'=8  (100 uniform random splits, seed=0)",
        f"raw: {raw_path}",
        "",
        f"{'LLM':<12}{'n':>4}  {'rho':>7}  {'r':>7}  {'Top1':>7}  {'oracle':>7}  {'pool mean':>9}",
    ]
    for s in summaries:
        lines.append(
            f"{s['llm']:<12}{s['n']:4d}  {s['spearman']:7.3f}  {s['pearson']:7.3f}  "
            f"{s['top1']:7.2f}  {s['oracle_best']:7.2f}  {s['pool_mean']:9.2f}"
        )
    lines += [
        "",
        "Top-1 = mean over splits of y_true of the held-out encoder with max y_pred.",
        "rho/r = mean over splits of Spearman/Pearson on the 34 held-out encoders.",
        "",
        "LaTeX:",
        "AC Policy ($k'=8)",
    ]
    by = {s["llm"]: s for s in summaries}
    q3, q25, sm = by["qwen3"], by["qwen25"], by["smollm2"]
    lines.append(
        f"& {q3['spearman']:.3f} & {q3['pearson']:.3f} & {q3['top1']:.2f} "
        f"& {q25['spearman']:.3f} & {q25['pearson']:.3f} & {q25['top1']:.2f} "
        f"& {sm['spearman']:.3f} & {sm['pearson']:.3f} & {sm['top1']:.2f} "
        f"& -- \\\\"
    )
    text = "\n".join(lines)
    (RESULTS_DIR / "ac_policy_k8_metrics.txt").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
