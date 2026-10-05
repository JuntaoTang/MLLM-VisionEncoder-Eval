#!/usr/bin/env python3
"""AC Policy k'=8 with the paper-style Spearman / Pearson / Top-1 protocol.

s_i is a single predicted score (poly2(A,C) fit to Avg on k' labeled MLLMs).
rho_t / r_t correlate s with each benchmark t on the held-out set.
We report mean_t rho_t and rho vs Avg (same for Pearson), plus Top-1 = Avg of argmax s.
"""

from __future__ import annotations

from vision_encoder_eval.core.runtime import asset_path

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_encoder_eval.workers.law.common import RESULTS_DIR, atomic_json  # noqa: E402
from vision_encoder_eval.workers.law.fit_ac import merge_shards, minmax  # noqa: E402

from scipy.stats import pearsonr, spearmanr

GT_CANDIDATES = (
    Path(__file__).resolve().parent.parent / "ground_truth.json",
    Path(asset_path('mllm', 'results/ground_truth.json')),
)
GT_PATH = next(p for p in GT_CANDIDATES if p.is_file() and p.stat().st_size > 0)
K_PRIME = 8
N_REPEAT = 100
SEED = 0
LLMS = ("qwen3", "qwen25", "smollm2")
LLM_LABEL = {
    "qwen3": "Qwen3-1.7B",
    "qwen25": "Qwen2.5-1.5B-Instruct",
    "smollm2": "SmolLM2-1.7B-Instruct",
}


def corr(a, b, kind: str) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    fn = spearmanr if kind == "spearman" else pearsonr
    return float(fn(a, b).statistic)


def load_gt():
    gt = json.loads(GT_PATH.read_text())
    a_map = merge_shards("a_score")
    c_map = merge_shards("c_score")
    benches = list(gt["benchmarks"])
    by_llm = {llm: [] for llm in LLMS}
    dropped = {llm: {"no_a": [], "no_c": []} for llm in LLMS}
    for vid in gt["order"]:
        enc = gt["encoders"][vid]
        for llm in LLMS:
            rec = enc["llms"][llm]
            scores = rec["scores"]
            row = {
                "vision_id": vid,
                "average": float(rec["average"]),
                "a_score": None,
                "c_score": None,
            }
            for b in benches:
                row[b] = float(scores[b])
            fk = rec.get("finish_key")
            a = a_map.get(fk) or a_map.get(f"continuous/{llm}/{vid}_mlp2x") or {}
            c = c_map.get(vid) or {}
            if a.get("a_score") is None:
                dropped[llm]["no_a"].append(vid)
            else:
                row["a_score"] = float(a["a_score"])
            if c.get("c_score") is None:
                dropped[llm]["no_c"].append(vid)
            else:
                row["c_score"] = float(c["c_score"])
            by_llm[llm].append(row)
    return benches, by_llm, dropped, gt["n_encoders"]


def summarize(arr: list[float]) -> dict:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    return {
        "mean": float(np.mean(x)) if x.size else float("nan"),
        "std": float(np.std(x)) if x.size else float("nan"),
        "n_splits": int(x.size),
    }


def run_llm(rows: list[dict], benches: list[str]) -> dict:
    complete = [r for r in rows if r["a_score"] is not None and r["c_score"] is not None]
    n = len(complete)
    a = np.array([r["a_score"] for r in complete], dtype=float)
    c = np.array([r["c_score"] for r in complete], dtype=float)
    avg = np.array([r["average"] for r in complete], dtype=float)
    yb = {b: np.array([r[b] for r in complete], dtype=float) for b in benches}
    x = np.column_stack([minmax(a), minmax(c)])
    poly = PolynomialFeatures(degree=2, include_bias=True)
    rng = np.random.RandomState(SEED)

    rho_avg, r_avg, top1 = [], [], []
    rho_mean_t, r_mean_t = [], []
    per_b = {b: {"rho": [], "r": [], "top1": []} for b in benches}

    for _ in range(N_REPEAT):
        idx = rng.permutation(n)
        tr, te = idx[:K_PRIME], idx[K_PRIME:]
        model = LinearRegression().fit(poly.fit_transform(x[tr]), avg[tr])
        s = model.predict(poly.transform(x[te]))
        y_avg = avg[te]
        rho_avg.append(corr(s, y_avg, "spearman"))
        r_avg.append(corr(s, y_avg, "pearson"))
        pick = int(np.argmax(s))
        top1.append(float(y_avg[pick]))
        rhos, rs = [], []
        for b in benches:
            rho = corr(s, yb[b][te], "spearman")
            r = corr(s, yb[b][te], "pearson")
            per_b[b]["rho"].append(rho)
            per_b[b]["r"].append(r)
            per_b[b]["top1"].append(float(yb[b][te][pick]))
            rhos.append(rho)
            rs.append(r)
        rho_mean_t.append(float(np.nanmean(rhos)))
        r_mean_t.append(float(np.nanmean(rs)))

    return {
        "n_gt": len(rows),
        "n_ac": n,
        "n_test": n - K_PRIME,
        "oracle_avg": float(np.max(avg)),
        "pool_mean_avg": float(np.mean(avg)),
        "spearman_mean_T": summarize(rho_mean_t),
        "spearman_avg": summarize(rho_avg),
        "pearson_mean_T": summarize(r_mean_t),
        "pearson_avg": summarize(r_avg),
        "top1_avg": summarize(top1),
        "benchmarks": {
            b: {
                "spearman": summarize(per_b[b]["rho"]),
                "pearson": summarize(per_b[b]["r"]),
                "top1": summarize(per_b[b]["top1"]),
            }
            for b in benches
        },
    }


def main() -> None:
    benches, by_llm, dropped, n_gt = load_gt()
    out = {
        "k_prime": K_PRIME,
        "n_repeat": N_REPEAT,
        "seed": SEED,
        "n_gt": n_gt,
        "benchmarks": benches,
        "ground_truth": str(GT_PATH),
        "protocol": (
            "Fit poly2(A,C) to Avg on k'=8 labeled encoders; s_i = prediction "
            "on held-out set. Spearman/Pearson use the same s_i vs each y_{i,t} "
            "and vs Avg. Top-1 = Avg of argmax s on the held-out set. Mean over 100 splits."
        ),
        "dropped": dropped,
        "llms": {},
    }
    print(f"GT {GT_PATH}  n_gt={n_gt}  k'={K_PRIME}  splits={N_REPEAT}", flush=True)
    print(
        f"{'LLM':<28}{'n':>4}  {'ρ̄_T':>7}  {'ρ_Avg':>7}  {'r̄_T':>7}  {'r_Avg':>7}  {'Top-1':>7}  {'oracle':>7}",
        flush=True,
    )
    for llm in LLMS:
        rec = run_llm(by_llm[llm], benches)
        rec["label"] = LLM_LABEL[llm]
        rec["dropped_a"] = dropped[llm]["no_a"]
        rec["dropped_c"] = dropped[llm]["no_c"]
        out["llms"][llm] = rec
        print(
            f"{LLM_LABEL[llm]:<28}{rec['n_ac']:4d}  "
            f"{rec['spearman_mean_T']['mean']:7.3f}  {rec['spearman_avg']['mean']:7.3f}  "
            f"{rec['pearson_mean_T']['mean']:7.3f}  {rec['pearson_avg']['mean']:7.3f}  "
            f"{rec['top1_avg']['mean']:7.2f}  {rec['oracle_avg']:7.2f}",
            flush=True,
        )
        da, dc = dropped[llm]["no_a"], dropped[llm]["no_c"]
        print(f"  no A ({len(da)}): {da}", flush=True)
        print(f"  no C ({len(dc)}): {dc}", flush=True)

    path = RESULTS_DIR / "ac_k8_protocol.json"
    atomic_json(path, out)
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
