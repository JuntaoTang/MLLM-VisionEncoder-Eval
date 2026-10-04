#!/usr/bin/env python3
"""Reproduce the paper's AC law: P ≈ poly2(A, C), then rank remaining tokenizers.

Isolated A / C dumps are intermediate quantities. The paper result is:
  1. a degree-2 polynomial in (A, C) that predicts MLLM downstream scores
  2. the AC policy: finetune k' models, fit the surface, rank the rest
  3. AC jointly beats A-only, C-only, and random (paper Fig. 4 analogue)

Fits are always within one LLM. Mixing Qwen2.5 / Qwen3 / SmolLM2 is invalid
because A-score NLL and benchmark averages do not share a scale across LLMs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_encoder_eval.workers.law.common import RESULTS_DIR, atomic_json, inventory_models  # noqa: E402

K_PRIME_PAPER = 4
# Degree-2 AC has 6 coefficients; k'=4 is underdetermined. Use 8 as the working budget.
K_PRIME_HEADLINE = 8
POLICY_METHODS = ("AC", "A", "C", "random")
HEADLINE_LLMS = ("qwen25", "qwen3", "smollm2")

# Ground-truth table in the user's paper (42 encoders present on all three LLMs).
PAPER_ENCODERS: list[tuple[str, str, str]] = [
    ("siglip2_sm14_384", "SigLIP2 So400m/14 (384)", "Lang."),
    ("siglip2_g16_384", "SigLIP2 ViT-G/16 (384)", "Lang."),
    ("siglip2_l16_384", "SigLIP2 ViT-L/16 (384)", "Lang."),
    ("siglip2_sm16_512", "SigLIP2 So400m/16 (512)", "Lang."),
    ("siglip2_sm16_384", "SigLIP2 So400m/16 (384)", "Lang."),
    ("siglip2_g16_256", "SigLIP2 ViT-G/16 (256)", "Lang."),
    ("mc2_g14_378", "MetaCLIP 2 ViT-G/14 (378)", "Lang."),
    ("siglip2_sm14_224", "SigLIP2 So400m/14 (224)", "Lang."),
    ("pe_core_g14_448", "PE-Core-G/14 (448)", "SSL"),
    ("siglip2_sm16_256", "SigLIP2 So400m/16 (256)", "Lang."),
    ("siglip2_l16_256", "SigLIP2 ViT-L/16 (256)", "Lang."),
    ("mc1_g14_224_2.5b", "MetaCLIP ViT-G/14 (2.5B, 224)", "Lang."),
    ("mc2_g14_224", "MetaCLIP 2 ViT-G/14 (224)", "Lang."),
    ("siglip2_b16_512", "SigLIP2 ViT-B/16 (512)", "Lang."),
    ("mc1_h14_224_v1.2", "MetaCLIP ViT-H/14 (v1.2, 224)", "Lang."),
    ("clip_openai__l14", "OpenAI CLIP ViT-L/14 (224)", "Lang."),
    ("mc1_h14_224_2.5b", "MetaCLIP ViT-H/14 (2.5B, 224)", "Lang."),
    ("mc1_l14_224_2.5b", "MetaCLIP ViT-L/14 (2.5B, 224)", "Lang."),
    ("siglip2_b16_256", "SigLIP2 ViT-B/16 (256)", "Lang."),
    ("siglip2_b16_224", "SigLIP2 ViT-B/16 (224)", "Lang."),
    ("mc2_l14_224", "MetaCLIP 2 ViT-L/14 (224)", "Lang."),
    ("pe_lang_l14_448", "PE-Lang-L/14 (448)", "Lang."),
    ("pe_core_b16_224", "PE-Core-B/16 (224)", "SSL"),
    ("mc1_b16_224_400m", "MetaCLIP ViT-B/16 (400M, 224)", "Lang."),
    ("mc1_b16_224_2.5b", "MetaCLIP ViT-B/16 (2.5B, 224)", "Lang."),
    ("pixio_vitl16", "Pixio ViT-L/16", "SSL"),
    ("dinov2_giant", "DINOv2 ViT-G/14", "SSL"),
    ("dinov2_large", "DINOv2 ViT-L/14", "SSL"),
    ("pixio_vitb16", "Pixio ViT-B/16", "SSL"),
    ("eupe_vit_b", "EUPE ViT-B", "SSL"),
    ("eupe_vit_s", "EUPE ViT-S", "SSL"),
    ("mc2_s16_224", "MetaCLIP 2 ViT-S/16 (224)", "Lang."),
    ("dinov2_base", "DINOv2 ViT-B/14", "SSL"),
    ("dinov2_small", "DINOv2 ViT-S/14", "SSL"),
    ("pixio_vith16", "Pixio ViT-H/16", "SSL"),
    ("ijepa_vith14", "I-JEPA ViT-H/14", "SSL"),
    ("dino_vitb16", "DINO ViT-B/16", "SSL"),
    ("webssl_mae3b_full2b_224", "Web-SSL MAE 3B (224)", "Recon."),
    ("eupe_vit_t", "EUPE ViT-T", "SSL"),
    ("dino_vits8", "DINO ViT-S/8", "SSL"),
    ("dino_vitb8", "DINO ViT-B/8", "SSL"),
    ("dino_vits16", "DINO ViT-S/16", "SSL"),
]
PAPER_VISION_IDS = {vid for vid, _, _ in PAPER_ENCODERS}
PAPER_META = {vid: {"name": name, "type": typ} for vid, name, typ in PAPER_ENCODERS}

BENCH_KEYS = [
    "MMMU_TEST",
    "MMBench_TEST_EN_V11",
    "VQAv2_VAL",
    "ScienceQA_VAL",
    "ChartQA_TEST",
    "DocVQA_VAL",
    "TextVQA_VAL",
    "POPE",
    "GQA_TestDev_Balanced",
    "MSCOCO_KARPATHY_TEST",
    "FLICKR30K_KARPATHY_TEST",
    "average",
]


def _score_fields(prefix: str) -> tuple[str, ...]:
    if prefix.startswith("a_score"):
        return ("a_score",)
    if prefix.startswith("c_score"):
        return ("c_score",)
    return ()


def _prefer_record(old: dict, new: dict, score_fields: tuple[str, ...]) -> dict:
    if not score_fields:
        return new
    old_ok = any(old.get(k) is not None for k in score_fields)
    new_ok = any(new.get(k) is not None for k in score_fields)
    if new_ok:
        return new
    if old_ok:
        return old
    return new


def merge_shards(prefix: str) -> dict:
    out = {}
    score_fields = _score_fields(prefix)
    for p in sorted(RESULTS_DIR.glob(f"{prefix}_shard*.json")):
        data = json.loads(p.read_text())
        items = []
        if isinstance(data, dict):
            items = [(rec.get("key") or rec.get("vision_id") or k, rec) for k, rec in data.items() if isinstance(rec, dict)]
        elif isinstance(data, list):
            items = [((rec.get("key") or rec.get("vision_id")), rec) for rec in data if isinstance(rec, dict)]
        for key, rec in items:
            if not key:
                continue
            out[key] = _prefer_record(out[key], rec, score_fields) if key in out else rec
    return out


def pearson(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d else 0.0


def spearman(x, y) -> float:
    def rank(a):
        a = np.asarray(a, float)
        o = a.argsort()
        r = np.empty_like(o, dtype=float)
        r[o] = np.arange(len(a))
        return r

    return pearson(rank(x), rank(y))


def minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < 1e-12:
        return np.zeros_like(v, dtype=float)
    return (v - lo) / (hi - lo)


def fit_poly(x: np.ndarray, y: np.ndarray, degree: int = 2):
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    xt = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(xt, y)
    pred = model.predict(xt)
    return poly, model, pred


def loo_preds(x: np.ndarray, y: np.ndarray, degree: int = 2) -> np.ndarray:
    n = len(y)
    pred = np.zeros(n)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        xt = poly.fit_transform(x[mask])
        model = LinearRegression().fit(xt, y[mask])
        pred[i] = model.predict(poly.transform(x[i : i + 1]))[0]
    return pred


def k_prime_grid(n: int) -> list[int]:
    """Sweep k' from 4 up to half the tokenizers. Headline budget is 8."""
    hi = n // 2
    if hi < 4:
        return []
    pts = {K_PRIME_PAPER, K_PRIME_HEADLINE, hi}
    pts.update(range(4, min(16, hi) + 1, 2))
    pts.update(range(16, hi + 1, 4))
    return sorted(k for k in pts if 4 <= k <= hi)


def _empty_acc(benches: list[str]) -> dict:
    return {b: {"rho": [], "best": [], "top3": [], "rank": []} for b in benches}


def _record_pick(acc_b: dict, pred: np.ndarray, yte: np.ndarray) -> None:
    acc_b["rho"].append(spearman(pred, yte))
    pick = int(np.argmax(pred))
    true_order = np.argsort(-yte)
    acc_b["best"].append(int(pick == true_order[0]))
    acc_b["top3"].append(int(pick in set(true_order[:3])))
    acc_b["rank"].append(int(np.where(true_order == pick)[0][0] + 1))


def _summarize_acc(acc: dict, n_train: int, n_test: int) -> dict:
    out = {"n_train": n_train, "n_test": n_test, "is_paper_k_prime": n_train == K_PRIME_PAPER, "benches": {}}
    for b, a in acc.items():
        if not a["rho"]:
            continue
        out["benches"][b] = {
            "spearman": float(np.mean(a["rho"])),
            "spearman_std": float(np.std(a["rho"])),
            "pick_is_true_best": float(np.mean(a["best"])),
            "pick_in_true_top3": float(np.mean(a["top3"])),
            "mean_rank_of_pick": float(np.mean(a["rank"])),
        }
    return out


def policy_sim_methods(
    df: pd.DataFrame,
    benches: list[str],
    n_train: int,
    n_repeat: int = 100,
    seed: int = 0,
) -> dict:
    """Fit on k' models; rank the remainder. Compare AC / A / C / random."""
    rng = np.random.RandomState(seed)
    n = len(df)
    if n_train < 3 or n_train >= n - 1:
        return {}
    xs = {
        "AC": df[["norm_a", "norm_c"]].to_numpy(),
        "A": df[["norm_a"]].to_numpy(),
        "C": df[["norm_c"]].to_numpy(),
    }
    ys = {}
    for b in benches:
        if b not in df.columns:
            continue
        col = df[b].to_numpy(dtype=float)
        if np.isfinite(col).sum() < n_train + 2:
            continue
        ys[b] = col
    if not ys:
        return {}
    acc = {m: _empty_acc(list(ys)) for m in POLICY_METHODS}
    poly_feat = PolynomialFeatures(degree=2, include_bias=True)
    for _ in range(n_repeat):
        idx = rng.permutation(n)
        tr, te = idx[:n_train], idx[n_train:]
        for method in ("AC", "A", "C"):
            x = xs[method]
            xtr = poly_feat.fit_transform(x[tr])
            xte = poly_feat.transform(x[te])
            for b, y in ys.items():
                ytr, yte = y[tr], y[te]
                mtr = np.isfinite(ytr)
                mte = np.isfinite(yte)
                if mtr.sum() < 3 or mte.sum() < 3:
                    continue
                model = LinearRegression().fit(xtr[mtr], ytr[mtr])
                pred = model.predict(xte[mte])
                _record_pick(acc[method][b], pred, yte[mte])
        for b, y in ys.items():
            yte = y[te]
            mte = np.isfinite(yte)
            if mte.sum() < 3:
                continue
            pred = rng.rand(int(mte.sum()))
            _record_pick(acc["random"][b], pred, yte[mte])
    methods = {m: _summarize_acc(acc[m], n_train, n - n_train) for m in POLICY_METHODS}
    return {
        "n_train": n_train,
        "n_test": n - n_train,
        "is_paper_k_prime": n_train == K_PRIME_PAPER,
        "methods": methods,
        # keep AC at top level so older readers of ac_fit.json still work
        "benches": methods["AC"].get("benches") or {},
    }


def run_for_split(df: pd.DataFrame, split_name: str, run_policy: bool = True) -> dict:
    df = df.dropna(subset=["a_score", "c_score", "average"]).copy()
    df = df.reset_index(drop=True)
    if len(df) < 6:
        return {"split": split_name, "n": len(df), "error": "too_few_models"}
    df["norm_a"] = minmax(df["a_score"].to_numpy())
    df["norm_c"] = minmax(df["c_score"].to_numpy())
    x_ac = df[["norm_a", "norm_c"]].to_numpy()
    x_a = df[["norm_a"]].to_numpy()
    x_c = df[["norm_c"]].to_numpy()

    bench_stats = {}
    for bench in BENCH_KEYS:
        if bench not in df.columns or df[bench].isna().all():
            continue
        y = minmax(df[bench].to_numpy())
        stats = {}
        for name, x, deg in (("AC_poly", x_ac, 2), ("A_poly", x_a, 2), ("C_poly", x_c, 2), ("A_lin", x_a, 1), ("C_lin", x_c, 1)):
            _, _, pred = fit_poly(x, y, deg)
            loo = loo_preds(x, y, deg)
            stats[name] = {
                "in_r2": float(r2_score(y, pred)),
                "in_mse": float(mean_squared_error(y, pred)),
                "loo_r2": float(r2_score(y, loo)),
                "loo_mse": float(mean_squared_error(y, loo)),
                "loo_pearson": pearson(y, loo),
                "loo_spearman": spearman(y, loo),
            }
        stats["corr_a"] = pearson(df["a_score"], df[bench])
        stats["corr_c"] = pearson(df["c_score"], df[bench])
        stats["rho_a"] = spearman(df["a_score"], df[bench])
        stats["rho_c"] = spearman(df["c_score"], df[bench])
        bench_stats[bench] = stats

    y = df["average"].to_numpy()
    yn = minmax(y)
    poly, model, pred_n = fit_poly(x_ac, yn, 2)
    loo_n = loo_preds(x_ac, yn, 2)
    df["pred_in_norm"] = pred_n
    df["pred_loo_norm"] = loo_n
    df["pred_in"] = pred_n * (y.max() - y.min()) + y.min()
    df["pred_loo"] = loo_n * (y.max() - y.min()) + y.min()

    policy = []
    if run_policy:
        for k in k_prime_grid(len(df)):
            rec = policy_sim_methods(df, BENCH_KEYS, k, n_repeat=100, seed=0)
            if rec and rec.get("benches"):
                policy.append(rec)

    actual_best = df.loc[df["average"].idxmax(), "key"]
    pred_best = df.loc[df["pred_in"].idxmax(), "key"]
    loo_best = df.loc[df["pred_loo"].idxmax(), "key"]

    return {
        "split": split_name,
        "n": int(len(df)),
        "actual_best": actual_best,
        "pred_in_best": pred_best,
        "pred_loo_best": loo_best,
        "poly_coef": model.coef_.tolist(),
        "poly_intercept": float(model.intercept_),
        "benchmarks": bench_stats,
        "policy": policy,
        "table": df[
            [
                "key",
                "llm_id",
                "vision_id",
                "a_score",
                "c_score",
                "norm_a",
                "norm_c",
                "average",
                "pred_in",
                "pred_loo",
            ]
        ].to_dict(orient="records"),
    }


def plot_policy_ac(summaries: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for s in summaries:
        if s.get("error") or not s.get("policy") or s["split"] not in HEADLINE_LLMS:
            continue
        ks, rho, r3, rank = [], [], [], []
        for p in s["policy"]:
            avg = (p.get("benches") or {}).get("average") or {}
            if not avg:
                continue
            ks.append(p["n_train"])
            rho.append(avg.get("spearman", float("nan")))
            r3.append(avg.get("pick_in_true_top3", float("nan")))
            rank.append(avg.get("mean_rank_of_pick", float("nan")))
        if not ks:
            continue
        axes[0].plot(ks, rho, marker="o", label=s["split"])
        axes[1].plot(ks, r3, marker="o", label=s["split"])
        axes[2].plot(ks, rank, marker="o", label=s["split"])
    for ax in axes:
        ax.axvline(K_PRIME_PAPER, color="0.5", ls=":", lw=1)
        ax.axvline(K_PRIME_HEADLINE, color="k", ls="--", lw=1)
        ax.set_xlabel("k' (finetuned models)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("held-out Spearman")
    axes[1].set_ylabel("Recall@3 of AC pick")
    axes[2].set_ylabel("mean rank of AC pick")
    axes[0].set_title("AC policy: rank remaining tokenizers")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "ac_policy_vs_kprime.png", dpi=140)
    plt.close(fig)


def plot_policy_methods(summaries: list[dict]) -> None:
    """Paper Fig.4 analogue: AC vs A vs C vs random, per LLM."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    style = {
        "AC": dict(color="tab:blue", marker="o", lw=2),
        "A": dict(color="tab:green", marker="s", lw=1.5, ls="--"),
        "C": dict(color="tab:orange", marker="^", lw=1.5, ls="--"),
        "random": dict(color="0.5", marker="x", lw=1, ls=":"),
    }
    plotted = 0
    for ax, s in zip(axes, [x for x in summaries if x.get("split") in HEADLINE_LLMS]):
        if s.get("error") or not s.get("policy"):
            ax.set_visible(False)
            continue
        plotted += 1
        for method in POLICY_METHODS:
            ks, rho = [], []
            for p in s["policy"]:
                avg = (((p.get("methods") or {}).get(method) or {}).get("benches") or {}).get("average") or {}
                if not avg:
                    continue
                ks.append(p["n_train"])
                rho.append(avg.get("spearman", float("nan")))
            if ks:
                ax.plot(ks, rho, label=method, **style[method])
        ax.axvline(K_PRIME_PAPER, color="0.5", ls=":", lw=1)
        ax.axvline(K_PRIME_HEADLINE, color="k", ls="--", lw=1)
        ax.set_xlabel("k' (finetuned models)")
        ax.set_title(s["split"])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    if plotted:
        axes[0].set_ylabel("held-out Spearman (average)")
        fig.suptitle("AC policy vs A-only / C-only / random", y=1.02, fontsize=11)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "ac_policy_methods.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, split: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(df["a_score"], df["average"], s=18, alpha=0.8)
    axes[0].set_xlabel("A-score (-NLL)")
    axes[0].set_ylabel("finish.json average")
    axes[0].set_title(f"{split}: A vs performance")
    axes[1].scatter(df["c_score"], df["average"], s=18, alpha=0.8, color="tab:orange")
    axes[1].set_xlabel("C-score (PCK@0.10)")
    axes[1].set_title(f"{split}: C vs performance")
    axes[2].scatter(df["pred_loo"], df["average"], s=18, alpha=0.8, color="tab:green")
    lo, hi = df["average"].min(), df["average"].max()
    axes[2].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[2].set_xlabel("LOO AC-poly prediction")
    axes[2].set_title(f"{split}: AC predicted vs actual")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"ac_scatter_{split}.png", dpi=140)
    plt.close(fig)


def plot_surface(df: pd.DataFrame, split: str) -> None:
    """Paper-style 3D AC surface: C × A → predicted MLLM average."""
    d = df.dropna(subset=["a_score", "c_score", "average", "pred_in"]).copy()
    if len(d) < 6:
        return
    x_ac = np.column_stack([minmax(d["a_score"].to_numpy()), minmax(d["c_score"].to_numpy())])
    y = d["average"].to_numpy()
    yn = minmax(y)
    poly, model, _ = fit_poly(x_ac, yn, 2)
    grid = np.linspace(0, 1, 40)
    aa, cc = np.meshgrid(grid, grid)
    z_n = model.predict(poly.transform(np.column_stack([aa.ravel(), cc.ravel()]))).reshape(aa.shape)
    z = z_n * (y.max() - y.min()) + y.min()
    a_lo, a_hi = float(d["a_score"].min()), float(d["a_score"].max())
    c_lo, c_hi = float(d["c_score"].min()), float(d["c_score"].max())
    a_grid = aa * (a_hi - a_lo) + a_lo
    c_grid = cc * (c_hi - c_lo) + c_lo

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(c_grid, a_grid, z, color="tab:blue", alpha=0.35, linewidth=0, antialiased=True)
    ax.scatter(d["c_score"], d["a_score"], d["average"], c="tab:orange", s=18, depthshade=True)
    ax.set_xlabel("C-score (PCK@0.10)", labelpad=6)
    ax.set_ylabel("A-score (-NLL)", labelpad=6)
    ax.set_zlabel("MLLM average", labelpad=6)
    ax.set_title(f"{split}: AC polynomial surface")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"ac_surface_{split}.png", dpi=140)
    plt.close(fig)


def _avg_stats(summary: dict) -> dict:
    return (summary.get("benchmarks") or {}).get("average") or {}


def _k_at(summary: dict, k: int, method: str = "AC") -> dict:
    for p in summary.get("policy") or []:
        if p.get("n_train") == k:
            if method == "AC":
                return (p.get("benches") or {}).get("average") or {}
            return (((p.get("methods") or {}).get(method) or {}).get("benches") or {}).get("average") or {}
    return {}


def write_paper_brief(summaries: list[dict]) -> str:
    by = {s["split"]: s for s in summaries}
    lines = [
        "Law of Vision Representation — AC polynomial on the 42-encoder GT table",
        "",
        f"Pool: {len(PAPER_VISION_IDS)} vision encoders from tab:ground_truth_mllm.",
        "Claim: MLLM score ≈ degree-2 poly(A, C). Fits are within one LLM.",
        f"Headline budget k'={K_PRIME_HEADLINE} (6 AC coefficients; k'=4 is underdetermined).",
        "k'=4 is kept only as a paper Fig.4 reference.",
        "",
        "## Leave-one-out: AC vs A vs C  (target = finish.json average)",
        f"{'LLM':<22}{'n':>5}  {'AC ρ':>8}  {'A ρ':>8}  {'C ρ':>8}  {'AC R²':>8}  {'A R²':>8}  {'C R²':>8}",
    ]
    paper = {"loo": {}, "k4": {}, "k8": {}, "best": {}, "pool": list(PAPER_VISION_IDS), "k_headline": K_PRIME_HEADLINE}
    for name in HEADLINE_LLMS + ("qwen25_wo_eupe_t",):
        s = by.get(name)
        if not s or s.get("error"):
            continue
        avg = _avg_stats(s)
        ac, a, c = avg.get("AC_poly") or {}, avg.get("A_lin") or {}, avg.get("C_lin") or {}
        lines.append(
            f"{name:<22}{s.get('n'):5d}  {ac.get('loo_spearman', float('nan')):8.3f}  "
            f"{a.get('loo_spearman', float('nan')):8.3f}  {c.get('loo_spearman', float('nan')):8.3f}  "
            f"{ac.get('loo_r2', float('nan')):8.3f}  {a.get('loo_r2', float('nan')):8.3f}  "
            f"{c.get('loo_r2', float('nan')):8.3f}"
        )
        paper["loo"][name] = {
            "n": s.get("n"),
            "ac_spearman": ac.get("loo_spearman"),
            "a_spearman": a.get("loo_spearman"),
            "c_spearman": c.get("loo_spearman"),
            "ac_r2": ac.get("loo_r2"),
            "a_r2": a.get("loo_r2"),
            "c_r2": c.get("loo_r2"),
            "ac_in_r2": ac.get("in_r2"),
        }
    mixed = by.get("all")
    if mixed and not mixed.get("error"):
        ac = _avg_stats(mixed).get("AC_poly") or {}
        lines.append(
            f"{'all (mixed LLM)':<22}{mixed.get('n'):5d}  {ac.get('loo_spearman', float('nan')):8.3f}  "
            f"{'n/a':>8}  {'n/a':>8}  {ac.get('loo_r2', float('nan')):8.3f}  invalid mix, ignore"
        )

    for k, key in ((K_PRIME_HEADLINE, "k8"), (K_PRIME_PAPER, "k4")):
        tag = "headline" if k == K_PRIME_HEADLINE else "paper Fig.4 reference"
        lines += [
            "",
            f"## AC policy at k' = {k}  ({tag}; 100 random draws)",
            f"{'LLM':<22}{'AC ρ':>8}  {'A ρ':>8}  {'C ρ':>8}  {'rand ρ':>8}  {'AC R@3':>8}  {'A R@3':>8}  {'rand R@3':>8}",
        ]
        for name in HEADLINE_LLMS:
            s = by.get(name)
            if not s:
                continue
            ac, a, c, rnd = _k_at(s, k, "AC"), _k_at(s, k, "A"), _k_at(s, k, "C"), _k_at(s, k, "random")
            lines.append(
                f"{name:<22}{ac.get('spearman', float('nan')):8.3f}  {a.get('spearman', float('nan')):8.3f}  "
                f"{c.get('spearman', float('nan')):8.3f}  {rnd.get('spearman', float('nan')):8.3f}  "
                f"{ac.get('pick_in_true_top3', float('nan')):8.3f}  {a.get('pick_in_true_top3', float('nan')):8.3f}  "
                f"{rnd.get('pick_in_true_top3', float('nan')):8.3f}"
            )
            paper[key][name] = {"ac": ac, "A": a, "C": c, "random": rnd}

    lines += [
        "",
        "## Predicted vs actual best tokenizer (in-sample AC surface)",
        f"{'LLM':<12}{'actual best':<42}{'AC-predicted best'}",
    ]
    for name in HEADLINE_LLMS:
        s = by.get(name)
        if not s:
            continue
        actual = str(s.get("actual_best") or "").split("/")[-1]
        pred = str(s.get("pred_in_best") or "").split("/")[-1]
        lines.append(f"{name:<12}{actual:<42}{pred}")
        paper["best"][name] = {"actual": s.get("actual_best"), "pred_in": s.get("pred_in_best"), "pred_loo": s.get("pred_loo_best")}

    lines += [
        "",
        "Notes:",
        "- Restricted to the 42 encoders in tab:ground_truth_mllm.",
        "- qwen25 AC LOO R² can still break on eupe_vit_t; Spearman is the ranking metric.",
        "- C-score is tokenizer-level; A-score is Stage-1 NLL of that MLLM.",
        "- Figures: ac_surface_{llm}.png, ac_policy_methods.png, ac_scatter_{llm}.png",
        "",
    ]
    for s in summaries:
        if s.get("split") not in HEADLINE_LLMS or s.get("error") or not s.get("policy"):
            continue
        lines.append(f"# {s['split']} n={s.get('n')}  held-out Spearman vs k' (AC poly)")
        ks = [p["n_train"] for p in s["policy"]]
        header = "    benchmark".ljust(28) + "".join(f"{k:>7d}" for k in ks)
        lines.append(header)
        for bench in BENCH_KEYS:
            row = f"    {bench:<24}"
            any_v = False
            for p in s["policy"]:
                v = (p.get("benches") or {}).get(bench, {}).get("spearman")
                if v is None:
                    row += "      -"
                else:
                    any_v = True
                    row += f"{v:7.3f}"
            if any_v:
                lines.append(row)
        lines.append("")

    text = "\n".join(lines)
    (RESULTS_DIR / "ac_paper_brief.txt").write_text(text)
    atomic_json(RESULTS_DIR / "ac_paper.json", paper)
    return text


def main():
    rows = inventory_models()
    a_map = merge_shards("a_score")
    c_map = merge_shards("c_score")
    records = []
    for r in rows:
        a = a_map.get(r["key"]) or {}
        c = c_map.get(r.get("vision_id")) or {}
        rec = {
            "key": r["key"],
            "llm_id": r.get("llm_id"),
            "vision_id": r.get("vision_id"),
            "a_score": a.get("a_score"),
            "c_score": c.get("c_score"),
            "average": r.get("average"),
        }
        scores = r.get("scores") or {}
        for b in BENCH_KEYS:
            rec[b] = scores.get(b)
        records.append(rec)
    df = pd.DataFrame(records)
    df = df[df["vision_id"].isin(PAPER_VISION_IDS)].copy()
    missing = PAPER_VISION_IDS - set(df["vision_id"].dropna())
    if missing:
        print(f"warning: paper encoders missing from inventory: {sorted(missing)}", flush=True)
    df.to_csv(RESULTS_DIR / "ac_joined.csv", index=False)

    summaries = []
    splits = {"all": df}
    for llm in sorted(df["llm_id"].dropna().unique()):
        splits[str(llm)] = df[df["llm_id"] == llm].copy()
    q25 = splits.get("qwen25")
    if q25 is not None:
        splits["qwen25_wo_eupe_t"] = q25[q25["vision_id"] != "eupe_vit_t"].copy()

    for name, sub in splits.items():
        run_policy = name in HEADLINE_LLMS
        print(f"fitting split={name} n={len(sub.dropna(subset=['a_score','c_score','average']))} policy={run_policy}", flush=True)
        summary = run_for_split(sub, name, run_policy=run_policy)
        summaries.append(summary)
        if "table" in summary:
            tdf = pd.DataFrame(summary["table"])
            tdf.to_csv(RESULTS_DIR / f"ac_pred_{name}.csv", index=False)
            plot_df = sub.dropna(subset=["a_score", "c_score", "average"]).copy()
            if len(plot_df) >= 6:
                plot_df = plot_df.merge(tdf[["key", "pred_in", "pred_loo"]], on="key", how="left")
                plot_scatter(plot_df, name)
                if name in HEADLINE_LLMS or name == "qwen25_wo_eupe_t":
                    plot_surface(plot_df, name)

    atomic_json(RESULTS_DIR / "ac_fit.json", summaries)
    ac_map = {}
    for rec in records:
        key = rec.get("key")
        if not key:
            continue
        ac_map[key] = {"a_score": rec.get("a_score"), "c_score": rec.get("c_score")}
    atomic_json(RESULTS_DIR / "ac_score.json", ac_map)
    plot_policy_ac(summaries)
    plot_policy_methods(summaries)
    write_gt_table(summaries)
    brief = write_paper_brief(summaries)
    (RESULTS_DIR / "ac_fit_brief.txt").write_text(brief)
    print(brief, flush=True)


def write_gt_table(summaries: list[dict]) -> None:
    """CSV matching tab:ground_truth_mllm plus per-LLM AC predictions."""
    by = {s["split"]: s for s in summaries if "table" in s}
    rows = []
    for rank, (vid, name, typ) in enumerate(PAPER_ENCODERS, start=1):
        rec = {"gt_rank": rank, "vision_id": vid, "name": name, "type": typ}
        avgs = []
        for llm in HEADLINE_LLMS:
            table = {r["vision_id"]: r for r in (by.get(llm) or {}).get("table") or []}
            r = table.get(vid) or {}
            rec[f"{llm}_actual"] = r.get("average")
            rec[f"{llm}_pred"] = r.get("pred_in")
            rec[f"{llm}_loo"] = r.get("pred_loo")
            rec[f"{llm}_A"] = r.get("a_score")
            rec[f"{llm}_C"] = r.get("c_score")
            if r.get("average") is not None:
                avgs.append(float(r["average"]))
        rec["mean_actual"] = float(sum(avgs) / len(avgs)) if avgs else None
        preds = [rec[f"{llm}_pred"] for llm in HEADLINE_LLMS if rec.get(f"{llm}_pred") is not None]
        rec["mean_pred"] = float(sum(preds) / len(preds)) if preds else None
        rows.append(rec)
    if any(r.get("mean_actual") is not None for r in rows):
        ranked = sorted((r for r in rows if r.get("mean_actual") is not None), key=lambda x: -x["mean_actual"])
        for i, r in enumerate(ranked, start=1):
            r["rank_actual"] = i
        ranked_p = sorted((r for r in rows if r.get("mean_pred") is not None), key=lambda x: -x["mean_pred"])
        for i, r in enumerate(ranked_p, start=1):
            r["rank_pred"] = i
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "ac_full_table.csv", index=False)
    atomic_json(RESULTS_DIR / "ac_full_table.json", rows)


if __name__ == "__main__":
    main()
