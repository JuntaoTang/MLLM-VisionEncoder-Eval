from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
K_PRIME_PAPER = 4
K_PRIME_HEADLINE = 8
POLICY_METHODS = ('AC','A','C','random')

def preprocess_kps_pad(kps: torch.Tensor, img_width: int, img_height: int, size: int):
    kps = kps.clone()
    scale = size / max(img_width, img_height)
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        offset_y = int((size - new_h) / 2)
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        offset_x = int((size - new_w) / 2)
        kps[:, 0] += offset_x
    kps *= kps[:, 2:3].clone()
    return kps, scale

def windowed_soft_argmax(sim: torch.Tensor, num_patches: int, window: int) -> torch.Tensor:
    """sim: (N, N) source-patch x target-patch cosine. Return (N, 2) xy in patch coords."""
    n = num_patches
    device = sim.device
    dtype = sim.dtype
    corr = sim.view(n * n, n, n)
    max_flat = corr.flatten(1).argmax(dim=-1)
    max_x = max_flat % n
    max_y = max_flat // n
    if window and window > 0:
        yy = torch.arange(n, device=device)
        xx = torch.arange(n, device=device)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        mask = (gx.unsqueeze(0) - max_x.view(-1, 1, 1)).abs() <= window
        mask &= (gy.unsqueeze(0) - max_y.view(-1, 1, 1)).abs() <= window
        corr = corr.masked_fill(~mask, -1e9)
    corr = corr.view(n * n, n * n)
    prob = torch.softmax(corr, dim=-1)
    ys = torch.arange(n, dtype=dtype, device=device).repeat_interleave(n)
    xs = torch.arange(n, dtype=dtype, device=device).repeat(n)
    nn_x = (prob * xs).sum(dim=-1)
    nn_y = (prob * ys).sum(dim=-1)
    return torch.stack([nn_x, nn_y], dim=-1)

def _prep_grid(g: torch.Tensor, device: str) -> torch.Tensor:
    g = g.to(device=device, dtype=torch.float32)
    if g.shape[-2] != g.shape[-1]:
        side = int(round(math.sqrt(g.shape[-2] * g.shape[-1])))
        g = F.interpolate(g.unsqueeze(0), size=(side, side), mode="bilinear", align_corners=False)[0]
    return g


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
