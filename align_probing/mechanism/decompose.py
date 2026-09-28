"""Split PCA whitening into sub-operations on the paper's MutualNN setting (k=10, global feats)."""
import numpy as np, json
from pathlib import Path
from tqdm import tqdm
import analyze as A, official as O_
import sys; sys.path.insert(0, "/home/ma-user/work_space/RAVEL"); from src.preprocessing import preprocess  # noqa

S, LL = A.SLUGS, A.LLMS
C = Path("/cache/ravel_multiseed/union_43_44")
CUSTOM = {"webssl_dino1b_full2b_224", "dinov3_vitl16", "ijepa_vith14", "raev2_dinov3l_k7"}
K = 10


def feats(seed):
    pos = np.load(f"/cache/ravel_multiseed/union_positions_seed{seed}.npy")
    v = {s: np.asarray(np.load(C / ("custom/global" if s in CUSTOM else "global") / f"img_feats_{s}.npy",
                               mmap_mode="r")[pos], np.float64) for s in tqdm(S, desc="visual", leave=False)}
    t = {l: np.asarray(np.load(C / "text" / f"text_{l}.npy", mmap_mode="r")[pos], np.float64) for l in LL}
    return v, t


def proc(x, mode):
    x = np.asarray(x, np.float64)
    if mode == "raw":
        z = x
    elif mode == "center":
        z = x - x.mean(0)
    elif mode in ("pca", "whiten", "whiten_full"):
        xc = x - x.mean(0)
        ev, U = np.linalg.eigh(np.cov(xc.T)); o = np.argsort(ev)[::-1]; ev, U = ev[o].clip(0), U[:, o]
        r = int((ev > 1e-10 * ev[0]).sum()); r = r if mode == "whiten_full" else min(256, r)
        z = xc @ U[:, :r]
        if mode != "pca": z = z / np.sqrt(ev[:r] + 1e-5)
    elif mode == "variance_only":           # equalise variance, keep all dims, no truncation, no centering of cone
        xc = x - x.mean(0)
        ev, U = np.linalg.eigh(np.cov(xc.T)); o = np.argsort(ev)[::-1]; ev, U = ev[o].clip(0), U[:, o]
        r = int((ev > 1e-10 * ev[0]).sum()); z = (xc @ U[:, :r]) / np.sqrt(ev[:r] + 1e-5)
    z = z / np.linalg.norm(z, axis=1, keepdims=True).clip(1e-12)
    return z.astype(np.float32)


def knn(z):
    s = z @ z.T; np.fill_diagonal(s, -np.inf)
    return np.argpartition(-s, K, axis=1)[:, :K]


def overlap(a, b):
    return float(np.mean([len(np.intersect1d(a[i], b[i], assume_unique=True)) for i in range(len(a))]) / K)


def run(seed, modes):
    v, t = feats(seed)
    tk = {m: {l: knn(proc(t[l], m)) for l in LL} for m in modes}
    res = {m: {l: [] for l in LL} for m in modes}
    for s in tqdm(S, desc=f"seed{seed}"):
        for m in modes:
            vk = knn(proc(v[s], m))
            for l in LL: res[m][l].append(overlap(vk, tk[m][l]))
    return {m: {l: np.array(res[m][l]) for l in LL} for m in modes}


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 43
    modes = ["raw", "center", "pca", "whiten", "whiten_full"]
    R = run(seed, modes)
    json.dump({m: {l: R[m][l].tolist() for l in LL} for m in modes}, open(f"decompose_seed{seed}.json", "w"))
    off = O_.load()[seed]
    print(f"official S0 rho: " + " ".join(f"{A.spearman(off['S0_mutualnn'][l], A.gt(l)):.3f}" for l in LL))
    print(f"official S1 rho: " + " ".join(f"{A.spearman(off['S1_whiten'][l], A.gt(l)):.3f}" for l in LL))
    for m in modes:
        rh = [A.spearman(R[m][l], A.gt(l)) for l in LL]
        print(f"{m:12s} rho " + " ".join(f"{x:.3f}" for x in rh) + f"  avg {np.mean(rh):.3f}")
