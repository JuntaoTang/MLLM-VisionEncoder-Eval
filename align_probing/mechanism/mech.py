#!/usr/bin/env python3
"""Mechanism analyses on top of ladder_scores.json."""
import json, itertools
import numpy as np
import analyze as A

L = json.load(open(A.M / "ladder_scores.json"))
S = A.SLUGS
LLMS = A.LLMS
VARS = ["V0_mutualnn", "V1_whiten", "V2_patch", "V3_ravel", "X_patch_nowhiten", "X_whiten_weighted"]


def score(v, l): return np.array([L[s]["scores"][l][v] for s in S])


def sail(l, key="score_5cap"):
    root = "/cache/wangky/align_probe/results/sweeps/cc3m2k"
    return np.array([json.load(open(f"{root}/{l}/{s}.json"))[key] for s in S])


# ---------------- anisotropy of pooled visual features ----------------
def anisotropy():
    out = {}
    for s in S:
        x = np.load(A.M / "cache" / f"{s}.npz")["pooled_raw"].astype(np.float64)
        x = x - x.mean(0)
        ev = np.linalg.eigvalsh(np.cov(x.T))[::-1].clip(0)
        out[s] = {"top1": float(ev[0] / ev.sum()),
                  "pr": float(ev.sum() ** 2 / (ev ** 2).sum()),          # participation ratio
                  "cos_mean": None}
        xn = np.load(A.M / "cache" / f"{s}.npz")["pooled_raw"].astype(np.float64)
        xn = xn / np.linalg.norm(xn, axis=1, keepdims=True)
        g = xn @ xn.T
        out[s]["cos_mean"] = float((g.sum() - np.trace(g)) / (len(g) * (len(g) - 1)))
    return out


# ---------------- controlled within-family pairs ----------------
RES = {  # same architecture & size, only input resolution differs
    "siglip2_b16": ["siglip2_b16_224", "siglip2_b16_256", "siglip2_b16_384", "siglip2_b16_512"],
    "siglip2_l16": ["siglip2_l16_256", "siglip2_l16_384", "siglip2_l16_512"],
    "siglip2_sm16": ["siglip2_sm16_256", "siglip2_sm16_384", "siglip2_sm16_512"],
    "siglip2_sm14": ["siglip2_sm14_224", "siglip2_sm14_384"],
    "siglip2_g16": ["siglip2_g16_256", "siglip2_g16_384"],
    "mc2_b16": ["mc2_b16_224", "mc2_b16_384"], "mc2_b32": ["mc2_b32_224", "mc2_b32_384"],
    "mc2_m16": ["mc2_m16_224", "mc2_m16_384"], "mc2_s16": ["mc2_s16_224", "mc2_s16_384"],
    "mc2_g14": ["mc2_g14_224", "mc2_g14_378"],
}
SIZE = {  # same family & resolution, only model size differs
    "siglip2@256": ["siglip2_b16_256", "siglip2_l16_256", "siglip2_sm16_256", "siglip2_g16_256"],
    "siglip2@384": ["siglip2_b16_384", "siglip2_l16_384", "siglip2_sm16_384", "siglip2_g16_384"],
    "siglip2@512": ["siglip2_b16_512", "siglip2_l16_512", "siglip2_sm16_512"],
    "mc2@224": ["mc2_s16_224", "mc2_m16_224", "mc2_b16_224", "mc2_l14_224", "mc2_g14_224"],
    "mc1_2.5b": ["mc1_b16_224_2.5b", "mc1_l14_224_2.5b", "mc1_h14_224_2.5b", "mc1_g14_224_2.5b"],
    "dinov2": ["dinov2_small", "dinov2_base", "dinov2_large", "dinov2_giant"],
    "dino16": ["dino_vits16", "dino_vitb16"], "dino8": ["dino_vits8", "dino_vitb8"],
    "eupe_vit": ["eupe_vit_t", "eupe_vit_s", "eupe_vit_b"],
    "pixio": ["pixio_vitb16", "pixio_vitl16", "pixio_vith16"],
    "webssl_mae": ["webssl_mae300m_full2b_224", "webssl_mae1b_full2b_224", "webssl_mae3b_full2b_224"],
}


def pair_acc(groups, values, l, min_gap=0.0):
    """Fraction of within-group pairs whose GT order the method reproduces."""
    g = dict(zip(S, A.gt(l))); v = dict(zip(S, values))
    hit = tot = 0
    for members in groups.values():
        for a, b in itertools.combinations(members, 2):
            dg = g[b] - g[a]
            if abs(dg) <= min_gap: continue
            tot += 1; hit += int(np.sign(v[b] - v[a]) == np.sign(dg))
    return hit, tot


if __name__ == "__main__":
    print("== ladder (Spearman vs GT) ==")
    for v in VARS:
        rh = [A.spearman(score(v, l), A.gt(l)) for l in LLMS]
        pr = [A.pearson(score(v, l), A.gt(l)) for l in LLMS]
        t1 = [A.gt(l)[int(np.argmax(score(v, l)))] for l in LLMS]
        print(f"{v:20s} rho " + " ".join(f"{x:.3f}" for x in rh) + f" | avg {np.mean(rh):.3f}"
              f" | r avg {np.mean(pr):.3f} | top1 avg {np.mean(t1):.2f}")
    rh = [A.spearman(sail(l), A.gt(l)) for l in LLMS]
    print(f"{'SAIL(5cap)':20s} rho " + " ".join(f"{x:.3f}" for x in rh) + f" | avg {np.mean(rh):.3f}")
