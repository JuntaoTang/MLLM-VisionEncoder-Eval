#!/usr/bin/env python3
"""Component ladder MutualNN -> RAVEL on identical inputs, plus scoring helpers."""
import json, sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/home/ma-user/work_space/RAVEL")
sys.path.insert(0, "/home/ma-user/work_space/RAVEL/experiments/patch_ravel_scaling_n5000_seed42/scripts")
from src.preprocessing import preprocess                       # noqa: E402
from graph_ops import neighbor_graph, distribution, overlap_score, dense_distribution  # noqa: E402

M = Path("/cache/wangky/align_probe/mechanism")
IDX = np.load("/home/ma-user/work_space/RAVEL/experiments/patch_ravel_scaling_n5000_seed42/"
              "inputs/indices_n2000_seed42.npy")
LLMS = ["qwen3", "qwen25", "smollm2"]
SLUGS = [l.split(".", 1)[1].strip() for l in open("/cache/wangky/tokenizer.txt") if l.strip()]
GT = json.load(open("/home/ma-user/work_space/VTB/results/ground_truth.json"))
K = 200


def gt(llm, bench=None):
    e = GT["encoders"]
    if bench is None:
        return np.array([e[s]["llms"][llm]["average"] for s in SLUGS])
    return np.array([e[s]["llms"][llm]["scores"][bench] for s in SLUGS])


def rank(a):
    a = np.asarray(a, float); o = a.argsort(kind="mergesort"); r = np.empty(len(a)); r[o] = np.arange(len(a))
    s = a[o]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
        if j > i: r[o[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def pearson(x, y):
    x = np.asarray(x, float) - np.mean(x); y = np.asarray(y, float) - np.mean(y)
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def spearman(x, y): return pearson(rank(x), rank(y))


def cos_sim(x):
    s = (x @ x.T).astype(np.float32); np.fill_diagonal(s, -np.inf); return s


def whiten(x):
    z, _ = preprocess(x, "pca_whiten_l2", max_components=256, whitening_eps=0.0, random_state=42)
    return z


def l2(x):
    x = np.asarray(x, np.float32); return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def topk_sets(sim, k=K):
    return np.argpartition(-sim, k, axis=1)[:, :k]


def binary_overlap(vis_sim, txt_sets, k=K):
    v = topk_sets(vis_sim, k)
    n = v.shape[0]
    return float(np.mean([len(np.intersect1d(v[i], txt_sets[i], assume_unique=True)) / k
                          for i in range(n)]))


def weighted_score(vis_sim, txt_dense):
    g, _ = neighbor_graph(vis_sim, K)
    return overlap_score(distribution(g, 0.25, 0.0), txt_dense)


def text_side(llm):
    raw = np.load(f"/cache/ravel_scaling_n5000_seed42/raw/text/text_{llm}_n5000.npy")[IDX]
    s_raw = cos_sim(l2(raw)); s_w = cos_sim(whiten(raw))
    g, _ = neighbor_graph(s_w, K)
    dense_w = dense_distribution(distribution(g, 0.25, 0.0), len(IDX))
    return {"raw": raw, "sets_raw": topk_sets(s_raw), "sets_w": topk_sets(s_w), "dense_w": dense_w}


def ladder():
    txt = {l: text_side(l) for l in tqdm(LLMS, desc="text views")}
    rows = {}
    for s in tqdm(SLUGS, desc="encoders"):
        z = np.load(M / "cache" / f"{s}.npz")
        pooled = z["pooled_raw"].astype(np.float32)
        s_pr = cos_sim(l2(pooled)); s_pw = cos_sim(whiten(pooled))
        s_patch_w = z["sim_patch_w"].astype(np.float32); np.fill_diagonal(s_patch_w, -np.inf)
        s_patch_r = z["sim_patch_r"].astype(np.float32); np.fill_diagonal(s_patch_r, -np.inf)
        r = {}
        for l in LLMS:
            t = txt[l]
            r[l] = {
                "V0_mutualnn": binary_overlap(s_pr, t["sets_raw"]),
                "V1_whiten": binary_overlap(s_pw, t["sets_w"]),
                "V2_patch": binary_overlap(s_patch_w, t["sets_w"]),
                "V3_ravel": weighted_score(s_patch_w, t["dense_w"]),
                "X_patch_nowhiten": binary_overlap(s_patch_r, t["sets_raw"]),
                "X_whiten_weighted": weighted_score(s_pw, t["dense_w"]),
            }
        rows[s] = {"scores": r, "n_tokens": int(z["n_tokens"]), "dim": int(z["dim"])}
    json.dump(rows, open(M / "ladder_scores.json", "w"), indent=1)
    return rows


if __name__ == "__main__":
    rows = ladder()
    variants = list(next(iter(rows.values()))["scores"]["qwen3"])
    print(f"\n{'variant':20s}" + "".join(f"{l:>10s}" for l in LLMS) + f"{'avg':>8s}")
    for v in variants:
        rhos = [spearman([rows[s]["scores"][l][v] for s in SLUGS], gt(l)) for l in LLMS]
        print(f"{v:20s}" + "".join(f"{x:10.3f}" for x in rhos) + f"{np.mean(rhos):8.3f}")
