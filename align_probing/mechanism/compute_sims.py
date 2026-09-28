#!/usr/bin/env python3
"""Per-encoder similarity matrices for the component-mechanism study.

Uses RAVEL's own cached raw patch tokens (n=5000 COCO val2017) and RAVEL's own
whitening / Chamfer code, restricted to the formal n=2000 seed-42 indices, so
that the variants below differ ONLY in the component being toggled:
  pooled_raw   : mean-pooled patch tokens            [n, D]  (MutualNN input)
  sim_patch_w  : Chamfer over PCA-whitened patches   [n, n]  (RAVEL visual kernel)
  sim_patch_r  : Chamfer over raw (L2) patches       [n, n]  (patch-set w/o whitening)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/home/ma-user/work_space/RAVEL")
from src.patch_similarity import chamfer_similarity, whiten_patch_tokens  # noqa: E402

RAW = Path("/cache/ravel_scaling_n5000_seed42/raw/patches")
IDX = np.load("/home/ma-user/work_space/RAVEL/experiments/patch_ravel_scaling_n5000_seed42/"
              "inputs/indices_n2000_seed42.npy")
OUT = Path("/cache/wangky/align_probe/mechanism/cache")
SLUGS = [l.split(".", 1)[1].strip() for l in open("/cache/wangky/tokenizer.txt") if l.strip()]


def one(slug: str) -> None:
    out = OUT / f"{slug}.npz"
    if out.exists():
        return
    raw = np.load(RAW / f"{slug}_patch_n5000.npy", mmap_mode="r")
    sel = np.asarray(raw[IDX], dtype=np.float32)            # [2000, T, D]
    pooled = sel.mean(axis=1)
    sim_r = chamfer_similarity(sel, device="cuda")           # L2-normalised inside
    w, meta = whiten_patch_tokens(sel, max_components=256, whitening_eps=0.0, random_state=42)
    del sel
    sim_w = chamfer_similarity(w, device="cuda")
    del w
    np.savez(out, pooled_raw=pooled.astype(np.float32),
             sim_patch_w=sim_w.astype(np.float16), sim_patch_r=sim_r.astype(np.float16),
             n_tokens=raw.shape[1], dim=raw.shape[2], r_u=meta.get("r_u", -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()
    todo = SLUGS[a.shard::a.num_shards]
    for s in tqdm(todo, desc=f"shard{a.shard}", unit="enc"):
        t0 = time.time()
        try:
            one(s)
            tqdm.write(f"[ok] {s} {time.time()-t0:.0f}s")
        except Exception as e:  # keep going
            tqdm.write(f"[FAIL] {s}: {e}")


if __name__ == "__main__":
    main()
