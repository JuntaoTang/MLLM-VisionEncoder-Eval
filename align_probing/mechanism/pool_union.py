"""Mean-pooled MLLM patch tokens on the seed-43 formal sample (same images as the official ladder)."""
import numpy as np, sys
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import analyze as A

C = Path("/cache/ravel_multiseed/union_43_44")
CUSTOM = {"webssl_dino1b_full2b_224", "dinov3_vitl16", "ijepa_vith14", "raev2_dinov3l_k7"}
OUT = Path("/cache/wangky/align_probe/mechanism/pooled_seed{}")


def one(args):
    s, seed = args
    out = Path(str(OUT).format(seed)) / f"{s}.npy"
    if out.exists(): return s, "skip"
    pos = np.load(f"/cache/ravel_multiseed/union_positions_seed{seed}.npy")
    cands = [C / d / f for d in ("patches", "custom/patches") for f in (f"{s}_patch.npy", f"{s}_patch_n3222.npy")]
    p = next((c for c in cands if c.exists()), None)
    if p is None: return s, "MISSING"
    raw = np.load(p, mmap_mode="r")
    order = np.argsort(pos)                                   # sequential read, then restore order
    pooled = np.empty((len(pos), raw.shape[-1]), np.float32)
    for i in range(0, len(pos), 64):
        sl = order[i:i + 64]
        pooled[sl] = np.asarray(raw[pos[sl]], np.float32).mean(1)
    out.parent.mkdir(parents=True, exist_ok=True); np.save(out, pooled)
    return s, raw.shape


if __name__ == "__main__":
    seed = int(sys.argv[1])
    with Pool(12) as pool:
        for s, info in tqdm(pool.imap_unordered(one, [(s, seed) for s in A.SLUGS]), total=70, desc=f"pool seed{seed}"):
            tqdm.write(f"{s} {info}")
