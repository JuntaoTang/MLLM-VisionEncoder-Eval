import numpy as np, json, sys
from tqdm import tqdm
import analyze as A, decompose as D
S, LL = A.SLUGS, A.LLMS
RS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]


def eig(x):
    xc = x - x.mean(0); ev, U = np.linalg.eigh(np.cov(xc.T)); o = np.argsort(ev)[::-1]
    return xc, ev[o].clip(0), U[:, o]


def z_at(xc, ev, U, r, whiten=True):
    r = min(r, int((ev > 1e-10 * ev[0]).sum()))
    z = xc @ U[:, :r]
    if whiten: z = z / np.sqrt(ev[:r] + 1e-5)
    return (z / np.linalg.norm(z, axis=1, keepdims=True).clip(1e-12)).astype(np.float32)


seed = int(sys.argv[1])
v, t = D.feats(seed)
T = {l: eig(t[l]) for l in LL}
out = {w: {r: {l: [] for l in LL} for r in RS} for w in ("whiten", "pca")}
for s in tqdm(S, desc=f"seed{seed}"):
    xc, ev, U = eig(v[s])
    for w in out:
        for r in RS:
            vk = D.knn(z_at(xc, ev, U, r, w == "whiten"))
            for l in LL:
                tk = D.knn(z_at(*T[l], r, w == "whiten"))
                out[w][r][l].append(D.overlap(vk, tk))
json.dump({w: {str(r): out[w][r] for r in RS} for w in out}, open(f"rsweep_seed{seed}.json", "w"))
for w in out:
    print(w)
    for r in RS:
        rh = [A.spearman(out[w][r][l], A.gt(l)) for l in LL]
        print(f"  r={r:5d} " + " ".join(f"{x:.3f}" for x in rh) + f"  avg {np.mean(rh):.3f}")
