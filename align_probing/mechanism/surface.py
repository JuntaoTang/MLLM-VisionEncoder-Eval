"""Disentangle feature surface (CLS vs MLLM patch tokens) from set matching (pooled vs Chamfer)."""
import numpy as np, json, sys
from tqdm import tqdm
import analyze as A, official as O_, decompose as D

S, LL = A.SLUGS, A.LLMS
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 43
v_cls, t = D.feats(seed)
v_pool = {s: np.load(f"pooled_seed{seed}/{s}.npy").astype(np.float64) for s in S}
tk = {m: {l: D.knn(D.proc(t[l], m)) for l in LL} for m in ("raw", "whiten")}
res = {k: {l: [] for l in LL} for k in ("cls_raw", "cls_w", "pool_raw", "pool_w")}
for s in tqdm(S, desc=f"seed{seed}"):
    for key, x, m in [("cls_raw", v_cls[s], "raw"), ("cls_w", v_cls[s], "whiten"),
                      ("pool_raw", v_pool[s], "raw"), ("pool_w", v_pool[s], "whiten")]:
        vk = D.knn(D.proc(x, m))
        for l in LL: res[key][l].append(D.overlap(vk, tk[m][l]))
off = O_.load()[seed]
res = {k: {l: np.array(v) for l, v in d.items()} for k, d in res.items()}
res["set_w (official S2)"] = off["S2_patch"]
res["RAVEL (official S3)"] = off["S3_ravel"]
json.dump({k: {l: list(map(float, v)) for l, v in d.items()} for k, d in res.items()},
          open(f"surface_seed{seed}.json", "w"))
gr = {l: dict(zip(S, 70 - A.rank(A.gt(l)))) for l in LL}
print(f"{'variant':22s}{'rho q3':>8s}{'q25':>7s}{'sm2':>7s}{'avg':>7s}{'top1':>7s}{'P@10':>7s}  eupe_vit_b rank(q3/q25/sm2)")
for k, d in res.items():
    rh = [A.spearman(d[l], A.gt(l)) for l in LL]
    t1 = np.mean([A.gt(l)[int(np.argmax(d[l]))] for l in LL])
    p10 = np.mean([len(set(np.argsort(-d[l])[:10]) & set(np.argsort(-A.gt(l))[:10])) / 10 for l in LL])
    er = [int(70 - A.rank(d[l])[S.index("eupe_vit_b")]) for l in LL]
    print(f"{k:22s}" + "".join(f"{x:8.3f}" if i == 0 else f"{x:7.3f}" for i, x in enumerate(rh))
          + f"{np.mean(rh):7.3f}{t1:7.2f}{p10:7.2f}  {er}")
