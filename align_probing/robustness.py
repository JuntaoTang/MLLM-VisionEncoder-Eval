#!/usr/bin/env python3
"""Cross-variant robustness of the COCO-2K alignment probe.

Reads the baseline (results/align/) plus every results/sweeps/<tag>/ variant and
reports, per LLM:
  * Spearman against the main MLLM table, per variant
  * rank stability of the probe itself across variants
Absolute recall is NOT comparable across ratios (the retrieval gallery shrinks
as the train split grows), so everything here is rank-based; the fixed-ratio
re-partition variants (same gallery) additionally get an absolute-score spread.
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from tqdm import tqdm

import common
from correlate import LLMS, perm_p, pearson, resolve_rows, spearman, _rank


def load_variant(tag: str, protocol: str) -> dict[tuple[str, str], float]:
    key = "score" if protocol == "1cap" else "score_5cap"
    root = (common.RESULTS_DIR / "sweeps" / tag) if tag else (common.RESULTS_DIR / "align")
    out = {}
    for llm in LLMS:
        for p in (root / llm).glob("*.json"):
            with open(p, encoding="utf-8") as f:
                out[(llm, p.stem)] = float(json.load(f)[key])
    return out


def variant_meta(tag: str) -> dict:
    root = (common.RESULTS_DIR / "sweeps" / tag) if tag else (common.RESULTS_DIR / "align")
    p = next(iter((root / LLMS[0]).glob("*.json")))
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    s = d.get("split") or {"n_train": 1600, "split_seed": -1}
    return {"tag": tag or "base", "n_train": int(s["n_train"]),
            "split_seed": int(s["split_seed"]), "n_gallery": int(d["n_test"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="1cap", choices=["1cap", "5cap"])
    ap.add_argument("--n-perm", type=int, default=10000)
    args = ap.parse_args()

    rows = resolve_rows()
    by_slug = {r["slug"]: r for r in rows}
    slugs = list(by_slug)

    tags = [""] + sorted(d.name for d in (common.RESULTS_DIR / "sweeps").iterdir()
                         if d.is_dir())
    data, meta = {}, {}
    for t in tqdm(tags, desc="loading variants", unit="variant", leave=False):
        v = load_variant(t, args.protocol)
        miss = [s for s in slugs for l in LLMS if (l, s) not in v]
        if miss:
            print(f"[warn] {t or 'base'}: {len(miss)} missing combos, skipped")
            continue
        data[t or "base"] = v
        meta[t or "base"] = variant_meta(t)

    order = sorted(data, key=lambda k: (meta[k]["split_seed"] >= 0, meta[k]["n_train"],
                                        meta[k]["split_seed"]))

    # ---------------- 1. Spearman vs main table, per variant x LLM ------------
    print(f"\n=== Spearman(probe, main table), protocol={args.protocol} ===")
    print(f"{'variant':9s}{'train/test':>12s}" + "".join(f"{l:>10s}" for l in LLMS)
          + f"{'mean-col':>10s}")
    table = {}
    for t in order:
        m = meta[t]
        cells = []
        for llm in LLMS:
            x = np.array([data[t][(llm, s)] for s in slugs])
            y = np.array([float(by_slug[s][llm]) for s in slugs])
            cells.append(spearman(x, y))
        xm = np.array([np.mean([data[t][(l, s)] for l in LLMS]) for s in slugs])
        ym = np.array([float(by_slug[s]["mean"]) for s in slugs])
        cells.append(spearman(xm, ym))
        table[t] = cells
        print(f"{t:9s}{m['n_train']}/{m['n_gallery']:<7d}"
              + "".join(f"{c:10.3f}" for c in cells))

    arr = np.array([table[t] for t in order])
    print(f"{'':9s}{'':12s}" + "".join(f"{v:10.3f}" for v in arr.mean(0)) + "   <- mean")
    print(f"{'':9s}{'':12s}" + "".join(f"{v:10.3f}" for v in arr.std(0)) + "   <- sd")

    # ratio vs re-partition subsets
    ratio = [t for t in order if meta[t]["split_seed"] < 0]
    repart = [t for t in order if meta[t]["split_seed"] >= 0] + ["base"]
    for name, grp in (("ratio sweep", ratio), ("re-partition (1600/400)", repart)):
        a = np.array([table[t] for t in grp])
        print(f"  {name:26s} n={len(grp)}  "
              + "  ".join(f"{l}={a[:, i].mean():.3f}±{a[:, i].std():.3f}"
                          for i, l in enumerate(LLMS)))

    # ---------------- 2. probe-internal rank stability -----------------------
    print("\n=== probe rank stability across variants (LLM-averaged) ===")
    R = {t: _rank(np.array([np.mean([data[t][(l, s)] for l in LLMS]) for s in slugs]))
         for t in order}
    pair = [(a, b, pearson(R[a], R[b])) for i, a in enumerate(order) for b in order[i + 1:]]
    allrho = np.array([p[2] for p in pair])
    print(f"pairwise Spearman between variants: mean={allrho.mean():.3f} "
          f"min={allrho.min():.3f} (n_pairs={len(pair)})")
    worst = sorted(pair, key=lambda p: p[2])[:5]
    for a, b, r in worst:
        print(f"  lowest: {a:7s} vs {b:7s}  rho={r:.3f}")
    rr = np.array([pearson(R[a], R[b]) for i, a in enumerate(repart)
                   for b in repart[i + 1:]])
    print(f"re-partition only (same 1600/400): mean={rr.mean():.3f} min={rr.min():.3f}")

    # ---------------- 3. absolute-score spread, fixed ratio only -------------
    print("\n=== absolute score spread across the 1600/400 re-partitions ===")
    per = {llm: np.array([[data[t][(llm, s)] for s in slugs] for t in repart])
           for llm in LLMS}
    for llm in LLMS:
        sd = per[llm].std(0)
        print(f"{llm:9s} mean={per[llm].mean():.2f}  per-tokenizer sd: "
              f"median={np.median(sd):.2f} p90={np.percentile(sd, 90):.2f} "
              f"max={sd.max():.2f}")

    # ---------------- 4. write per-variant csv -------------------------------
    out = common.RESULTS_DIR / f"robustness_{args.protocol}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n_train", "n_gallery", "split_seed"]
                   + [f"spearman_{l}" for l in LLMS] + ["spearman_mean_col"])
        for t in order:
            m = meta[t]
            w.writerow([t, m["n_train"], m["n_gallery"], m["split_seed"]]
                       + [f"{c:.4f}" for c in table[t]])
    js = common.RESULTS_DIR / f"robustness_{args.protocol}.json"
    with open(js, "w", encoding="utf-8") as f:
        json.dump({"protocol": args.protocol, "meta": meta,
                   "spearman_vs_main": {t: dict(zip(LLMS + ["mean_col"], table[t]))
                                        for t in order},
                   "probe_rank_agreement": {"all_pairs_mean": float(allrho.mean()),
                                            "all_pairs_min": float(allrho.min()),
                                            "repartition_mean": float(rr.mean()),
                                            "repartition_min": float(rr.min())}},
                  f, indent=2)
    print(f"\nwrote:\n  {out}\n  {js}")


if __name__ == "__main__":
    main()
