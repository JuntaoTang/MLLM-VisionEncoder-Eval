#!/usr/bin/env python3
"""Correlate the COCO-2K alignment-probe score with the main MLLM benchmark table.

The main table (70 tokenizers x 3 LLMs) is transcribed in
data/main_table_image.csv; each row is resolved to a tokenizer slug by matching
its values against results/main_table_scores.json of the OCR experiment, which
is verified to be a bijection onto the 70 slugs.
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from tqdm import tqdm

import common

LLMS = ["qwen3", "qwen25", "smollm2"]
MAIN_JSON = "/cache/wangky/ocr_exp/results/main_table_scores.json"
IMAGE_CSV = common.DATA_DIR / "main_table_image.csv"


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), so Spearman is correct with duplicates."""
    a = np.asarray(a, float)
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average tied ranks
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.arange(i, j + 1).mean()
        i = j + 1
    return ranks


def pearson(x, y) -> float:
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d else float("nan")


def spearman(x, y) -> float:
    return pearson(_rank(x), _rank(y))


def kendall(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    c = d = 0
    for i in range(n):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        s = np.sign(dx) * np.sign(dy)
        c += int((s > 0).sum())
        d += int((s < 0).sum())
    return (c - d) / (c + d) if c + d else float("nan")


def perm_p(x, y, stat=spearman, n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided permutation p-value (no scipy on this box)."""
    rng = np.random.default_rng(seed)
    obs = abs(stat(x, y))
    y = np.asarray(y, float)
    hits = sum(abs(stat(x, rng.permutation(y))) >= obs for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1)


def resolve_rows() -> list[dict]:
    """Attach a slug to every transcribed main-table row."""
    main = json.load(open(MAIN_JSON, encoding="utf-8"))
    slugs = common.read_tokenizer_list()
    jv = {}
    for llm in LLMS:
        for s in slugs:
            k = f"{llm}__{s}"
            if k in main:
                jv[(llm, s)] = round(float(main[k]["average"]), 2)

    rows = list(csv.DictReader(open(IMAGE_CSV, encoding="utf-8")))
    out = []
    for r in rows:
        cands = []
        for s in slugs:
            known = [(llm, jv[(llm, s)]) for llm in LLMS if (llm, s) in jv]
            if known and all(abs(v - float(r[llm])) < 0.011 for llm, v in known):
                cands.append(s)
        if len(cands) != 1:
            raise SystemExit(f"row {r['rank']} ({r['name']}) resolved to {cands}")
        out.append({**r, "slug": cands[0]})
    if len({r["slug"] for r in out}) != len(slugs):
        raise SystemExit("main-table rows are not a bijection onto the 70 slugs")
    return out


def probe_scores(protocol: str, tag: str = "") -> dict[tuple[str, str], float]:
    key = "score" if protocol == "1cap" else "score_5cap"
    root = (common.RESULTS_DIR / "sweeps" / tag) if tag else (common.RESULTS_DIR / "align")
    out = {}
    for llm in LLMS:
        for p in (root / llm).glob("*.json"):
            with open(p, encoding="utf-8") as f:
                out[(llm, p.stem)] = float(json.load(f)[key])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="1cap", choices=["1cap", "5cap"])
    ap.add_argument("--tag", default="", help="read results/sweeps/<tag>/ instead")
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = resolve_rows()
    probe = probe_scores(args.protocol, args.tag)
    by_slug = {r["slug"]: r for r in rows}

    report: dict = {"protocol": args.protocol, "tag": args.tag,
                    "n_tokenizers": len(rows), "per_llm": {}}
    print(f"=== alignment probe ({args.protocol} mean recall"
          f"{', ' + args.tag if args.tag else ''}) vs main MLLM table ===")
    have = {s for (_, s) in probe}
    short = [r["slug"] for r in rows if r["slug"] not in have]
    if short:
        raise SystemExit(f"missing probe results for {len(short)} slugs: {short[:8]}")
    print(f"n = {len(rows)} tokenizers\n")
    print(f"{'LLM':10s} {'Spearman':>9s} {'p(perm)':>9s} {'Pearson':>9s} {'Kendall':>9s}")
    for llm in tqdm(LLMS, desc="correlating", unit="llm", leave=False):
        x = np.array([probe[(llm, s)] for s in by_slug])
        y = np.array([float(by_slug[s][llm]) for s in by_slug])
        rho, r, tau = spearman(x, y), pearson(x, y), kendall(x, y)
        p = perm_p(x, y, spearman, args.n_perm)
        report["per_llm"][llm] = {"spearman": rho, "p_perm": p, "pearson": r,
                                  "kendall": tau, "n": int(len(x))}
        print(f"{llm:10s} {rho:9.3f} {p:9.4f} {r:9.3f} {tau:9.3f}")

    # LLM-averaged columns
    xm = np.array([np.mean([probe[(l, s)] for l in LLMS]) for s in by_slug])
    ym = np.array([float(by_slug[s]["mean"]) for s in by_slug])
    rho, r, tau = spearman(xm, ym), pearson(xm, ym), kendall(xm, ym)
    p = perm_p(xm, ym, spearman, args.n_perm)
    report["mean"] = {"spearman": rho, "p_perm": p, "pearson": r, "kendall": tau}
    print(f"{'mean':10s} {rho:9.3f} {p:9.4f} {r:9.3f} {tau:9.3f}")

    # within-type, to check the correlation is not just language-supervised vs not
    print("\n--- within tokenizer type (LLM-averaged) ---")
    report["by_type"] = {}
    types = sorted({r["type"] for r in rows})
    for t in types:
        keep = [s for s in by_slug if by_slug[s]["type"] == t]
        if len(keep) < 4:
            continue
        x = np.array([np.mean([probe[(l, s)] for l in LLMS]) for s in keep])
        y = np.array([float(by_slug[s]["mean"]) for s in keep])
        rho = spearman(x, y)
        p = perm_p(x, y, spearman, args.n_perm)
        report["by_type"][t] = {"n": len(keep), "spearman": rho, "p_perm": p}
        print(f"{t:10s} n={len(keep):3d}  Spearman={rho:6.3f}  p={p:.4f}")

    # cross-LLM: does the probe of one LLM predict the table of another?
    print("\n--- cross-LLM Spearman (rows = probe LLM, cols = table LLM) ---")
    print(f"{'':10s}" + "".join(f"{l:>10s}" for l in LLMS))
    report["cross"] = {}
    for lp in LLMS:
        x = np.array([probe[(lp, s)] for s in by_slug])
        cells = []
        for lt in LLMS:
            y = np.array([float(by_slug[s][lt]) for s in by_slug])
            cells.append(spearman(x, y))
        report["cross"][lp] = dict(zip(LLMS, cells))
        print(f"{lp:10s}" + "".join(f"{c:10.3f}" for c in cells))

    stem = f"correlation_{args.protocol}"
    base = (common.RESULTS_DIR / "sweeps" / args.tag) if args.tag else common.RESULTS_DIR
    out = args.out or str(base / f"{stem}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # per-tokenizer table for eyeballing outliers
    csv_out = str(base / f"{stem}_rows.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["slug", "name", "type", "main_rank"]
                   + [f"probe_{l}" for l in LLMS] + [f"main_{l}" for l in LLMS]
                   + ["probe_mean", "main_mean"])
        for s, r in sorted(by_slug.items(), key=lambda kv: int(kv[1]["rank"])):
            pv = [probe[(l, s)] for l in LLMS]
            mv = [float(r[l]) for l in LLMS]
            w.writerow([s, r["name"], r["type"], r["rank"]]
                       + [f"{v:.2f}" for v in pv] + [f"{v:.2f}" for v in mv]
                       + [f"{np.mean(pv):.2f}", r["mean"]])
    print(f"\nwrote:\n  {out}\n  {csv_out}")


if __name__ == "__main__":
    main()
