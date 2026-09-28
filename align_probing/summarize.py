#!/usr/bin/env python3
"""Aggregate the 70 x 3 alignment-probing results into CSV + markdown tables."""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from tqdm import tqdm

import common

LLM_ORDER = ["qwen25", "qwen3", "smollm2"]


def align_dir(tag: str):
    return (common.RESULTS_DIR / "sweeps" / tag) if tag else (common.RESULTS_DIR / "align")


def collect(protocol: str, tag: str = "") -> tuple[list[dict], dict]:
    slugs = common.read_tokenizer_list()
    rows, wide = [], {}
    for slug in tqdm(slugs, desc="collecting", unit="tok"):
        wide[slug] = {}
        for llm in LLM_ORDER:
            p = align_dir(tag) / llm / f"{slug}.json"
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            t = r["test"]
            row = {
                "slug": slug,
                "llm": llm,
                "vision_dim": r["vision_dim"],
                "text_dim": r["text_dim"],
                "score": r[f"score{'' if protocol == '1cap' else '_5cap'}"],
                "mean_recall_1cap": t["mean_recall_1cap"],
                "mean_recall_5cap": t["mean_recall_5cap"],
                "val_mean_recall": r["selection"]["val_mean_recall"],
                "best_step": r["selection"]["best_step"],
                "wd": r["selection"]["wd"],
                "lr": r["selection"]["lr"],
                "seconds": r.get("seconds"),
            }
            for direction in ("i2t", "t2i"):
                for k in (1, 5, 10):
                    row[f"{direction}_R@{k}"] = t[f"{direction}_{protocol}"][f"R@{k}"]
            rows.append(row)
            wide[slug][llm] = row["score"]
    return rows, wide


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="1cap", choices=["1cap", "5cap"],
                    help="1cap = 400x400 pair matrix; 5cap = standard COCO 5-caption")
    ap.add_argument("--tag", default="", help="read results/sweeps/<tag>/ instead")
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()

    common.ensure_dirs()
    rows, wide = collect(args.protocol, args.tag)
    if not rows:
        raise SystemExit("no results found under results/align/")

    base = align_dir(args.tag) if args.tag else common.RESULTS_DIR
    base.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or str(base / f"summary_{args.protocol}")
    long_csv = f"{prefix}_long.csv"
    with open(long_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    wide_csv = f"{prefix}_wide.csv"
    with open(wide_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tokenizer"] + LLM_ORDER + ["mean"])
        for slug, d in wide.items():
            vals = [d.get(l) for l in LLM_ORDER]
            got = [v for v in vals if v is not None]
            w.writerow([slug] + [f"{v:.2f}" if v is not None else "" for v in vals]
                       + [f"{np.mean(got):.2f}" if got else ""])

    md = f"{prefix}.md"
    lines = [f"# Alignment probing on COCO-2K "
             f"(protocol: {args.protocol}{', ' + args.tag if args.tag else ''})", "",
             "Score = mean recall over i2t/t2i R@1/5/10 on the 400-image test split.", "",
             "| tokenizer | " + " | ".join(LLM_ORDER) + " | mean |",
             "|---|" + "---|" * (len(LLM_ORDER) + 1)]
    ranked = sorted(
        wide.items(),
        key=lambda kv: -np.mean([v for v in kv[1].values()]) if kv[1] else 1e9,
    )
    for slug, d in ranked:
        vals = [d.get(l) for l in LLM_ORDER]
        got = [v for v in vals if v is not None]
        cells = " | ".join(f"{v:.2f}" if v is not None else "-" for v in vals)
        mean = f"{np.mean(got):.2f}" if got else "-"
        lines.append(f"| {slug} | {cells} | {mean} |")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    done = len(rows)
    print(f"{done}/{len(wide) * len(LLM_ORDER)} combos scored")
    for llm in LLM_ORDER:
        s = [r["score"] for r in rows if r["llm"] == llm]
        if s:
            print(f"  {llm:9s} n={len(s):3d}  mean={np.mean(s):.2f}  max={np.max(s):.2f}")
    print(f"\nwrote:\n  {long_csv}\n  {wide_csv}\n  {md}")


if __name__ == "__main__":
    main()
