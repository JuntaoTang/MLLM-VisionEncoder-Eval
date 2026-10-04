#!/usr/bin/env python3
"""Sample the CC3M-2K training set used by the SAIL-style alignment probe.

SAIL trains its alignment layer on DreamLIP's MLLM-recaptioned CC3M, pairing
each image with `raw_caption` and using `longSV_captions` as a second positive
(scripts/alignment_probing.sh). We keep exactly that, just at 2000 pairs.

Only part of CC3M is downloaded locally, so the pool is "rows of the DreamLIP
csv whose image is present and decodable". Every candidate is opened once so a
corrupt jpg can never silently enter the set.

Writes data/cc3m2k.json (the dataset) and data/cc3m2k_selected.csv (a
standalone, human-readable list of exactly which 2000 rows were used).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random

from PIL import Image, ImageFile
from tqdm import tqdm

import vision_encoder_eval.workers.alignment.common as common
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Degenerate images (a side of a few pixels) break HF channel inference, so they
# are excluded at build time rather than special-cased during encoding.
MIN_SIDE = 32
CAPTION_FIELDS = common.DATASETS["cc3m2k"]["captions"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=common.CC3M_CSV)
    ap.add_argument("--root", default=common.CC3M_ROOT)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scan-rows", type=int, default=500_000,
                    help="csv rows to scan for locally available images")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    common.ensure_dirs()
    csv.field_size_limit(10**7)

    print(f"scanning {args.csv} for locally downloaded images…", flush=True)
    pool = []
    with open(args.csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(tqdm(reader, total=args.scan_rows, desc="csv rows",
                                     unit="row")):
            if i >= args.scan_rows:
                break
            rel = row["Image Path"]
            if not os.path.exists(os.path.join(args.root, rel)):
                continue
            if any(not (row.get(c) or "").strip() for c in CAPTION_FIELDS):
                continue
            pool.append(row)
    print(f"  {len(pool)} candidate pairs with image + all captions present", flush=True)
    if len(pool) < args.n:
        raise SystemExit(f"only {len(pool)} candidates, need {args.n}")

    rng = random.Random(args.seed)
    order = list(range(len(pool)))
    rng.shuffle(order)

    samples, skipped = [], []
    bar = tqdm(total=args.n, desc="verifying + selecting", unit="pair")
    for j in order:
        if len(samples) >= args.n:
            break
        row = pool[j]
        rel = row["Image Path"]
        path = os.path.join(args.root, rel)
        try:                      # decode once: a corrupt jpg must not get in
            with Image.open(path) as im:
                im.convert("RGB")
                w, h = im.size
            if min(w, h) < MIN_SIDE:
                raise ValueError(f"degenerate size {w}x{h}")
        except Exception as exc:
            skipped.append({"image_path": rel, "error": str(exc)})
            continue
        samples.append({
            "idx": len(samples),
            "csv_row": j,
            "image_url": row["Image Url"],
            "image_path": path,
            "rel_path": rel,
            "captions": [row[c].strip() for c in CAPTION_FIELDS],
            "split": "train",
        })
        bar.update(1)
    bar.close()

    if len(samples) < args.n:
        raise SystemExit(f"only {len(samples)} decodable pairs found")

    out = args.out or str(common.DATASETS["cc3m2k"]["json"])
    payload = {"source": args.csv, "root": args.root, "seed": args.seed,
               "n": len(samples), "n_train": len(samples), "n_test": 0,
               "caption_fields": CAPTION_FIELDS,
               "captions_per_image": len(CAPTION_FIELDS),
               "scanned_rows": args.scan_rows, "pool_size": len(pool),
               "skipped_corrupt": skipped, "samples": samples}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    # standalone listing of exactly which pairs were used
    sel = out.replace(".json", "_selected.csv")
    with open(sel, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "csv_row", "rel_path", "image_url"] + CAPTION_FIELDS)
        for s in samples:
            w.writerow([s["idx"], s["csv_row"], s["rel_path"], s["image_url"]]
                       + s["captions"])

    print(f"\nwrote {out}  ({len(samples)} pairs, {len(skipped)} corrupt skipped)")
    print(f"wrote {sel}  (standalone list of the selected pairs)")


if __name__ == "__main__":
    main()
