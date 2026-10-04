#!/usr/bin/env python3
"""Build the COCO-2K image-text split used by the alignment probe.

Source: MSCOCO Karpathy test (5000 images x 5 captions) already staged at
/cache/data/instructions/test/MSCOCO_KARPATHY_TEST.tsv.

Sample 2000 images with a fixed seed, then split 1600 train / 400 test.
Caption 0 of each image is the "primary" caption (the 1-to-1 pair); all five
are kept so the standard 5-caption COCO retrieval protocol can also be scored.
"""

from __future__ import annotations

import argparse
import json
import random

from tqdm import tqdm

import vision_encoder_eval.workers.alignment.common as common
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=common.COCO_TSV)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n-train", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(common.COCO_JSON))
    args = ap.parse_args()

    common.ensure_dirs()
    print(f"reading {args.tsv} …", flush=True)
    rows = common.read_karpathy_tsv(args.tsv)
    print(f"  {len(rows)} usable image/caption groups", flush=True)
    if len(rows) < args.n:
        raise SystemExit(f"only {len(rows)} usable rows, need {args.n}")

    rng = random.Random(args.seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    picked = [rows[i] for i in tqdm(idx[: args.n], desc="sampling COCO-2K", unit="img")]

    n_test = args.n - args.n_train
    samples = []
    for i, r in enumerate(picked):
        samples.append(
            {
                "idx": i,
                "coco_id": r["id"],
                "image_path": r["image_path"],
                "captions": r["captions"],
                "split": "train" if i < args.n_train else "test",
            }
        )

    payload = {
        "source": args.tsv,
        "seed": args.seed,
        "n": args.n,
        "n_train": args.n_train,
        "n_test": n_test,
        "captions_per_image": 5,
        "samples": samples,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}: {args.n_train} train / {n_test} test", flush=True)


if __name__ == "__main__":
    main()
