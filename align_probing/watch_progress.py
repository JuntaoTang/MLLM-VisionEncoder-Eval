#!/usr/bin/env python3
"""Global tqdm bar over sharded workers (each shard also logs its own bar)."""

from __future__ import annotations

import argparse
import os
import time

from tqdm import tqdm

import common

LLMS = ["qwen25", "qwen3", "smollm2"]


def _alive(pids: list[int]) -> bool:
    for p in pids:
        try:
            os.kill(p, 0)
        except OSError:
            continue
        return True
    return False


def count(stage: str, tag: str = "", dataset: str = "coco2k") -> int:
    if stage == "vision":
        return sum(
            (common.vision_cache(dataset, s) / "emb.npy").exists()
            for s in common.read_tokenizer_list()
        )
    root = (common.RESULTS_DIR / "sweeps" / tag) if tag else (common.RESULTS_DIR / "align")
    return sum(
        (root / llm / f"{s}.json").exists()
        for llm in LLMS
        for s in common.read_tokenizer_list()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["vision", "align", "sweep"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--dataset", default="coco2k")
    ap.add_argument("--pids", default="")
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    pids = [int(p) for p in args.pids.split() if p.strip().isdigit()]
    n_tok = len(common.read_tokenizer_list())
    total = n_tok if args.stage == "vision" else n_tok * len(LLMS)
    unit = "tokenizer" if args.stage == "vision" else "combo"

    bar = tqdm(total=total, desc=f"{args.stage}{'/' + args.tag if args.tag else ''} "
                                 f"(all shards)", unit=unit)
    done = count(args.stage, args.tag, args.dataset)
    bar.update(done)
    while _alive(pids) if pids else done < total:
        time.sleep(args.interval)
        now = count(args.stage, args.tag, args.dataset)
        if now > done:
            bar.update(now - done)
            done = now
    now = count(args.stage, args.tag, args.dataset)
    if now > done:
        bar.update(now - done)
    bar.close()
    print(f"{args.stage}: {now}/{total} done", flush=True)


if __name__ == "__main__":
    main()
