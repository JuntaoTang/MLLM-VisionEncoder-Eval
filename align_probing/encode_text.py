#!/usr/bin/env python3
"""Pre-encode COCO-2K captions with a frozen LLM (SAIL stage-1, text side).

Writes cache/text/<llm>/emb_<pool>.npy of shape [2000, 5, D] (float32) plus a
meta.json. The LLM is frozen; only the alignment layer is trained later.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from tqdm import tqdm

import common

common.bootstrap()


@torch.no_grad()
def encode(llm_path: str, texts: list[str], device: str, batch: int, pool: str,
           max_length: int) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(
        llm_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()

    out = []
    for i in tqdm(range(0, len(texts), batch), desc="captions", unit="batch"):
        chunk = texts[i : i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = model(**enc, use_cache=False).last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        if pool == "mean":
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        elif pool == "last":
            last = enc["attention_mask"].sum(1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=device), last]
        else:
            raise ValueError(pool)
        out.append(pooled.cpu().numpy().astype(np.float32))

    del model
    torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True, choices=sorted(common.LLMS))
    ap.add_argument("--dataset", default="coco2k", choices=sorted(common.DATASETS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--pool", default="mean", choices=["mean", "last"])
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    common.ensure_dirs()
    data = common.load_dataset(args.dataset)
    samples = data["samples"]
    n_cap = data["captions_per_image"]

    out_dir = common.text_cache(args.dataset, args.llm)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = out_dir / f"emb_{args.pool}.npy"
    if out_npy.exists() and not args.overwrite:
        print(f"[skip] {out_npy} exists", flush=True)
        return

    flat = [c for s in samples for c in s["captions"][:n_cap]]
    print(f"[{args.dataset}] {args.llm}: encoding {len(flat)} captions "
          f"({len(samples)} images x {n_cap})", flush=True)

    emb = encode(common.LLMS[args.llm]["path"], flat, args.device, args.batch,
                 args.pool, args.max_length)
    emb = emb.reshape(len(samples), n_cap, -1)
    np.save(out_npy, emb)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "llm": args.llm,
                "dataset": args.dataset,
                "caption_fields": data.get("caption_fields"),
                "path": common.LLMS[args.llm]["path"],
                "pool": args.pool,
                "dim": int(emb.shape[-1]),
                "shape": list(emb.shape),
                "max_length": args.max_length,
            },
            f,
            indent=2,
        )
    print(f"saved {out_npy}  shape={emb.shape}", flush=True)


if __name__ == "__main__":
    main()
