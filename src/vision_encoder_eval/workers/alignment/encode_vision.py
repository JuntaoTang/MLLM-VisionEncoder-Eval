#!/usr/bin/env python3
"""Pre-encode COCO-2K images with a frozen visual tokenizer (SAIL stage-1,
vision side).

Continuous towers are built exactly as VTB's LLaVA pipeline builds them
(llava.model.multimodal_encoder.builder + VTB env plumbing); discrete
tokenizers go through VTB's discrete factory and expose the same features the
discrete adapter feeds to its projector. Visual tokens are mean-pooled into a
single vector per image.

Writes cache/vision/<slug>/emb.npy of shape [2000, Dv] (float32) + meta.json.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import vision_encoder_eval.workers.alignment.common as common
common.bootstrap()
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ContinuousImages(Dataset):
    """Images preprocessed by the tower's own HF/open_clip image processor."""

    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img = Image.open(self.samples[i]["image_path"]).convert("RGB")
        px = self.processor(images=img, return_tensors="pt")["pixel_values"]
        return px.squeeze(0), i


class DiscreteImages(Dataset):
    """Raw [0,1] tensors at the tokenizer's native resolution (VTB convention)."""

    def __init__(self, samples, image_size):
        import torchvision.transforms as T

        self.samples = samples
        self.transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img = Image.open(self.samples[i]["image_path"]).convert("RGB")
        return self.transform(img), i


def _pool(out: torch.Tensor) -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        out = out[0]
    out = out.float()
    if out.ndim == 3:          # [B, N, D] visual tokens
        return out.mean(dim=1)
    if out.ndim == 4:          # [B, C, H, W] conv feature map
        return out.flatten(2).mean(dim=2)
    return out                 # [B, D]


@torch.no_grad()
def _discrete_features(tok, vis_mode: str, pixels: torch.Tensor) -> torch.Tensor:
    """Same feature taps as src/discrete/model/discrete_adapter.py."""
    from vision_encoder_eval.mllm.discrete.model.discrete_adapter import POST_QUANT_VIS_MODES

    if vis_mode == "unitok_quant":
        return tok.encode_quant_features(pixels)
    if vis_mode == "unitok":
        return tok.encode(pixels)
    if vis_mode in POST_QUANT_VIS_MODES:
        return tok.encode_post_quant_features(pixels)
    if vis_mode == "qlip_pre_quant":
        return tok.encode_pre_quant_features(pixels)
    raise ValueError(f"unsupported discrete vis_mode {vis_mode!r}")


@torch.no_grad()
def run(slug: str, device: str, batch: int, workers: int,
        dataset: str = "coco2k", reuse_from: str = "") -> dict:
    data = common.load_dataset(dataset)
    samples = data["samples"]

    # Nested datasets (cc3m10k starts with the cc3m2k samples): reuse the cached
    # prefix and encode only the remainder.
    reuse = None
    if reuse_from:
        src = common.load_dataset(reuse_from)["samples"]
        emb = common.vision_cache(reuse_from, slug) / "emb.npy"
        keyof = lambda x: x.get("rel_path") or x["image_path"]  # noqa: E731
        if emb.exists() and len(src) < len(samples) and \
                [keyof(a) for a in src] == [keyof(b) for b in samples[:len(src)]]:
            reuse = np.load(emb)
            samples = samples[len(src):]
            print(f"  reusing {len(reuse)} cached embeddings from {reuse_from}, "
                  f"encoding {len(samples)} new", flush=True)
    mode, cfg_path = common.resolve_slug(slug)
    cfg = common.load_cfg(cfg_path)

    if mode == "continuous":
        vision = cfg["vision_encoder"]
        tower = common.build_continuous_tower(vision, device)
        ds = ContinuousImages(samples, tower.image_processor)
        param_dtype = next(tower.parameters()).dtype
        extractor = lambda px: tower(px)  # noqa: E731
        info = {"type": vision.get("type"), "select_layer": vision.get("select_layer"),
                "select_feature": vision.get("select_feature")}
    else:
        tok, vis_mode = common.build_discrete_tokenizer(cfg, device)
        ds = DiscreteImages(samples, int(tok.image_size))
        param_dtype = next(tok.parameters()).dtype
        extractor = lambda px: _discrete_features(tok, vis_mode, px)  # noqa: E731
        info = {"type": cfg["tokenizer"].get("type"), "vis_mode": vis_mode,
                "num_image_tokens": int(tok.num_image_tokens)}

    def forward(px: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=param_dtype == torch.float32):
            out = extractor(px)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    feats: list[np.ndarray | None] = [None] * len(samples)
    n_tokens: int | None = None
    loader = DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True)
    bar = tqdm(total=len(samples), desc=f"{slug} [{mode}]", unit="img")
    micro = batch
    for pixels, idxs in loader:
        pixels = pixels.to(device, non_blocking=True)
        if param_dtype in (torch.float16, torch.bfloat16):
            pixels = pixels.to(param_dtype)
        while True:
            try:
                chunks = []
                for s0 in range(0, pixels.shape[0], micro):
                    out = forward(pixels[s0 : s0 + micro])
                    if out.ndim == 3 and n_tokens is None:
                        n_tokens = int(out.shape[1])
                    chunks.append(_pool(out).cpu())
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if micro <= 1:
                    raise
                micro = max(1, micro // 2)
                tqdm.write(f"  OOM -> retrying with micro-batch {micro}")
        pooled = torch.cat(chunks, 0).numpy().astype(np.float32)
        for j, k in enumerate(idxs.tolist()):
            feats[k] = pooled[j]
        bar.update(len(idxs))
    bar.close()

    emb = np.stack(feats, axis=0)
    if reuse is not None:
        emb = np.concatenate([reuse, emb], axis=0)
    out_dir = common.vision_cache(dataset, slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "emb.npy", emb)
    meta = {"slug": slug, "dataset": dataset, "mode": mode, "config": str(cfg_path),
            "dim": int(emb.shape[1]), "shape": list(emb.shape),
            "visual_tokens": n_tokens, **info}
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="coco2k", choices=sorted(common.DATASETS))
    ap.add_argument("--reuse-from", default="",
                    help="nested dataset whose cached prefix can be reused")
    ap.add_argument("--slugs", default="", help="comma list; empty = all 70")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    common.ensure_dirs()
    slugs = ([s.strip() for s in args.slugs.split(",") if s.strip()]
             if args.slugs else common.read_tokenizer_list())
    slugs = slugs[args.shard :: args.num_shards]
    print(f"[{args.dataset}] shard {args.shard}/{args.num_shards}: {len(slugs)} "
          f"tokenizers on {args.device}", flush=True)

    fail_log = (common.RESULTS_DIR /
                f"vision_failures_{args.dataset}_shard{args.shard}.json")
    failures = []
    outer = tqdm(slugs, desc=f"shard{args.shard} tokenizers", unit="tok", position=0)
    for slug in outer:
        outer.set_postfix_str(slug)
        out_npy = common.vision_cache(args.dataset, slug) / "emb.npy"
        if out_npy.exists() and not args.overwrite:
            tqdm.write(f"[skip] {slug} (cached)")
            continue
        t0 = time.time()
        try:
            meta = run(slug, args.device, args.batch, args.workers, args.dataset,
                       args.reuse_from)
            tqdm.write(f"[ok]   {slug}  D={meta['dim']} "
                       f"tokens={meta['visual_tokens']} ({time.time()-t0:.0f}s)")
        except Exception as exc:  # keep the sweep alive
            tqdm.write(f"[FAIL] {slug}: {exc}")
            traceback.print_exc()
            failures.append({"slug": slug, "error": str(exc),
                             "trace": traceback.format_exc()})
            with open(fail_log, "w", encoding="utf-8") as f:
                json.dump(failures, f, indent=2)
        finally:
            torch.cuda.empty_cache()

    if failures:
        print(f"\n{len(failures)} failures → {fail_log}", flush=True)


if __name__ == "__main__":
    main()
