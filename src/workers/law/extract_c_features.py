#!/usr/bin/env python3
"""Extract VTB vision-encoder patch features on SPair-71k JPEGImages."""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_encoder_eval.workers.law.common import (  # noqa: E402
    FEAT_DIR,
    RESULTS_DIR,
    SPAIR_DIR,
    apply_vision_env,
    atomic_json,
    build_vision_tower,
    ensure_vtb_path,
    inventory_models,
    load_vision_cfg,
    shard_list,
    tokens_to_grid,
    unique_vision_ids,
)


def list_jpegs(root: Path) -> list[Path]:
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            out.append(p)
    return out


def _save_grid_slice(grid, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(grid.cpu().contiguous().half(), dest)


@torch.no_grad()
def extract_one(
    vision_id: str,
    images: list[Path],
    device: str,
    skip_existing: bool,
    batch_size: int = 32,
) -> dict:
    out_root = FEAT_DIR / vision_id
    meta_path = out_root / "meta.json"
    remaining = []
    for img_path in images:
        rel = img_path.relative_to(SPAIR_DIR / "JPEGImages")
        dest = out_root / rel.parent / f"{rel.stem}.pt"
        if skip_existing and dest.is_file():
            continue
        remaining.append((img_path, dest))
    if not remaining and meta_path.is_file():
        return {"vision_id": vision_id, "status": "skip", "n": 0}

    vision = load_vision_cfg(vision_id)
    apply_vision_env(vision)
    tower = build_vision_tower(vision, device)
    processor = tower.image_processor

    n_ok = 0
    grid_shape = None
    t0 = time.time()
    bs = max(1, int(batch_size))
    for start in tqdm(range(0, len(remaining), bs), desc=vision_id, leave=False):
        chunk = remaining[start : start + bs]
        try:
            imgs = [Image.open(p).convert("RGB") for p, _ in chunk]
            pixel = processor(images=imgs, return_tensors="pt")["pixel_values"]
            pixel = pixel.to(device=device, dtype=torch.bfloat16)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                feat = tower(pixel)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            grid = tokens_to_grid(feat.float())
            grid_shape = list(grid[0].shape)
            for j, (_, dest) in enumerate(chunk):
                _save_grid_slice(grid[j], dest)
            n_ok += len(chunk)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            for img_path, dest in chunk:
                img = Image.open(img_path).convert("RGB")
                pixel = processor(images=img, return_tensors="pt")["pixel_values"]
                pixel = pixel.to(device=device, dtype=torch.bfloat16)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    feat = tower(pixel)
                if isinstance(feat, (tuple, list)):
                    feat = feat[0]
                grid = tokens_to_grid(feat.float())
                grid_shape = list(grid[0].shape)
                _save_grid_slice(grid[0], dest)
                n_ok += 1

    meta = {
        "vision_id": vision_id,
        "n_saved": n_ok,
        "n_total_images": len(images),
        "grid_shape": grid_shape,
        "seconds": round(time.time() - t0, 1),
        "type": vision.get("type"),
        "image_size": vision.get("image_size"),
        "select_layer": vision.get("select_layer"),
    }
    if meta_path.is_file() and n_ok == 0:
        pass
    else:
        prev = {}
        if meta_path.is_file():
            import json

            prev = json.loads(meta_path.read_text())
        if prev.get("grid_shape") and not grid_shape:
            meta["grid_shape"] = prev["grid_shape"]
        atomic_json(meta_path, meta)
    del tower
    torch.cuda.empty_cache()
    return {"vision_id": vision_id, "status": "ok", "n": n_ok, "grid_shape": grid_shape}


def _c_score_done() -> set[str]:
    import json

    done = set()
    for p in RESULTS_DIR.glob("c_score_shard*.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and v.get("c_score") is not None:
                    done.add(k)
    return done


def features_complete(vision_id: str, n_images: int) -> bool:
    meta_path = FEAT_DIR / vision_id / "meta.json"
    if not meta_path.is_file():
        return False
    n_pt = sum(1 for _ in (FEAT_DIR / vision_id).rglob("*.pt"))
    return n_pt >= max(1, int(0.95 * n_images))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--vision-id", type=str, default="")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--missing-only", action="store_true")
    args = parser.parse_args()

    ensure_vtb_path()
    jpeg_root = SPAIR_DIR / "JPEGImages"
    if not jpeg_root.is_dir():
        raise FileNotFoundError(f"SPair images not found: {jpeg_root}")
    images = list_jpegs(jpeg_root)
    if args.max_images:
        images = images[: args.max_images]
    print(f"[extract] {len(images)} images under {jpeg_root}", flush=True)

    if args.vision_id:
        vids = [args.vision_id]
    else:
        rows = inventory_models()
        vids = unique_vision_ids(rows)
        if args.missing_only:
            scored = _c_score_done()
            vids = [v for v in vids if v not in scored and not features_complete(v, len(images))]
        vids = shard_list(vids, args.shard, args.nshards)
    print(
        f"[extract] shard {args.shard}/{args.nshards}: {len(vids)} encoders missing_only={args.missing_only}",
        flush=True,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log = []
    for vid in vids:
        try:
            rec = extract_one(
                vid, images, device, skip_existing=not args.no_skip, batch_size=args.batch_size
            )
            print(rec, flush=True)
            log.append(rec)
        except Exception as exc:
            traceback.print_exc()
            log.append({"vision_id": vid, "status": "error", "error": str(exc)})
            torch.cuda.empty_cache()
    atomic_json(RESULTS_DIR / f"c_extract_shard{args.shard}.json", log)


if __name__ == "__main__":
    main()
