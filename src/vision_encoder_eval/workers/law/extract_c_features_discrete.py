#!/usr/bin/env python3
"""Extract discrete-tokenizer patch features on SPair-71k for C-score."""

from __future__ import annotations

from vision_encoder_eval.core.runtime import asset_path

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_encoder_eval.workers.law.common import FEAT_DIR, RESULTS_DIR, SPAIR_DIR, atomic_json, tokens_to_grid  # noqa: E402
from vision_encoder_eval.workers.law.extract_c_features import _save_grid_slice, list_jpegs  # noqa: E402

VTB = Path(asset_path('mllm', ''))
TOK_YAML = VTB / "configs" / "discrete" / "tokenizer"
DISCRETE_IDS = ("toklip_l_384", "toklip_s_256", "unitok_attn", "vilau_256", "uniar_bsq")


def build_tokenizer(vision_id: str, device: str):
    sys.path.insert(0, str(VTB))
    import yaml
    from vision_encoder_eval.mllm.discrete.model.tokenizers.factory import build_visual_tokenizer

    path = TOK_YAML / f"{vision_id}.yaml"
    data = yaml.safe_load(path.read_text()) or {}
    tok_cfg = data.get("tokenizer") or data
    cfg = {"tokenizer": tok_cfg, "arch": {}}
    if tok_cfg.get("type") == "unitok":
        cfg["arch"] = {"vis_mode": "unitok"}
    tok = build_visual_tokenizer(cfg)
    tok = tok.to(device).eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    return tok


def encode_grid(tok, pixel: torch.Tensor) -> torch.Tensor:
    feat = None
    if hasattr(tok, "encode_post_quant_features"):
        feat = tok.encode_post_quant_features(pixel)
    elif hasattr(tok, "encode"):
        cand = tok.encode(pixel)
        if cand.dtype not in (torch.int32, torch.int64, torch.long):
            feat = cand
    if feat is None and hasattr(tok, "encode_quant_features"):
        feat = tok.encode_quant_features(pixel)
    if feat is None:
        raise RuntimeError(f"no dense encode path for {type(tok)}")
    if feat.dim() == 3:
        return tokens_to_grid(feat.float())
    if feat.dim() == 4:
        return tokens_to_grid(feat.float())
    raise RuntimeError(f"bad feat shape {tuple(feat.shape)}")


@torch.no_grad()
def extract_one(vision_id: str, images: list[Path], device: str, batch_size: int = 8) -> dict:
    out_root = FEAT_DIR / vision_id
    remaining = []
    for img_path in images:
        rel = img_path.relative_to(SPAIR_DIR / "JPEGImages")
        dest = out_root / rel.parent / f"{rel.stem}.pt"
        if dest.is_file():
            continue
        remaining.append((img_path, dest))
    if not remaining:
        return {"vision_id": vision_id, "status": "skip", "n": 0}

    tok = build_tokenizer(vision_id, device)
    size = int(tok.image_size)
    tfm = T.Compose([T.Resize((size, size)), T.ToTensor()])
    n_ok = 0
    grid_shape = None
    t0 = time.time()
    bs = max(1, int(batch_size))
    for start in tqdm(range(0, len(remaining), bs), desc=vision_id):
        chunk = remaining[start : start + bs]
        try:
            imgs = torch.stack([tfm(Image.open(p).convert("RGB")) for p, _ in chunk])
            imgs = imgs.to(device=device, dtype=torch.bfloat16)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                grid = encode_grid(tok, imgs)
            grid_shape = list(grid[0].shape)
            for j, (_, dest) in enumerate(chunk):
                _save_grid_slice(grid[j], dest)
            n_ok += len(chunk)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            for img_path, dest in chunk:
                img = tfm(Image.open(img_path).convert("RGB")).unsqueeze(0)
                img = img.to(device=device, dtype=torch.bfloat16)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    grid = encode_grid(tok, img)
                grid_shape = list(grid[0].shape)
                _save_grid_slice(grid[0], dest)
                n_ok += 1
    meta = {
        "vision_id": vision_id,
        "n_saved": n_ok,
        "n_total_images": len(images),
        "grid_shape": grid_shape,
        "seconds": round(time.time() - t0, 1),
        "family": "discrete",
    }
    from vision_encoder_eval.workers.law.common import atomic_json as dump

    dump(out_root / "meta.json", meta)
    del tok
    torch.cuda.empty_cache()
    return {"vision_id": vision_id, "status": "ok", "n": n_ok, "grid_shape": grid_shape}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-id", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    os.environ.setdefault("VTB_ROOT", str(VTB))
    jpeg_root = SPAIR_DIR / "JPEGImages"
    images = list_jpegs(jpeg_root)
    vids = [args.vision_id] if args.vision_id else list(DISCRETE_IDS)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log = []
    for vid in vids:
        print(f"[c-disc] {vid} device={device} n_img={len(images)}", flush=True)
        rec = extract_one(vid, images, device, batch_size=args.batch_size)
        print(rec, flush=True)
        log.append(rec)
    atomic_json(RESULTS_DIR / f"c_extract_discrete_{vids[0]}.json", log)


if __name__ == "__main__":
    main()
