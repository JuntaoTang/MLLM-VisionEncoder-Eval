#!/usr/bin/env python3
"""Zero-shot PCK@0.10 on SPair-71k test pairs (paper C-score)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ANNO_SIZE,
    FEAT_DIR,
    RESULTS_DIR,
    SOFT_EVAL_WINDOW,
    SPAIR_DIR,
    atomic_json,
    inventory_models,
    shard_list,
    unique_vision_ids,
)


def preprocess_kps_pad(kps: torch.Tensor, img_width: int, img_height: int, size: int):
    kps = kps.clone()
    scale = size / max(img_width, img_height)
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        offset_y = int((size - new_h) / 2)
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        offset_x = int((size - new_w) / 2)
        kps[:, 0] += offset_x
    kps *= kps[:, 2:3].clone()
    return kps, scale


def load_spair_test(path: Path, category: str, size: int = ANNO_SIZE, subsample: int = 0):
    pairs = sorted(glob(str(path / "PairAnnotation" / "test" / f"*:{category}.json")))
    if subsample and subsample > 0:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(pairs), size=min(subsample, len(pairs)), replace=False)
        pairs = [pairs[i] for i in sorted(idx)]
    files, kps, thresholds = [], [], []
    cat_annos = glob(str(path / "ImageAnnotation" / category / "*.json"))
    if not cat_annos:
        return [], torch.zeros(0, 1, 3), []
    with open(cat_annos[0]) as f:
        num_kps = len(json.load(f)["kps"])
    for pair in pairs:
        source_kps = torch.zeros(num_kps, 3)
        target_kps = torch.zeros(num_kps, 3)
        with open(pair) as f:
            data = json.load(f)
        source_fn = str(path / "JPEGImages" / category / data["src_imname"])
        target_fn = str(path / "JPEGImages" / category / data["trg_imname"])
        source_json = source_fn.replace("JPEGImages", "ImageAnnotation").replace(".jpg", ".json")
        target_json = target_fn.replace("JPEGImages", "ImageAnnotation").replace(".jpg", ".json")
        target_bbox = np.asarray(data["trg_bndbox"])
        with open(source_json) as f:
            kpts_src = json.load(f)["kps"]
        with open(target_json) as f:
            kpts_trg = json.load(f)["kps"]
        source_size = data["src_imsize"][:2]
        target_size = data["trg_imsize"][:2]
        for i in range(num_kps):
            point = kpts_src.get(str(i))
            if point is not None:
                source_kps[i, :2] = torch.tensor(point).float()
                source_kps[i, 2] = 1
            point = kpts_trg.get(str(i))
            if point is not None:
                target_kps[i, :2] = torch.tensor(point).float()
                target_kps[i, 2] = 1
        source_kps, _ = preprocess_kps_pad(source_kps, source_size[0], source_size[1], size)
        target_kps, trg_scale = preprocess_kps_pad(target_kps, target_size[0], target_size[1], size)
        thresholds.append(max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0]) * trg_scale)
        kps.extend([source_kps, target_kps])
        files.extend([source_fn, target_fn])
    if not kps:
        return [], torch.zeros(0, 1, 3), []
    kps = torch.stack(kps)
    used, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used, :]
    return files, kps, thresholds


def feat_path(vision_id: str, jpeg_path: str) -> Path:
    p = Path(jpeg_path)
    return FEAT_DIR / vision_id / p.parent.name / f"{p.stem}.pt"


def load_grid(path: Path) -> torch.Tensor:
    x = torch.load(path, map_location="cpu")
    if x.dim() == 4:
        x = x[0]
    return x.float()  # (C, H, W)


def windowed_soft_argmax(sim: torch.Tensor, num_patches: int, window: int) -> torch.Tensor:
    """sim: (N, N) source-patch x target-patch cosine. Return (N, 2) xy in patch coords."""
    n = num_patches
    device = sim.device
    dtype = sim.dtype
    corr = sim.view(n * n, n, n)
    max_flat = corr.flatten(1).argmax(dim=-1)
    max_x = max_flat % n
    max_y = max_flat // n
    if window and window > 0:
        yy = torch.arange(n, device=device)
        xx = torch.arange(n, device=device)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        mask = (gx.unsqueeze(0) - max_x.view(-1, 1, 1)).abs() <= window
        mask &= (gy.unsqueeze(0) - max_y.view(-1, 1, 1)).abs() <= window
        corr = corr.masked_fill(~mask, -1e9)
    corr = corr.view(n * n, n * n)
    prob = torch.softmax(corr, dim=-1)
    ys = torch.arange(n, dtype=dtype, device=device).repeat_interleave(n)
    xs = torch.arange(n, dtype=dtype, device=device).repeat(n)
    nn_x = (prob * xs).sum(dim=-1)
    nn_y = (prob * ys).sum(dim=-1)
    return torch.stack([nn_x, nn_y], dim=-1)


def _prep_grid(g: torch.Tensor, device: str) -> torch.Tensor:
    g = g.to(device=device, dtype=torch.float32)
    if g.shape[-2] != g.shape[-1]:
        side = int(round(math.sqrt(g.shape[-2] * g.shape[-1])))
        g = F.interpolate(g.unsqueeze(0), size=(side, side), mode="bilinear", align_corners=False)[0]
    return g


def pck_category(vision_id: str, category: str, device: str, subsample: int) -> dict | None:
    files, kps, thresholds = load_spair_test(SPAIR_DIR, category, subsample=subsample)
    if not files:
        return None
    n_pairs = len(files) // 2
    img_acc, kpt_correct, kpt_total = [], 0, 0
    missing = 0
    num_patches = None
    kps = kps.to(device)
    cache: dict[Path, torch.Tensor] = {}
    for i in range(n_pairs):
        p1, p2 = feat_path(vision_id, files[2 * i]), feat_path(vision_id, files[2 * i + 1])
        if not p1.is_file() or not p2.is_file():
            missing += 1
            continue
        if p1 not in cache:
            cache[p1] = _prep_grid(load_grid(p1), device)
        if p2 not in cache:
            cache[p2] = _prep_grid(load_grid(p2), device)
        g1, g2 = cache[p1], cache[p2]
        num_patches = g1.shape[-1]
        f1 = F.normalize(g1.flatten(1).T, dim=-1)
        f2 = F.normalize(g2.flatten(1).T, dim=-1)
        sim = f1 @ f2.T
        flow = windowed_soft_argmax(sim, num_patches, SOFT_EVAL_WINDOW)
        k1, k2 = kps[2 * i], kps[2 * i + 1]
        vis = (k1[:, 2] * k2[:, 2]) > 0
        if vis.sum() == 0:
            continue
        stride = ANNO_SIZE / num_patches
        y = (k1[:, 1] / ANNO_SIZE * num_patches).long().clamp(0, num_patches - 1)
        x = (k1[:, 0] / ANNO_SIZE * num_patches).long().clamp(0, num_patches - 1)
        idx = y * num_patches + x
        pred_patch = flow[idx]
        pred_xy = pred_patch * stride + stride / 2.0
        gt_xy = k2[:, :2]
        err = (pred_xy - gt_xy).norm(dim=-1)
        th = float(thresholds[i]) * 0.10
        hit = (err < th) & vis
        img_acc.append(float(hit[vis].float().mean().cpu()))
        kpt_correct += int(hit.sum().item())
        kpt_total += int(vis.sum().item())
    if not img_acc:
        return {"category": category, "n_pairs": n_pairs, "missing": missing, "pck10_img": None}
    return {
        "category": category,
        "n_pairs": n_pairs,
        "n_eval": len(img_acc),
        "missing": missing,
        "num_patches": num_patches,
        "pck10_img": float(np.mean(img_acc)),
        "pck10_kpt": (kpt_correct / kpt_total) if kpt_total else None,
        "n_kpt": kpt_total,
    }


def eval_encoder(vision_id: str, device: str, subsample: int) -> dict:
    jpeg = SPAIR_DIR / "JPEGImages"
    cats = sorted([p.name for p in jpeg.iterdir() if p.is_dir()]) if jpeg.is_dir() else []
    per = []
    for cat in tqdm(cats, desc=f"pck:{vision_id}", leave=False):
        rec = pck_category(vision_id, cat, device, subsample)
        if rec:
            per.append(rec)
    img_scores = [r["pck10_img"] for r in per if r.get("pck10_img") is not None]
    c_score = float(np.mean(img_scores)) if img_scores else None
    return {
        "vision_id": vision_id,
        "c_score": c_score,
        "c_score_pct": None if c_score is None else c_score * 100.0,
        "n_categories": len(per),
        "per_category": per,
    }


def load_done_c_scores() -> dict:
    done = {}
    for p in RESULTS_DIR.glob("c_score_shard*.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and v.get("c_score") is not None:
                    done[k] = v
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--vision-id", type=str, default="")
    parser.add_argument("--subsample", type=int, default=0, help="per-category pair cap, 0=all")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--missing-only", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    print(f"[c] device={args.device}", flush=True)

    if args.vision_id:
        vids = [args.vision_id]
    else:
        vids = unique_vision_ids(inventory_models())
        if args.missing_only:
            done0 = load_done_c_scores()
            vids = [v for v in vids if v not in done0]
        vids = shard_list(vids, args.shard, args.nshards)

    out = {}
    shard_path = RESULTS_DIR / f"c_score_shard{args.shard}.json"
    if shard_path.is_file():
        out = json.loads(shard_path.read_text())
    done = load_done_c_scores()
    for vid in vids:
        if vid in done:
            print(f"[c] skip {vid}", flush=True)
            continue
        try:
            rec = eval_encoder(vid, args.device, args.subsample)
            print(f"[c] {vid} C={rec.get('c_score_pct')}", flush=True)
            out[vid] = rec
            done[vid] = rec
            atomic_json(shard_path, out)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            out[vid] = {"vision_id": vid, "c_score": None, "error": str(exc)}
            atomic_json(shard_path, out)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
