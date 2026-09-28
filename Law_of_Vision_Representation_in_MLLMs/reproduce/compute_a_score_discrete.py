#!/usr/bin/env python3
"""A-score for discrete MLLMs: Stage-1 (mix/pretrain) caption NLL on 100 samples."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import A_N_DATAPOINTS, PRETRAIN_IMG, PRETRAIN_JSON, RESULTS_DIR, atomic_json, shard_list  # noqa: E402
from compute_a_score import load_done_a_scores, load_pretrain_samples  # noqa: E402

VTB = Path("/home/ma-user/work_space/VTB")
DISCRETE_IDS = ("toklip_l_384", "toklip_s_256", "unitok_attn", "vilau_256", "uniar_bsq")
LLMS = ("qwen3", "qwen25", "smollm2")
RUNTIME = VTB / "configs" / "runtime.yaml"


def _cfg_from_ctx(ctx) -> dict:
    return {
        "llm": ctx.llm,
        "tokenizer": ctx.tokenizer,
        "projector": ctx.projector,
        "arch": ctx.arch,
    }


def load_discrete_pretrain(recipe: str, device: str):
    sys.path.insert(0, str(VTB))
    from src.discrete.config import load_run_context
    from src.discrete.train.train import build_model

    ctx = load_run_context(str(RUNTIME), recipe_override=recipe)
    pretrain_dir = ctx.pretrain_dir
    ckpt = os.path.join(pretrain_dir, "pytorch_model.bin")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)
    model, tokenizer, vq = build_model(_cfg_from_ctx(ctx))
    state = torch.load(ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[a-disc] loaded {ckpt} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    return model, tokenizer, vq, ctx, pretrain_dir


@torch.no_grad()
def a_score_one(recipe: str, samples: list[dict], device: str) -> dict:
    sys.path.insert(0, str(VTB))
    from src.discrete.data.dataset import LLaVADataset, collate_fn

    model, tokenizer, vq, ctx, pretrain_dir = load_discrete_pretrain(recipe, device)
    ds = LLaVADataset(
        data_path=PRETRAIN_JSON,
        image_folder=PRETRAIN_IMG,
        tokenizer=tokenizer,
        image_size=int(vq.image_size),
        max_length=2048,
        samples=samples,
    )
    n = min(len(ds), A_N_DATAPOINTS)
    losses = []
    t0 = time.time()
    for i in tqdm(range(n), desc=recipe):
        batch = collate_fn([ds[i]])
        pixel = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attn = batch["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(pixel_values=pixel, input_ids=ids, labels=labels, attention_mask=attn)
        loss = float(out["loss"].detach().float().cpu())
        if loss == loss:
            losses.append(loss)
    del model
    torch.cuda.empty_cache()
    if not losses:
        raise RuntimeError("no valid losses")
    avg = sum(losses) / len(losses)
    llm, vid = recipe.split("/", 1)
    return {
        "key": f"discrete/{llm}/{vid}",
        "vision_id": vid,
        "llm_id": llm,
        "n": len(losses),
        "avg_loss": avg,
        "a_score": -avg,
        "seconds": round(time.time() - t0, 1),
        "pretrain_dir": pretrain_dir,
        "family": "discrete",
        "losses": losses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=str, default="")
    parser.add_argument("--n", type=int, default=A_N_DATAPOINTS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    args = parser.parse_args()
    os.environ.setdefault("VTB_ROOT", str(VTB))
    sys.path.insert(0, str(VTB))
    recipes = [args.recipe] if args.recipe else [f"{llm}/{vid}" for llm in LLMS for vid in DISCRETE_IDS]
    if not args.recipe:
        recipes = shard_list(recipes, args.shard, args.nshards)
    print(f"[a-disc] shard {args.shard}/{args.nshards}: {recipes}", flush=True)
    done = load_done_a_scores()
    sample_rows = []
    for row in load_pretrain_samples(args.n):
        img = row["image"]
        rel = os.path.relpath(img, PRETRAIN_IMG) if os.path.isabs(img) else img
        sample_rows.append({"image": rel, "conversations": row["conversations"]})
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    shard_path = RESULTS_DIR / f"a_score_shard_d{args.shard}.json"
    out = __import__("json").loads(shard_path.read_text()) if shard_path.is_file() else {}
    for recipe in recipes:
        key = "discrete/" + recipe
        if key in done or (out.get(key) or {}).get("a_score") is not None:
            print(f"[a-disc] skip {key}", flush=True)
            continue
        try:
            rec = a_score_one(recipe, sample_rows, device)
            print(f"[a-disc] {key} A={rec['a_score']:.6f}", flush=True)
            out[key] = rec
            done[key] = rec
            atomic_json(shard_path, out)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            out[key] = {"key": key, "a_score": None, "error": str(exc)}
            atomic_json(shard_path, out)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
