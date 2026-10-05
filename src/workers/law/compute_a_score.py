#!/usr/bin/env python3
"""A-score: negative mean caption NLL of Stage-1 (pretrain) MLLM on 100 samples."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_encoder_eval.workers.law.common import (  # noqa: E402
    A_N_DATAPOINTS,
    CLIP_PROC,
    PRETRAIN_IMG,
    PRETRAIN_JSON,
    RESULTS_DIR,
    apply_vision_env,
    atomic_json,
    ckpt_has_weights,
    ensure_vtb_path,
    inventory_models,
    load_vision_cfg,
    shard_list,
)


def _build_overwrite_config(model_path: str, vision: dict) -> dict:
    overwrite = {"delay_load": True}
    cfg_path = os.path.join(model_path, "config.json")
    saved = None
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        saved = cfg.get("mm_vision_tower") or cfg.get("vision_tower")
    if saved and os.path.isdir(str(saved)):
        return overwrite
    non_hf = ("open_clip_hub:", "hf:", "vtb_ssl:")
    if saved and str(saved).startswith(non_hf):
        overwrite["mm_vision_tower"] = saved
        return overwrite
    vision_tower = vision.get("vision_tower") or vision.get("model_name_or_path")
    weights = vision.get("weights_path")
    processor = vision.get("processor_path") or CLIP_PROC
    if vision_tower and str(vision_tower).startswith(non_hf):
        overwrite["mm_vision_tower"] = vision_tower
        return overwrite
    if weights and os.path.isdir(str(weights)):
        overwrite["mm_vision_tower"] = weights
    elif vision_tower and os.path.isdir(str(vision_tower)):
        overwrite["mm_vision_tower"] = vision_tower
    elif processor and os.path.isdir(processor):
        if (not saved) or "clip" in str(saved).lower() or str(saved).startswith("openai/"):
            overwrite["mm_vision_tower"] = processor
    return overwrite


def _resolve_model_name(model_path: str, llm_path: str | None) -> str:
    for source in (model_path, llm_path or ""):
        cfg_path = os.path.join(source, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            if cfg.get("model_type") == "qwen3" or "Qwen3" in str(cfg.get("architectures", [])):
                return "qwen3"
            if cfg.get("model_type") == "qwen2" or "Qwen2" in str(cfg.get("architectures", [])):
                return "qwen"
            if cfg.get("model_type") == "llama" or "LlavaLlama" in str(cfg.get("architectures", [])):
                if "smol" in source.lower():
                    return "smollm2"
                return "llava_llama_3"
        lower = source.lower()
        if "qwen3" in lower:
            return "qwen3"
        if "qwen2" in lower or "qwen25" in lower or "qwen2.5" in lower:
            return "qwen"
        if "smol" in lower:
            return "smollm2"
        if "llama" in lower:
            return "llava_llama_3"
    return "qwen3"


def load_pretrain_samples(n: int) -> list[dict]:
    data = json.load(open(PRETRAIN_JSON))
    out = []
    for row in data:
        if not row.get("image"):
            continue
        img = os.path.join(PRETRAIN_IMG, row["image"])
        if not os.path.isfile(img):
            continue
        conv = row.get("conversations") or []
        if len(conv) < 2:
            continue
        out.append({"image": img, "conversations": conv})
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"only found {len(out)} pretrain samples with images")
    return out


def load_mllm(row: dict, device_index: int):
    ensure_vtb_path()
    vision = load_vision_cfg(row["vision_id"])
    apply_vision_env(vision)
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    disable_torch_init()
    ckpt = row["pretrain_dir"]
    if not ckpt_has_weights(Path(ckpt)):
        raise FileNotFoundError(ckpt)
    adapter = os.path.isfile(os.path.join(ckpt, "mm_projector.bin")) and not (
        os.path.isfile(os.path.join(ckpt, "model.safetensors"))
        or os.path.isfile(os.path.join(ckpt, "pytorch_model.bin"))
        or os.path.isfile(os.path.join(ckpt, "model.safetensors.index.json"))
    )
    model_base = row["llm_path"] if adapter else None
    model_name = _resolve_model_name(ckpt if not adapter else row["llm_path"], row["llm_path"])
    overwrite = _build_overwrite_config(ckpt, vision)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        ckpt,
        model_base,
        model_name,
        multimodal=True,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        device_map={"": device_index},
        overwrite_config=overwrite,
    )
    if image_processor is None:
        raise RuntimeError("image_processor is None — vision tower failed to load")
    device = torch.device(f"cuda:{device_index}")
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    return tokenizer, model, image_processor, vision


def _pad_ids_labels(ids_list, labels_list, pad_id: int):
    max_len = max(int(x.shape[-1]) for x in ids_list)
    ids_out, lab_out, attn_out = [], [], []
    for iid, lab in zip(ids_list, labels_list):
        if iid.dim() == 2:
            iid = iid[0]
            lab = lab[0]
        n = int(iid.shape[0])
        pad = max_len - n
        if pad:
            iid = torch.nn.functional.pad(iid, (0, pad), value=pad_id)
            lab = torch.nn.functional.pad(lab, (0, pad), value=-100)
        ids_out.append(iid)
        lab_out.append(lab)
        attn_out.append(torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
    return torch.stack(ids_out), torch.stack(lab_out), torch.stack(attn_out)


def _prep_sample(sample, tokenizer, data_args):
    from llava.train.train import preprocess, preprocess_multimodal

    sources = preprocess_multimodal(copy.deepcopy([sample["conversations"]]), data_args)
    batch = preprocess(sources, tokenizer, has_image=True)
    return batch["input_ids"], batch["labels"], Image.open(sample["image"]).convert("RGB")


def _forward_loss(model, input_ids, labels, attention_mask, images, device):
    if isinstance(images, list):
        images = [t.to(device=device, dtype=torch.bfloat16) for t in images]
    else:
        images = images.to(device=device, dtype=torch.bfloat16)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=input_ids.to(device),
            labels=labels.to(device),
            attention_mask=attention_mask.to(device),
            images=images,
        )
    return float(out.loss.detach().float().cpu())


@torch.no_grad()
def a_score_one(row: dict, samples: list[dict], device_index: int, batch_size: int = 8) -> dict:
    from llava import conversation as conversation_lib
    from llava.mm_utils import process_images

    tokenizer, model, image_processor, _vision = load_mllm(row, device_index)
    version = row.get("llm_version") or "qwen_3"
    if version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[version]
    data_args = SimpleNamespace(is_multimodal=True, mm_use_im_start_end=False)
    device = torch.device(f"cuda:{device_index}")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    losses = []
    t0 = time.time()
    bs = max(1, int(batch_size))
    pbar = tqdm(range(0, len(samples), bs), desc=row["key"], leave=False)
    for start in pbar:
        chunk = samples[start : start + bs]
        try:
            ids_list, lab_list, pil_list = [], [], []
            for sample in chunk:
                iid, lab, img = _prep_sample(sample, tokenizer, data_args)
                ids_list.append(iid)
                lab_list.append(lab)
                pil_list.append(img)
            input_ids, labels, attn = _pad_ids_labels(ids_list, lab_list, pad_id)
            images = process_images(pil_list, image_processor, model.config)
            loss = _forward_loss(model, input_ids, labels, attn, images, device)
            if math_isnan(loss):
                continue
            losses.extend([loss] * len(chunk))
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            for sample in chunk:
                try:
                    iid, lab, img = _prep_sample(sample, tokenizer, data_args)
                    input_ids, labels, attn = _pad_ids_labels([iid], [lab], pad_id)
                    images = process_images([img], image_processor, model.config)
                    loss = _forward_loss(model, input_ids, labels, attn, images, device)
                    if not math_isnan(loss):
                        losses.append(loss)
                except RuntimeError as exc2:
                    if "out of memory" in str(exc2).lower():
                        torch.cuda.empty_cache()
                        continue
                    raise
    del model
    torch.cuda.empty_cache()
    if not losses:
        raise RuntimeError("no valid losses")
    avg = sum(losses) / len(losses)
    return {
        "key": row["key"],
        "vision_id": row["vision_id"],
        "llm_id": row["llm_id"],
        "n": len(losses),
        "avg_loss": avg,
        "a_score": -avg,
        "seconds": round(time.time() - t0, 1),
        "batch_size": bs,
        "losses": losses,
    }


def math_isnan(x: float) -> bool:
    return x != x


def load_done_a_scores() -> dict:
    done = {}
    for p in RESULTS_DIR.glob("a_score_shard*.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and v.get("a_score") is not None:
                    done[k] = v
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--n", type=int, default=A_N_DATAPOINTS)
    parser.add_argument("--key", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--missing-only", action="store_true")
    args = parser.parse_args()

    ensure_vtb_path()
    rows = inventory_models()
    rows = [r for r in rows if r.get("pretrain_ok")]
    if args.key:
        rows = [r for r in rows if r["key"] == args.key or r["recipe_name"] == args.key]
    done = load_done_a_scores()
    if args.missing_only:
        rows = [r for r in rows if r["key"] not in done]
    if not args.key:
        rows = shard_list(rows, args.shard, args.nshards)
    print(
        f"[a] shard {args.shard}/{args.nshards}: {len(rows)} models, n={args.n}, bs={args.batch_size}, missing_only={args.missing_only}",
        flush=True,
    )
    samples = load_pretrain_samples(args.n)
    shard_path = RESULTS_DIR / f"a_score_shard{args.shard}.json"
    out = json.loads(shard_path.read_text()) if shard_path.is_file() else {}
    device_index = 0
    for row in rows:
        key = row["key"]
        if key in done:
            print(f"[a] skip {key}", flush=True)
            continue
        try:
            rec = a_score_one(row, samples, device_index, batch_size=args.batch_size)
            print(f"[a] {key} A={rec['a_score']:.6f} loss={rec['avg_loss']:.6f}", flush=True)
            out[key] = rec
            done[key] = rec
            atomic_json(shard_path, out)
        except Exception as exc:
            traceback.print_exc()
            out[key] = {"key": key, "a_score": None, "error": str(exc)}
            atomic_json(shard_path, out)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
