"""Batch ImageNet-1K zero-shot evaluation for MetaCLIP 2 worldwide models.

The default batch contains the 12 checkpoints that complement the already
evaluated B/16 224px baseline. Use ``--include-baseline`` to evaluate all 13.

Example:
    python zero_shot/zero_shot_metaclip2_imagenet.py \
        --root /path/to/metaclip2/checkpoints \
        --val /path/to/imagenet/val \
        --mt5-spm google/mt5-base
"""

import argparse
import csv
import gc
import json
import math
import os
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer, MetaClip2Config, MetaClip2Model

BASELINE_ALREADY_EVALUATED = "metaclip2_b16_224px_worldwide.pt"
ALL_MODEL_FILENAMES = (
    "metaclip2_b16_224px_worldwide.pt",
    "metaclip2_b16_384px_worldwide.pt",
    "metaclip2_b32_224px_mt5_worldwide.pt",
    "metaclip2_b32_224px_worldwide.pt",
    "metaclip2_b32_384px_worldwide.pt",
    "metaclip2_h14_378px_worldwide.pt",
    "metaclip2_l14_224px_worldwide.pt",
    "metaclip2_m16_224px_mt5_worldwide.pt",
    "metaclip2_m16_224px_worldwide.pt",
    "metaclip2_m16_384px_worldwide.pt",
    "metaclip2_s16_224px_mt5_worldwide.pt",
    "metaclip2_s16_224px_worldwide.pt",
    "metaclip2_s16_384px_worldwide.pt",
)
DEFAULT_MODEL_FILENAMES = tuple(
    name for name in ALL_MODEL_FILENAMES if name != BASELINE_ALREADY_EVALUATED
)


def resolve_mt5_spm(source):
    """Resolve a local SentencePiece file or download it from a Hub repo."""
    path = Path(source).expanduser()
    if path.is_file():
        return path
    return Path(hf_hub_download(repo_id=source, filename="spiece.model"))


def block_count(state_dict, prefix):
    marker = prefix + ".resblocks."
    return len({int(k[len(marker) :].split(".", 1)[0]) for k in state_dict if k.startswith(marker)})


def make_config(sd, tokenizer_kind):
    tw = sd["token_embedding.weight"].shape[1]
    vw, _, patch, patch2 = sd["visual.conv1.weight"].shape
    assert patch == patch2
    grid = int(math.sqrt(sd["visual.positional_embedding"].shape[0] - 1))
    assert grid * grid + 1 == sd["visual.positional_embedding"].shape[0]
    image_size = grid * patch
    projection_dim = sd["text_projection"].shape[1]
    pad, bos, eos = (0, None, 1) if tokenizer_kind == "mt5" else (1, 0, 2)
    return MetaClip2Config(
        text_config={
            "vocab_size": sd["token_embedding.weight"].shape[0],
            "hidden_size": tw,
            "intermediate_size": sd["transformer.resblocks.0.mlp.c_fc.weight"].shape[0],
            "projection_dim": projection_dim,
            "num_hidden_layers": block_count(sd, "transformer"),
            "num_attention_heads": tw // 64,
            "max_position_embeddings": sd["positional_embedding"].shape[0],
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-5,
            "pad_token_id": pad,
            "bos_token_id": bos,
            "eos_token_id": eos,
        },
        vision_config={
            "hidden_size": vw,
            "intermediate_size": sd["visual.transformer.resblocks.0.mlp.c_fc.weight"].shape[0],
            "projection_dim": projection_dim,
            "num_hidden_layers": block_count(sd, "visual.transformer"),
            "num_attention_heads": vw // 64,
            "image_size": image_size,
            "patch_size": patch,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-5,
        },
        projection_dim=projection_dim,
        logit_scale_init_value=float(sd["logit_scale"]),
    )


def map_block_key(key, source_prefix, target_prefix, converted, value):
    block, tail = key[len(source_prefix) + 1 :].split(".", 1)
    base = f"{target_prefix}.{block}"
    simple = {
        "attn.out_proj.weight": "self_attn.out_proj.weight",
        "attn.out_proj.bias": "self_attn.out_proj.bias",
        "ln_1.weight": "layer_norm1.weight",
        "ln_1.bias": "layer_norm1.bias",
        "ln_2.weight": "layer_norm2.weight",
        "ln_2.bias": "layer_norm2.bias",
        "mlp.c_fc.weight": "mlp.fc1.weight",
        "mlp.c_fc.bias": "mlp.fc1.bias",
        "mlp.c_proj.weight": "mlp.fc2.weight",
        "mlp.c_proj.bias": "mlp.fc2.bias",
    }
    if tail in simple:
        converted[f"{base}.{simple[tail]}"] = value
    elif tail in ("attn.in_proj_weight", "attn.in_proj_bias"):
        suffix = "weight" if tail.endswith("weight") else "bias"
        q, k, v = value.chunk(3, dim=0)
        converted[f"{base}.self_attn.q_proj.{suffix}"] = q
        converted[f"{base}.self_attn.k_proj.{suffix}"] = k
        converted[f"{base}.self_attn.v_proj.{suffix}"] = v
    else:
        raise KeyError(key)


def convert_state_dict(sd):
    converted = {}
    direct = {
        "logit_scale": "logit_scale",
        "token_embedding.weight": "text_model.embeddings.token_embedding.weight",
        "positional_embedding": "text_model.embeddings.position_embedding.weight",
        "ln_final.weight": "text_model.final_layer_norm.weight",
        "ln_final.bias": "text_model.final_layer_norm.bias",
        "visual.class_embedding": "vision_model.embeddings.class_embedding",
        "visual.positional_embedding": "vision_model.embeddings.position_embedding.weight",
        "visual.conv1.weight": "vision_model.embeddings.patch_embedding.weight",
        "visual.ln_pre.weight": "vision_model.pre_layrnorm.weight",
        "visual.ln_pre.bias": "vision_model.pre_layrnorm.bias",
        "visual.ln_post.weight": "vision_model.post_layernorm.weight",
        "visual.ln_post.bias": "vision_model.post_layernorm.bias",
    }
    for key, value in sd.items():
        if key in direct:
            converted[direct[key]] = value
        elif key == "text_projection":
            converted["text_projection.weight"] = value.T.contiguous()
        elif key == "visual.proj":
            converted["visual_projection.weight"] = value.T.contiguous()
        elif key.startswith("transformer.resblocks."):
            map_block_key(key, "transformer.resblocks", "text_model.encoder.layers", converted, value)
        elif key.startswith("visual.transformer.resblocks."):
            map_block_key(key, "visual.transformer.resblocks", "vision_model.encoder.layers", converted, value)
        else:
            raise KeyError(f"Unmapped checkpoint key: {key}")
    return converted


def load_model(checkpoint, tokenizer_kind, device):
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = raw["state_dict"]
    config = make_config(sd, tokenizer_kind)
    converted = convert_state_dict(sd)
    model = MetaClip2Model(config)
    missing, unexpected = model.load_state_dict(converted, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    del raw, sd, converted
    return model.to(device=device, dtype=torch.bfloat16).eval(), config.vision_config.image_size


class TextTokenizer:
    def __init__(self, kind, xlm_tokenizer, mt5_spm):
        self.kind = kind
        if kind == "mt5":
            self.sp = spm.SentencePieceProcessor(model_file=str(mt5_spm))
            assert self.sp.pad_id() == 0 and self.sp.eos_id() == 1
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(xlm_tokenizer)

    def __call__(self, texts):
        if self.kind != "mt5":
            return self.tokenizer(texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        rows = []
        for text in texts:
            ids = self.sp.encode(text, out_type=int)[:76] + [self.sp.eos_id()]
            ids += [self.sp.pad_id()] * (77 - len(ids))
            assert max(ids) < 250000
            rows.append(ids)
        input_ids = torch.tensor(rows, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": input_ids.ne(self.sp.pad_id()).long()}


def projected(output):
    return output.pooler_output if hasattr(output, "pooler_output") else output


def build_classifier(model, tokenizer, device, text_batch_size):
    prompts = [template(name) for name in IMAGENET_CLASSNAMES for template in OPENAI_IMAGENET_TEMPLATES]
    batches = []
    with torch.inference_mode():
        for start in range(0, len(prompts), text_batch_size):
            tokens = {k: v.to(device) for k, v in tokenizer(prompts[start : start + text_batch_size]).items()}
            batches.append(F.normalize(projected(model.get_text_features(**tokens)).float(), dim=-1).cpu())
    features = torch.cat(batches).view(len(IMAGENET_CLASSNAMES), len(OPENAI_IMAGENET_TEMPLATES), -1)
    return F.normalize(features.mean(1), dim=-1).T.contiguous().to(device=device, dtype=torch.bfloat16)


def make_loader(val_root, image_size, batch_size, workers):
    preprocess = transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    dataset = datasets.ImageFolder(val_root, transform=preprocess)
    assert len(dataset) == 50000 and len(dataset.classes) == 1000
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    return dataset, loader


def auto_batch_size(image_size, vision_width, requested):
    if requested:
        return requested
    if image_size >= 378 and vision_width >= 1280:
        return 96
    if image_size >= 378:
        return 256
    if vision_width >= 1024:
        return 256
    return 512


def evaluate_one(checkpoint, args, device):
    name = checkpoint.name
    kind = "mt5" if "_mt5_" in name else "xlm"
    print(f"\n===== {name} ({kind}) =====", flush=True)
    setup_start = time.time()
    model, image_size = load_model(checkpoint, kind, device)
    batch_size = auto_batch_size(image_size, model.config.vision_config.hidden_size, args.batch_size)
    tokenizer = TextTokenizer(kind, args.xlm_tokenizer, args.mt5_spm)
    classifier = build_classifier(model, tokenizer, device, args.text_batch_size)
    print(f"image_size={image_size}, batch_size={batch_size}, setup={time.time()-setup_start:.1f}s", flush=True)
    dataset, loader = make_loader(args.val, image_size, batch_size, args.workers)
    correct1 = seen = 0
    started = time.time()
    with torch.inference_mode():
        for images, target in loader:
            images = images.to(device=device, dtype=torch.bfloat16, non_blocking=True)
            target = target.to(device=device, non_blocking=True)
            features = F.normalize(projected(model.get_image_features(pixel_values=images)).float(), dim=-1)
            correct1 += (features @ classifier.float()).argmax(1).eq(target).sum().item()
            old_seen, seen = seen, seen + target.numel()
            if seen == len(dataset) or seen // 10000 != old_seen // 10000:
                print(f"[{seen:5d}/{len(dataset)}] top1={100*correct1/seen:.4f}% elapsed={time.time()-started:.1f}s", flush=True)
    result = {
        "model": name, "checkpoint": checkpoint.name, "tokenizer": kind, "image_size": image_size,
        "images": seen, "classes": 1000, "templates": 80, "precision": "bfloat16",
        "top1_correct": correct1, "top1_percent": 100.0 * correct1 / seen,
        "eval_seconds": time.time() - started,
    }
    del loader, dataset, classifier, model
    gc.collect(); torch.cuda.empty_cache()
    return result


def save_results(results, output_json, output_csv):
    Path(output_json).write_text(json.dumps(results, indent=2) + "\n")
    if results:
        with Path(output_csv).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("METACLIP2_MODEL_ROOT", "checkpoints/metaclip2")),
        help="Directory containing the MetaCLIP 2 .pt checkpoints",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=Path(os.environ.get("IMAGENET_VAL_ROOT", "data/imagenet/val")),
        help="Official ImageNet-1K validation ImageFolder",
    )
    parser.add_argument(
        "--xlm-tokenizer",
        default=os.environ.get(
            "METACLIP2_XLM_TOKENIZER", "facebook/metaclip-2-worldwide-b16"
        ),
        help="Hugging Face repo id or local tokenizer directory",
    )
    parser.add_argument(
        "--mt5-spm",
        default=os.environ.get("METACLIP2_MT5_SPM", "google/mt5-base"),
        help="Local spiece.model or Hugging Face repo id",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=0, help="0 selects per-model batch size")
    parser.add_argument("--text-batch-size", type=int, default=512)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--output-json",
        default="results/zero_shot/metaclip2_12models_imagenet_zeroshot.json",
    )
    parser.add_argument(
        "--output-csv",
        default="results/zero_shot/metaclip2_12models_imagenet_zeroshot.csv",
    )
    args = parser.parse_args()
    selected_names = (
        tuple(args.only)
        if args.only
        else (ALL_MODEL_FILENAMES if args.include_baseline else DEFAULT_MODEL_FILENAMES)
    )
    if args.list_models:
        for name in selected_names:
            print(name)
        return

    if args.only:
        unknown = sorted(set(args.only) - set(ALL_MODEL_FILENAMES))
        if unknown:
            raise ValueError(f"Unknown model filenames: {unknown}")

    assert torch.cuda.is_available(), "CUDA is required"
    args.mt5_spm = resolve_mt5_spm(args.mt5_spm)
    checkpoints = [args.root / name for name in selected_names]
    missing = [p.name for p in checkpoints if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints under {args.root}: {missing}")
    print(f"Models to evaluate ({len(checkpoints)}): {[p.name for p in checkpoints]}", flush=True)
    if not args.only:
        assert len(checkpoints) == (13 if args.include_baseline else 12)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    results = []
    for checkpoint in checkpoints:
        result = evaluate_one(checkpoint, args, torch.device("cuda")); results.append(result)
        save_results(results, args.output_json, args.output_csv); print("RESULT " + json.dumps(result), flush=True)
    print("\n===== SUMMARY =====", flush=True)
    for result in results: print(f"{result['model']}: {result['top1_percent']:.4f}%", flush=True)
    print(f"JSON: {args.output_json}\nCSV: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
