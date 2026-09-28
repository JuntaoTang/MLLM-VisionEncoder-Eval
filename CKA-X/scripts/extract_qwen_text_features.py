# -*- coding: utf-8 -*-
"""
extract_qwen_text_features.py
===============================
Encode image-associated text via Qwen3-1.7B (the LLM used downstream by the MLLM)
for every image of the calibration set in features_diverse.

Qwen3 text space is 2048d, shared with the LLM that processes questions in the
actual MLLM evaluation — cross-modal metrics in this space are directly probing
"can the tokenizer features predict what the LLM reads".

Reuses text collection from extract_text_features.py. Only the encoder differs.

Output (by --encoder):
    qwen3      -> sample_data/text_features/text_features_qwen.pt        (n_images, 2048)
    qwen25     -> sample_data/text_features/text_features_qwen25.pt      (n_images, 1536)
    qwen25_3b  -> sample_data/text_features/text_features_qwen25_3b.pt   (n_images, 2048)
    qwen25_7b  -> sample_data/text_features/text_features_qwen25_7b.pt   (n_images, 3584)
    qwen25_14b -> sample_data/text_features/text_features_qwen25_14b.pt  (n_images, 5120)
    llama32    -> sample_data/text_features/text_features_llama32.pt     (n_images, 2048)
    qwen3_14b -> sample_data/text_features/text_features_qwen3_14b.pt  (n_images, 5120)
    qwen3_32b -> sample_data/text_features/text_features_qwen3_32b.pt  (n_images, 5120)

Usage (from the CKA-X/ root):
    python scripts/extract_qwen_text_features.py --device cuda
"""
import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:  # Windows: C long cap
    csv.field_size_limit(2**31 - 1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_text_features import (
    load_texts_from_lmu_tsv, load_texts_from_ocr_vqa, parse_image_paths,
)
from ckax_common import resolve_data_path

SEED = 42
# Frozen text-encoder registry. All encoders use BASE weights (no instruct),
# so comparisons are flavor-matched.
_LLM_DEFAULT = os.environ.get(
    "LLM_ROOT", os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "llm"))

ENCODERS = {
    "qwen3": {
        "dir": _LLM_DEFAULT + "/Qwen3-1.7B",
        "out": "text_features_qwen.pt",
        "label": "Qwen3-1.7B (base)",
    },
    "qwen25": {
        "dir": _LLM_DEFAULT + "/Qwen2.5-1.5B",
        "out": "text_features_qwen25.pt",
        "label": "Qwen2.5-1.5B (base)",
    },
    "qwen25_3b": {
        "dir": _LLM_DEFAULT + "/Qwen2.5-3B",
        "out": "text_features_qwen25_3b.pt",
        "label": "Qwen2.5-3B (base)",
    },
    "qwen25_7b": {
        "dir": _LLM_DEFAULT + "/Qwen2.5-7B",
        "out": "text_features_qwen25_7b.pt",
        "label": "Qwen2.5-7B (base)",
    },
    "qwen25_14b": {
        "dir": _LLM_DEFAULT + "/Qwen2.5-14B",
        "out": "text_features_qwen25_14b.pt",
        "label": "Qwen2.5-14B (base)",
    },
    "llama32": {
        "dir": _LLM_DEFAULT + "/Llama-3.2-1B",
        "out": "text_features_llama32.pt",
        "label": "Llama-3.2-1B (base)",
    },
    "qwen3_14b": {
        "dir": _LLM_DEFAULT + "/Qwen3-14B",
        "out": "text_features_qwen3_14b.pt",
        "label": "Qwen3-14B (base)",
    },
    "qwen3_32b": {
        "dir": _LLM_DEFAULT + "/Qwen3-32B",
        "out": "text_features_qwen3_32b.pt",
        "label": "Qwen3-32B (base)",
    },
}

# --- The registry above lists where each checkpoint is expected.  LLM_ROOT
#     re-roots all of them onto one directory, keeping only the checkpoint's
#     name (e.g. $LLM_ROOT/Qwen2.5-1.5B); it defaults to ./llm inside this
#     package.  Point LLM_ROOT (or --encoder) at your own download.
_LLM_ROOT = os.environ.get("LLM_ROOT")
if _LLM_ROOT:
    for _spec in ENCODERS.values():
        _spec["dir"] = os.path.join(_LLM_ROOT,
                                    os.path.basename(_spec["dir"].rstrip("/")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diverse_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--ocr_text_map", type=str, default=None,
                    help="path to images_diverse/ocr_text_map.json. "
                         "Auto-derived next to ocr_vqa_*.jpg when omitted.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_images", type=int, default=0)
    ap.add_argument("--encoder", type=str, default="qwen3",
                    choices=sorted(ENCODERS),
                    help="frozen text encoder: qwen3 (base) or "
                         "qwen25 (Instruct)")
    args = ap.parse_args()

    if args.diverse_dir is None:
        args.diverse_dir = resolve_data_path("features_diverse")
    if args.out_dir is None:
        args.out_dir = resolve_data_path(os.path.join("sample_data"))

    diverse_dir = args.diverse_dir
    lmu_path = os.environ.get("LMU_DATA", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LMUData"))
    ocr_cache = os.environ.get("OCR_VQA_CACHE", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ocr-vqa"))

    tok_dirs = sorted(d for d in os.listdir(diverse_dir)
                      if os.path.isdir(os.path.join(diverse_dir, d)) and
                      os.path.isfile(os.path.join(diverse_dir, d, "image_paths.txt")))
    if not tok_dirs:
        raise SystemExit("  [ERROR] no tokenizer dirs with image_paths.txt found")
    ref_paths = os.path.join(diverse_dir, tok_dirs[0], "image_paths.txt")

    print("  Image paths reference:", ref_paths)
    src_map = parse_image_paths(ref_paths)
    num_image_paths = sum(1 for _ in open(ref_paths))

    # ocr_text_map.json sits next to the ocr_vqa_*.jpg images and
    # maps "<row_idx>" -> "question... answer..." text (HF arrow cache lost).
    ocr_text_map = args.ocr_text_map
    if ocr_text_map is None:
        for line in open(ref_paths, "r", encoding="utf-8"):
            line = line.strip()
            if os.path.basename(line).startswith("ocr_vqa_"):
                cand = os.path.join(os.path.dirname(line), "ocr_text_map.json")
                if os.path.isfile(cand):
                    ocr_text_map = cand
                    break
    if ocr_text_map and os.path.isfile(ocr_text_map):
        print("  OCR text map:", ocr_text_map)
    else:
        print("  [WARN] ocr_text_map.json not found; OCR texts will fall "
              "back to HF cache (likely empty on this server)")

    all_texts = {}
    for (src_type, dataset), entries in tqdm(sorted(src_map.items()), desc="  Parsing sources", unit="source"):
        positions, row_indices = zip(*entries)
        row_set = set(row_indices)
        if src_type == "lmu":
            tsv_path = os.path.join(lmu_path, dataset + ".tsv")
            if not os.path.isfile(tsv_path):
                print(f"  [SKIP] TSV not found: {tsv_path}")
                continue
            row_texts = load_texts_from_lmu_tsv(tsv_path, row_set)
        else:
            row_texts = load_texts_from_ocr_vqa(ocr_cache, row_set,
                                                text_map_path=ocr_text_map)

        matched = 0
        for pos, ridx in zip(positions, row_indices):
            if ridx in row_texts:
                all_texts[pos] = row_texts[ridx]
                matched += 1
        print(f"    {dataset}: {matched}/{len(positions)} matched")

    N = num_image_paths
    text_list = [all_texts.get(i, ".") for i in range(N)]
    empty = sum(1 for t in text_list if t == ".")
    print(f"\n  Total images: {N}, with text: {N - empty}, empty: {empty}")

    device = args.device if torch.cuda.is_available() else "cpu"
    enc = ENCODERS[args.encoder]
    model_dir = enc["dir"]
    print(f"  Loading {enc['label']} from {model_dir}")
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True,
                                       torch_dtype=torch.float16, device_map="auto" if device != "cpu" else None)
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    print(f"  {enc['label']} hidden size: {model.config.hidden_size}")

    if args.num_images > 0:
        N = min(args.num_images, N)
        text_list = text_list[:N]

    bs = args.batch_size
    all_feats = []
    for b_start in tqdm(range(0, N, bs), desc="  Encoding text"):
        batch_texts = text_list[b_start:b_start + bs]
        with torch.no_grad():
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            feats = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            feats = F.normalize(feats.float(), dim=-1)
        all_feats.append(feats.cpu())
        del inputs, outputs, hidden
    features = torch.cat(all_feats, dim=0)
    print(f"  Features: {features.shape}")

    out_dir = os.path.join(args.out_dir, "text_features")
    os.makedirs(out_dir, exist_ok=True)
    out_pt = os.path.join(out_dir, enc["out"])
    torch.save(features, out_pt)
    print(f"  Saved: {out_pt}")


if __name__ == "__main__":
    main()
