# -*- coding: utf-8 -*-
"""
extract_text_features.py
=========================
Encode image-associated text (question/caption) via CLIP ViT-L-14 text encoder
for every image of the calibration set in features_diverse. One shared text_features.pt for all
tokenizers (text is image-level, not tokenizer-level).

Mapping: image_paths.txt -> source TSV/Arrow -> extract text -> CLIP encode.

Data sources:
  LMUData TSVs: lmu_{dataset}_{row_idx:06d}.jpg -> row row_idx in dataset.tsv
  OCR-VQA Arrow: ocr_vqa_{i:06d}.jpg -> i-th sample in Arrow iteration

Output: sample_data/text_features/text_features.pt (n_images, 768) float32

Usage (from the CKA-X/ root):
    python scripts/extract_text_features.py --device cuda
"""
import argparse
import csv
import json
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
from vision_encoder_eval.workers.ckax.scripts.ckax_common import resolve_data_path
from vision_encoder_eval.workers.ckax.config import PATHS
from vision_encoder_eval.core.runtime import asset_path

SEED = 42
# CLIP text encoder directory; override with CLIP_DIR.
CLIP_DIR = os.environ.get(
    "CLIP_DIR", asset_path('download', 'tokenizer/continuous/clip-vit-large-patch14'))


def load_texts_from_lmu_tsv(tsv_path, target_row_indices):
    texts = {}
    bench_name = os.path.splitext(os.path.basename(tsv_path))[0]
    max_ridx = max(target_row_indices)
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_idx, row in enumerate(tqdm(reader, desc=f"    TSV {bench_name}", leave=False)):
            if row_idx > max_ridx:
                break
            if row_idx not in target_row_indices:
                continue
            if "VizWiz" in bench_name:
                caption = row.get("blip_caption_beam_5", "")
                question = row.get("question", "")
                texts[row_idx] = f"Caption: {caption} Question: {question}"
            elif "MMBench" in bench_name:
                question = row.get("question", "")
                ahint = row.get("hint", "")
                a_opt = "A: " + row.get("A", "") + " B: " + row.get("B", "") + \
                        " C: " + row.get("C", "") + " D: " + row.get("D", "")
                texts[row_idx] = f"Question: {question} {ahint} Options: {a_opt}"
            elif "MME" in bench_name:
                question = row.get("question", "")
                category = row.get("category", "")
                texts[row_idx] = f"Question: {question} Category: {category}"
            elif "ScienceQA" in bench_name:
                question = row.get("question", "")
                ahint = row.get("hint", "")
                a_opt = "A: " + row.get("A", "") + " B: " + row.get("B", "") + \
                        " C: " + row.get("C", "") + " D: " + row.get("D", "") + \
                        " E: " + row.get("E", "")
                texts[row_idx] = f"Question: {question} {ahint} Options: {a_opt}"
            else:
                question = row.get("question", "")
                texts[row_idx] = f"Question: {question}"
    return texts


def load_texts_from_ocr_vqa(cache_dir, target_indices, text_map_path=None):
    """Return {row_idx: text} for OCR-VQA samples.

    If the OCR-VQA HF arrow cache is unavailable (
    datasets/howard-hou___ocr-vqa) was lost. OCR images now live in
    images_diverse/ocr_vqa_%06d.jpg with a sidecar
    images_diverse/ocr_text_map.json: {"<row_idx>": "question... answer..."}.
    Prefer that map when available; fall back to the HF-cache path.
    """
    if text_map_path and os.path.isfile(text_map_path):
        try:
            with open(text_map_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            texts = {}
            for idx in target_indices:
                t = m.get(str(idx))
                if t:
                    texts[idx] = t
            print(f"  [OCR] {len(texts)}/{len(target_indices)} texts "
                  f"from {text_map_path}")
            return texts
        except Exception as e:
            print(f"  [WARN] ocr_text_map load failed ({e}); "
                  f"falling back to HF cache")

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [WARN] datasets not installed")
        return {}
    try:
        ds = load_dataset(cache_dir, split="test")
    except Exception as e:
        print(f"  [WARN] datasets load failed: {e}")
        return {}

    texts = {}
    max_ridx = max(target_indices)
    for idx in target_indices:
        if idx >= len(ds):
            break
        item = ds[int(idx)]
        questions = item.get("questions", []) or []
        answers = item.get("answers", []) or []
        pairs = []
        for q, a in zip(questions, answers[:len(questions)] + [""] * max(0, len(questions) - len(answers))):
            pairs.append(f"{q} Answer: {a}")
        texts[idx] = "Questions: " + "; ".join(pairs) if pairs else ""
    return texts


def parse_image_paths(image_paths_file):
    sources = {}
    with open(image_paths_file, "r", encoding="utf-8") as f:
        for pos, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            fname = os.path.basename(line)
            if fname.startswith("lmu_"):
                parts = fname.replace("lmu_", "").replace(".jpg", "").rsplit("_", 1)
                if len(parts) == 2:
                    dataset, ridx_str = parts[0], parts[1]
                    row_idx = int(ridx_str)
                    sources.setdefault(("lmu", dataset), set()).add((pos, row_idx))
            elif fname.startswith("ocr_vqa_"):
                ridx_str = fname.replace("ocr_vqa_", "").replace(".jpg", "")
                row_idx = int(ridx_str)
                sources.setdefault(("ocr", "ocr-vqa"), set()).add((pos, row_idx))
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diverse_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--ocr_text_map", type=str, default=None,
                    help="path to images_diverse/ocr_text_map.json. "
                         "Auto-derived next to ocr_vqa_*.jpg when omitted.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=77)
    args = ap.parse_args()

    if args.diverse_dir is None:
        args.diverse_dir = resolve_data_path("features_diverse")
    if args.out_dir is None:
        args.out_dir = resolve_data_path(os.path.join("sample_data"))

    diverse_dir = args.diverse_dir
    lmu_path = PATHS["lmu_data"]
    ocr_cache = PATHS["ocr_vqa_cache"]

    tok_dirs = sorted(d for d in os.listdir(diverse_dir)
                      if os.path.isdir(os.path.join(diverse_dir, d)) and
                      os.path.isfile(os.path.join(diverse_dir, d, "image_paths.txt")))
    if not tok_dirs:
        raise SystemExit("  [ERROR] no tokenizer dirs with image_paths.txt found")
    ref_paths = os.path.join(diverse_dir, tok_dirs[0], "image_paths.txt")

    print("  Image paths reference:", ref_paths)
    src_map = parse_image_paths(ref_paths)

    # ocr_text_map.json sits next to the ocr_vqa_*.jpg images and
    # maps "<row_idx>" -> "question... answer..." text. Auto-derive its path
    # from the first ocr image line when --ocr_text_map is not given.
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

    num_image_paths = sum(1 for _ in open(ref_paths))
    N = num_image_paths
    text_list = [all_texts.get(i, "") for i in range(N)]
    empty = sum(1 for t in text_list if not t)
    for i in range(N):
        if not text_list[i]:
            text_list[i] = "."
    print(f"\n  Total images: {N}, with text: {N - empty}, empty: {empty}")

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"  Loading CLIP text encoder from {CLIP_DIR}")
    from transformers import CLIPTextModel, CLIPTokenizer
    model = CLIPTextModel.from_pretrained(CLIP_DIR, local_files_only=True).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_DIR, local_files_only=True)
    print(f"  CLIP text dim: {model.config.hidden_size}")

    bs = args.batch_size
    all_feats = []
    for b_start in tqdm(range(0, N, bs), desc="  Encoding text"):
        batch_texts = text_list[b_start:b_start + bs]
        batch_texts = [t if t else "." for t in batch_texts]
        with torch.no_grad():
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=args.max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            feats = outputs.pooler_output
            feats = F.normalize(feats.float(), dim=-1)
        all_feats.append(feats.cpu())
    features = torch.cat(all_feats, dim=0)
    print(f"  Features: {features.shape}")

    feat_dir = os.path.join(args.out_dir, "text_features")
    os.makedirs(feat_dir, exist_ok=True)
    out_pt = os.path.join(feat_dir, "text_features.pt")
    torch.save(features, out_pt)
    print(f"  Saved: {out_pt}")


if __name__ == "__main__":
    main()
