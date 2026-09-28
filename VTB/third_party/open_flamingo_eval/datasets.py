"""LMUData TSV loaders for OpenFlamingo-style eval."""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import os
import random
import tempfile
from dataclasses import dataclass
from typing import Any

import pandas as pd
from PIL import Image


@dataclass
class CaptionSample:
    index: Any
    image_path: str
    captions: list[str]


@dataclass
class VQASample:
    index: Any
    image_path: str
    question: str
    answers: list[str]


@dataclass
class ClassSample:
    index: Any
    image_path: str
    question: str
    label: str  # e.g. yes / no
    candidates: list[str]


def _parse_list_cell(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            continue
    return [text]


def _image_cache_dir() -> str:
    root = os.environ.get("VTB_OF_IMAGE_CACHE") or os.path.join(
        tempfile.gettempdir(), "vtb_open_flamingo_images"
    )
    os.makedirs(root, exist_ok=True)
    return root


def _materialize_image(row: pd.Series, dataset_name: str) -> str | None:
    """Resolve image_path or base64 ``image`` cell to a local file path."""
    if "image_path" in row and not pd.isna(row.get("image_path")):
        path = str(row["image_path"])
        if os.path.isfile(path):
            return path
    if "image" in row and not pd.isna(row.get("image")):
        raw = str(row["image"])
        if os.path.isfile(raw):
            return raw
        try:
            data = base64.b64decode(raw)
        except Exception:
            return None
        digest = hashlib.md5(f"{dataset_name}:{row.get('index', '')}".encode()).hexdigest()
        out = os.path.join(_image_cache_dir(), f"{dataset_name}_{digest}.jpg")
        if not os.path.isfile(out):
            Image.open(io.BytesIO(data)).convert("RGB").save(out, format="JPEG")
        return out
    return None


def resolve_tsv(lmudata_dir: str, dataset_name: str) -> str:
    # Prefer path-based local variants when present (e.g. POPE_local).
    candidates = [f"{dataset_name}.tsv"]
    if not dataset_name.endswith("_local"):
        candidates.insert(0, f"{dataset_name}_local.tsv")
    for name in candidates:
        path = os.path.join(lmudata_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"LMUData TSV not found for {dataset_name} under {lmudata_dir} "
        f"(tried {', '.join(candidates)})"
    )


def load_caption_samples(tsv_path: str) -> list[CaptionSample]:
    df = pd.read_csv(tsv_path, sep="\t")
    dataset_name = os.path.splitext(os.path.basename(tsv_path))[0]
    samples: list[CaptionSample] = []
    for _, row in df.iterrows():
        caps = _parse_list_cell(row.get("answer"))
        if not caps:
            continue
        image_path = _materialize_image(row, dataset_name)
        if not image_path:
            continue
        samples.append(
            CaptionSample(
                index=row.get("index", len(samples)),
                image_path=image_path,
                captions=caps,
            )
        )
    return samples


def load_vqa_samples(tsv_path: str) -> list[VQASample]:
    df = pd.read_csv(tsv_path, sep="\t")
    dataset_name = os.path.splitext(os.path.basename(tsv_path))[0]
    samples: list[VQASample] = []
    for _, row in df.iterrows():
        answers = _parse_list_cell(row.get("answer"))
        if not answers:
            continue
        image_path = _materialize_image(row, dataset_name)
        if not image_path:
            continue
        samples.append(
            VQASample(
                index=row.get("index", len(samples)),
                image_path=image_path,
                question=str(row["question"]).strip(),
                answers=answers,
            )
        )
    return samples


def load_pope_samples(tsv_path: str) -> list[ClassSample]:
    df = pd.read_csv(tsv_path, sep="\t")
    dataset_name = os.path.splitext(os.path.basename(tsv_path))[0]
    samples: list[ClassSample] = []
    for _, row in df.iterrows():
        answers = _parse_list_cell(row.get("answer"))
        label = (answers[0] if answers else str(row.get("answer", ""))).strip().lower()
        if label not in ("yes", "no"):
            label = str(row.get("answer", "")).strip().lower()
        if label not in ("yes", "no"):
            continue
        image_path = _materialize_image(row, dataset_name)
        if not image_path:
            continue
        samples.append(
            ClassSample(
                index=row.get("index", len(samples)),
                image_path=image_path,
                question=str(row["question"]).strip(),
                label=label,
                candidates=["yes", "no"],
            )
        )
    return samples


def split_demo_eval(
    samples: list,
    *,
    num_shots: int,
    max_samples: int | None,
    seed: int,
    demo_pool_size: int | None = None,
) -> tuple[list, list]:
    """Hold out a demo pool; evaluate on a disjoint subsample."""
    rng = random.Random(seed)
    order = list(range(len(samples)))
    rng.shuffle(order)
    ordered = [samples[i] for i in order]

    # Reserve demos for few-shot and for Flamingo 0-shot text-only format examples.
    if demo_pool_size is not None:
        demo_n = demo_pool_size
    elif num_shots <= 0:
        demo_n = 200
    else:
        demo_n = max(num_shots * 20, 200)
    demo_n = min(demo_n, max(0, len(ordered) - 1))

    demos = ordered[:demo_n]
    rest = ordered[demo_n:]
    if max_samples is not None and max_samples > 0:
        rest = rest[:max_samples]
    return demos, rest
