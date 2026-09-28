"""CIDEr and VQA accuracy metrics."""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence


def normalize_answer(text: str) -> str:
    """Light VQA-style normalization."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def vqa_accuracy(prediction: str, answers: Sequence[str]) -> float:
    """Soft accuracy used by VQAv2 (min(#matches/3, 1))."""
    pred = normalize_answer(prediction)
    if not pred:
        return 0.0
    gt = [normalize_answer(a) for a in answers]
    counts = Counter(gt)
    # Standard soft score: for each GT leave-one-out, Acc = min(#other_matches/3, 1)
    # Equivalent average when using all annotators:
    match = counts.get(pred, 0)
    return min(float(match) / 3.0, 1.0)


def compute_cider(
    predictions: dict[str, list[str]],
    references: dict[str, list[str]],
) -> float:
    """CIDEr via pycocoevalcap. Keys must align (string ids)."""
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError as exc:
        raise ImportError(
            "pycocoevalcap is required for CIDEr. Install with: pip install pycocoevalcap"
        ) from exc

    scorer = Cider()
    score, _ = scorer.compute_score(references, predictions)
    return float(score)


def strip_generation(text: str, stop_strings: Sequence[str] | None = None) -> str:
    text = text.strip()
    stops = list(stop_strings or []) + ["<|endofchunk|>", "<|im_end|>", "\n"]
    for stop in stops:
        if stop and stop in text:
            text = text.split(stop, 1)[0]
    return text.strip().strip("\"'")
