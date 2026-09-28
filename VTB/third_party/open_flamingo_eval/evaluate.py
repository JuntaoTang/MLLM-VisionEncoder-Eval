"""OpenFlamingo-style evaluation loops."""

from __future__ import annotations

import os
import random
from typing import Any, Sequence

from PIL import Image
from tqdm import tqdm

from .base import BaseOFEvalModel
from .datasets import (
    CaptionSample,
    ClassSample,
    VQASample,
    load_caption_samples,
    load_pope_samples,
    load_vqa_samples,
    resolve_tsv,
    split_demo_eval,
)
from .fewshot import sample_demos
from .metrics import compute_cider, strip_generation, vqa_accuracy
from .prompts import caption_prompt, pope_prompt


def _open_images(paths: Sequence[str]) -> list[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]


def evaluate_captioning(
    model: BaseOFEvalModel,
    *,
    lmudata_dir: str,
    dataset_name: str,
    num_shots: int = 4,
    max_samples: int | None = 1000,
    seed: int = 42,
    max_new_tokens: int = 20,
    num_beams: int = 3,
    zero_shot_text_demos: int = 2,
    demo_tsv: str | None = None,
) -> dict[str, Any]:
    eval_tsv = resolve_tsv(lmudata_dir, dataset_name)
    eval_all = load_caption_samples(eval_tsv)
    if demo_tsv:
        demo_pool = load_caption_samples(demo_tsv)
        _, eval_samples = split_demo_eval(
            eval_all, num_shots=0, max_samples=max_samples, seed=seed
        )
    else:
        demo_pool, eval_samples = split_demo_eval(
            eval_all, num_shots=num_shots, max_samples=max_samples, seed=seed
        )

    preds: dict[str, list[str]] = {}
    refs: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []

    for i, sample in enumerate(tqdm(eval_samples, desc=f"caption/{dataset_name}/{num_shots}shot")):
        demos = sample_demos(demo_pool, num_shots, seed=seed + i, exclude_index=sample.index)
        captions: list[str | None] = [d.captions[0] for d in demos] + [None]
        include_images = [True] * len(demos) + [True]
        image_paths = [d.image_path for d in demos] + [sample.image_path]

        # Flamingo 0-shot: keep text-only format demos (no images).
        if num_shots == 0 and zero_shot_text_demos > 0 and demo_pool:
            text_demos = sample_demos(demo_pool, zero_shot_text_demos, seed=seed + i)
            captions = [d.captions[0] for d in text_demos] + [None]
            include_images = [False] * len(text_demos) + [True]
            image_paths = [sample.image_path]

        prompt = caption_prompt(captions, include_images=include_images)
        images = _open_images(image_paths)
        raw = model.generate(
            images,
            prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            stop_strings=["<|endofchunk|>", "\n"],
        )
        pred = strip_generation(raw)
        key = str(sample.index)
        preds[key] = [pred]
        refs[key] = sample.captions
        rows.append({"index": sample.index, "prediction": pred, "references": sample.captions})

    cider = compute_cider(preds, refs) if preds else 0.0
    return {
        "dataset": dataset_name,
        "task": "captioning",
        "num_shots": num_shots,
        "num_samples": len(eval_samples),
        "metric": "cider",
        "score": cider,
        "cider": cider,
        "predictions": rows,
    }


def evaluate_vqa(
    model: BaseOFEvalModel,
    *,
    lmudata_dir: str,
    dataset_name: str,
    num_shots: int = 4,
    max_samples: int | None = 1000,
    seed: int = 42,
    max_new_tokens: int = 5,
    num_beams: int = 3,
    zero_shot_text_demos: int = 2,
    demo_tsv: str | None = None,
) -> dict[str, Any]:
    from .prompts import vqa_prompt

    eval_tsv = resolve_tsv(lmudata_dir, dataset_name)
    eval_all = load_vqa_samples(eval_tsv)
    if demo_tsv:
        demo_pool = load_vqa_samples(demo_tsv)
        _, eval_samples = split_demo_eval(
            eval_all, num_shots=0, max_samples=max_samples, seed=seed
        )
    else:
        demo_pool, eval_samples = split_demo_eval(
            eval_all, num_shots=num_shots, max_samples=max_samples, seed=seed
        )

    total = 0.0
    rows: list[dict[str, Any]] = []
    for i, sample in enumerate(tqdm(eval_samples, desc=f"vqa/{dataset_name}/{num_shots}shot")):
        demos = sample_demos(demo_pool, num_shots, seed=seed + i, exclude_index=sample.index)
        qa_pairs: list[tuple[str, str | None]] = [
            (d.question, d.answers[0]) for d in demos
        ] + [(sample.question, None)]
        include_images = [True] * len(demos) + [True]
        image_paths = [d.image_path for d in demos] + [sample.image_path]

        if num_shots == 0 and zero_shot_text_demos > 0 and demo_pool:
            text_demos = sample_demos(demo_pool, zero_shot_text_demos, seed=seed + i)
            qa_pairs = [(d.question, d.answers[0]) for d in text_demos] + [
                (sample.question, None)
            ]
            include_images = [False] * len(text_demos) + [True]
            image_paths = [sample.image_path]

        prompt = vqa_prompt(qa_pairs, include_images=include_images)
        images = _open_images(image_paths)
        raw = model.generate(
            images,
            prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            stop_strings=["<|endofchunk|>", "\n"],
        )
        pred = strip_generation(raw)
        acc = vqa_accuracy(pred, sample.answers)
        total += acc
        rows.append(
            {
                "index": sample.index,
                "question": sample.question,
                "prediction": pred,
                "answers": sample.answers,
                "accuracy": acc,
            }
        )

    n = max(len(eval_samples), 1)
    score = total / n
    return {
        "dataset": dataset_name,
        "task": "vqa",
        "num_shots": num_shots,
        "num_samples": len(eval_samples),
        "metric": "vqa_accuracy",
        "score": score,
        "accuracy": score,
        "predictions": rows,
    }


def evaluate_classification(
    model: BaseOFEvalModel,
    *,
    lmudata_dir: str,
    dataset_name: str = "POPE",
    num_shots: int = 4,
    max_samples: int | None = 1000,
    seed: int = 42,
    zero_shot_text_demos: int = 2,
    demo_tsv: str | None = None,
) -> dict[str, Any]:
    """POPE-style yes/no via log-prob ranking (OpenFlamingo classification path)."""
    eval_tsv = resolve_tsv(lmudata_dir, dataset_name)
    eval_all = load_pope_samples(eval_tsv)
    if demo_tsv:
        demo_pool = load_pope_samples(demo_tsv)
        _, eval_samples = split_demo_eval(
            eval_all, num_shots=0, max_samples=max_samples, seed=seed
        )
    else:
        demo_pool, eval_samples = split_demo_eval(
            eval_all, num_shots=num_shots, max_samples=max_samples, seed=seed
        )

    correct = 0
    rows: list[dict[str, Any]] = []
    for i, sample in enumerate(
        tqdm(eval_samples, desc=f"cls/{dataset_name}/{num_shots}shot")
    ):
        demos = sample_demos(demo_pool, num_shots, seed=seed + i, exclude_index=sample.index)
        qa_pairs: list[tuple[str, str | None]] = [
            (d.question, d.label) for d in demos
        ] + [(sample.question, None)]
        include_images = [True] * len(demos) + [True]
        image_paths = [d.image_path for d in demos] + [sample.image_path]

        if num_shots == 0 and zero_shot_text_demos > 0 and demo_pool:
            text_demos = sample_demos(demo_pool, zero_shot_text_demos, seed=seed + i)
            qa_pairs = [(d.question, d.label) for d in text_demos] + [(sample.question, None)]
            include_images = [False] * len(text_demos) + [True]
            image_paths = [sample.image_path]

        prompt = pope_prompt(qa_pairs, include_images=include_images)
        images = _open_images(image_paths)
        scores = model.score_completions(images, prompt, sample.candidates)
        pred = sample.candidates[int(max(range(len(scores)), key=lambda j: scores[j]))]
        ok = int(pred == sample.label)
        correct += ok
        rows.append(
            {
                "index": sample.index,
                "question": sample.question,
                "label": sample.label,
                "prediction": pred,
                "scores": dict(zip(sample.candidates, scores)),
                "correct": ok,
            }
        )

    n = max(len(eval_samples), 1)
    acc = correct / n
    return {
        "dataset": dataset_name,
        "task": "classification",
        "num_shots": num_shots,
        "num_samples": len(eval_samples),
        "metric": "accuracy",
        "score": acc,
        "accuracy": acc,
        "predictions": rows,
    }


def evaluate_caption_rank(
    model: BaseOFEvalModel,
    *,
    lmudata_dir: str,
    dataset_name: str = "MSCOCO_KARPATHY_TEST",
    max_samples: int | None = 1000,
    seed: int = 42,
    num_negatives: int = 9,
    **_unused,
) -> dict[str, Any]:
    """Caption alignment via continuation log-prob ranking.

    Prompt is ``<image> Output:``; each candidate caption is scored as the
    continuation (length-normalized mean log-prob). Positive = one gold caption
    of the image; negatives = captions from other images.
    """
    from .prompts import caption_rank_prompt

    eval_tsv = resolve_tsv(lmudata_dir, dataset_name)
    eval_all = load_caption_samples(eval_tsv)
    _, eval_samples = split_demo_eval(
        eval_all, num_shots=0, max_samples=max_samples, seed=seed
    )
    # Negatives drawn from full pool (exclude query image).
    pool = eval_all
    prompt = caption_rank_prompt()
    rng = random.Random(seed)

    ranks: list[int] = []
    hits_at_1 = 0
    hits_at_5 = 0
    rows: list[dict[str, Any]] = []

    for i, sample in enumerate(
        tqdm(eval_samples, desc=f"caption_rank/{dataset_name}")
    ):
        pos = sample.captions[0]
        neg_caps: list[str] = []
        # Sample negatives with replacement avoidance by index.
        candidates_idx = [j for j, s in enumerate(pool) if s.index != sample.index]
        rng.shuffle(candidates_idx)
        for j in candidates_idx:
            cap = pool[j].captions[0]
            if cap == pos:
                continue
            neg_caps.append(cap)
            if len(neg_caps) >= num_negatives:
                break
        while len(neg_caps) < num_negatives and pool:
            # Fallback if pool too small / too many duplicates.
            alt = pool[rng.randrange(len(pool))].captions[0]
            if alt != pos:
                neg_caps.append(alt)

        completions = [pos] + neg_caps[:num_negatives]
        images = _open_images([sample.image_path])
        scores = model.score_completions(
            images, prompt, completions, length_normalize=True
        )
        order = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
        rank = int(order.index(0)) + 1  # 1-based rank of positive
        ranks.append(rank)
        hits_at_1 += int(rank == 1)
        hits_at_5 += int(rank <= 5)
        rows.append(
            {
                "index": sample.index,
                "positive": pos,
                "rank": rank,
                "scores": {
                    "positive": scores[0],
                    "negatives": scores[1:],
                },
                "top1_is_positive": rank == 1,
            }
        )

    n = max(len(eval_samples), 1)
    r1 = hits_at_1 / n
    r5 = hits_at_5 / n
    mean_rank = sum(ranks) / n
    mrr = sum(1.0 / r for r in ranks) / n
    return {
        "dataset": dataset_name,
        "task": "caption_rank",
        "num_shots": 0,
        "num_samples": len(eval_samples),
        "num_negatives": num_negatives,
        "metric": "recall@1",
        "score": r1,
        "recall@1": r1,
        "recall@5": r5,
        "mean_rank": mean_rank,
        "mrr": mrr,
        "predictions": rows,
    }


TASK_DISPATCH = {
    "captioning": evaluate_captioning,
    "vqa": evaluate_vqa,
    "classification": evaluate_classification,
    "caption_rank": evaluate_caption_rank,
}

DEFAULT_DATASET_TASK = {
    "MSCOCO_KARPATHY_TEST": "captioning",
    "FLICKR30K_KARPATHY_TEST": "captioning",
    "COCO_VAL": "captioning",
    "VQAv2_VAL": "vqa",
    "TextVQA_VAL": "vqa",
    "GQA_TestDev_Balanced": "vqa",
    "POPE": "classification",
}


def evaluate_dataset(
    model: BaseOFEvalModel,
    dataset_name: str,
    *,
    lmudata_dir: str,
    task: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    task_name = task or DEFAULT_DATASET_TASK.get(dataset_name)
    if task_name is None:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}; pass task=captioning|vqa|classification|caption_rank"
        )
    fn = TASK_DISPATCH[task_name]
    return fn(model, lmudata_dir=lmudata_dir, dataset_name=dataset_name, **kwargs)
