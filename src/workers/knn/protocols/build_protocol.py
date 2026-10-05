#!/usr/bin/env python3
"""Build deterministic, nested ImageNet KNN protocols.

The generated JSON stores paths relative to the ImageFolder root, rather than
machine-specific integer positions only.  This makes the split stable across
machines while preserving the exact torchvision ImageFolder ordering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from torchvision.datasets import ImageFolder


def build(root: Path, output: Path, shots: tuple[int, ...], query_per_class: int, seed: int) -> None:
    dataset = ImageFolder(str(root))
    if len(dataset.classes) != 1000:
        raise ValueError(f"Expected ImageNet-1K (1000 classes), found {len(dataset.classes)}")
    largest = max(shots)
    rng = random.Random(seed)
    by_class: list[list[int]] = [[] for _ in dataset.classes]
    for index, (_, label) in enumerate(dataset.samples):
        by_class[label].append(index)

    train_by_shot = {str(shot): [] for shot in shots}
    query_indices: list[int] = []
    train_pool_indices: list[int] = []
    for label, indices in enumerate(by_class):
        if len(indices) < largest + query_per_class:
            raise ValueError(
                f"Class {dataset.classes[label]} has {len(indices)} images; "
                f"need at least {largest + query_per_class}"
            )
        ordered = list(indices)
        rng.shuffle(ordered)
        pool = ordered[:largest]
        query = ordered[largest : largest + query_per_class]
        train_pool_indices.extend(pool)
        query_indices.extend(query)
        for shot in shots:
            train_by_shot[str(shot)].extend(pool[:shot])

    relpaths = [str(Path(path).relative_to(root)).replace("\\", "/") for path, _ in dataset.samples]
    protocol = {
        "format_version": 1,
        "dataset": "ImageNet-1K",
        "dataset_split": root.name,
        "seed": seed,
        "num_classes": len(dataset.classes),
        "class_to_idx": dataset.class_to_idx,
        "train_shots": list(shots),
        "train_pool_per_class": largest,
        "query_per_class": query_per_class,
        "num_train_pool": len(train_pool_indices),
        "num_query": len(query_indices),
        "train_pool_indices": train_pool_indices,
        "train_indices_by_shot": train_by_shot,
        "query_indices": query_indices,
        "sample_relpaths_sha256": hashlib.sha256("\n".join(relpaths).encode()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, nargs="+", default=[5, 10, 20, 45, 95, 195])
    parser.add_argument("--query-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    shots = tuple(sorted(set(args.shots)))
    build(args.imagenet_root.resolve(), args.output.resolve(), shots, args.query_per_class, args.seed)


if __name__ == "__main__":
    main()
