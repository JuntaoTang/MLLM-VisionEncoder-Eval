#!/usr/bin/env python3
"""Evaluate one model's exported features on the common nested-shot protocol."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_encoder_eval.workers.knn.common.exact_knn import exact_weighted_knn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True, help="N x D array in ImageFolder order")
    p.add_argument("--labels", type=Path, required=True, help="N labels in the same order")
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--flops-per-image", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    features = np.load(args.features, mmap_mode="r")
    labels = np.load(args.labels, mmap_mode="r")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    shots = tuple(protocol["train_shots"])
    query_rows = np.asarray(protocol["query_indices"], dtype=np.int64)
    query = features[query_rows]
    query_labels = labels[query_rows]
    rows = []
    for shot in shots:
        train_rows = np.asarray(protocol["train_indices_by_shot"][str(shot)], dtype=np.int64)
        predictions = exact_weighted_knn(features[train_rows], labels[train_rows], query)
        top1 = float((predictions == query_labels).mean() * 100.0)
        search_flops = 2 * len(query_rows) * len(train_rows) * features.shape[1]
        feature_flops = (len(query_rows) + len(train_rows)) * args.flops_per_image
        rows.append({
            "Model": args.model,
            "Shot": shot,
            "Train": len(train_rows),
            "Top1": f"{top1:.2f}%",
            "FeatureExtractionFLOPs": int(feature_flops),
            "ExactKNNSearchFLOPs": int(search_flops),
            "TFLOPs": (feature_flops + search_flops) / 1e12,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print("| Model | Shot | Train | Top1 | TFLOPs |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(f"| {row['Model']} | {row['Shot']} | {row['Train']} | {row['Top1']} | {row['TFLOPs']:.6f} |")


if __name__ == "__main__":
    main()
