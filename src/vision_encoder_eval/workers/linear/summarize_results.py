#!/usr/bin/env python3
"""Collect 70 probe results and reproduce the paper correlation row."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
MINI_ROOT = HERE.parent


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator


def load_manifest(path: Path) -> list[dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        rank, label, model, _head = line.split("\t")
        rows.append({"rank": rank, "tokenizer": label, "model": model})
    if len(rows) != 70:
        raise RuntimeError(f"expected 70 manifest rows, found {len(rows)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=MINI_ROOT / "output" / "linear_probing_runs",
    )
    parser.add_argument(
        "--mllm-scores",
        type=Path,
        default=MINI_ROOT / "output" / "mllm_scores.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=MINI_ROOT / "output")
    args = parser.parse_args()

    manifest = load_manifest(HERE / "tokenizers.tsv")
    with args.mllm_scores.open(newline="", encoding="utf-8") as handle:
        scores = {row["tokenizer"]: row for row in csv.DictReader(handle)}

    result_rows = []
    for row in manifest:
        result_path = args.results_root / row["tokenizer"] / "cap_0005" / "results_eval_linear.json"
        # The checked-in paper archive is flattened to keep ``mini/output``
        # small and reviewable; fresh runs use the directory layout above.
        if not result_path.is_file():
            result_path = args.results_root / f'{row["tokenizer"]}.json'
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        best = payload["best_classifier"]
        result_rows.append(
            {
                **row,
                "top1": best["accuracy"],
                "top5": best["top5_accuracy"],
                "base_lr": best["base_lr"],
                "effective_lr": best["effective_lr"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = args.output_dir / "linear_probing_5shot_results.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)

    probe = [float(row["top1"]) for row in result_rows]
    correlation_rows = []
    for column, display_name in (
        ("qwen3_1.7b", "Qwen3-1.7B"),
        ("qwen2.5_1.5b", "Qwen2.5-1.5B"),
        ("smollm2_1.7b", "SmolLM2-1.7B"),
    ):
        target = [float(scores[row["tokenizer"]][column]) for row in result_rows]
        correlation_rows.append(
            {
                "language_model": display_name,
                "n": len(target),
                "spearman_rho": pearson(average_ranks(probe), average_ranks(target)),
                "pearson_r": pearson(probe, target),
            }
        )
    correlations_csv = args.output_dir / "linear_probing_correlations.csv"
    with correlations_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=correlation_rows[0].keys())
        writer.writeheader()
        writer.writerows(correlation_rows)
    print(results_csv)
    print(correlations_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
