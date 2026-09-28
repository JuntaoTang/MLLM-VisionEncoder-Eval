"""Helpers for mixing multiple instruction datasets with per-source sampling."""

from __future__ import annotations

import json
import math
import random
from typing import Any


def normalize_stage_datasets(stage_cfg: dict, root: str, join_root) -> list[dict]:
    """Return resolved dataset entries from ``datasets`` list or legacy single paths."""
    if "datasets" in stage_cfg:
        raw_entries = stage_cfg["datasets"]
    elif stage_cfg.get("data_path"):
        raw_entries = [{
            "data_path": stage_cfg["data_path"],
            "image_folder": stage_cfg.get("image_folder", ""),
            "sampling_strategy": stage_cfg.get("sampling_strategy", "all"),
        }]
    else:
        raise ValueError("Stage config must define `datasets` or `data_path`")

    if not raw_entries:
        raise ValueError("datasets must not be empty")

    resolved: list[dict] = []
    for entry in raw_entries:
        data_path = join_root(root, entry.get("data_path", ""))
        image_folder = join_root(root, entry.get("image_folder", ""))
        if not data_path:
            raise ValueError("Each dataset entry must define `data_path`")
        if not image_folder:
            raise ValueError(f"Dataset {data_path!r} must define `image_folder`")
        resolved.append({
            "data_path": data_path,
            "image_folder": image_folder,
            "sampling_strategy": str(entry.get("sampling_strategy", "all")),
        })
    return resolved


def load_llava_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON list in {path}")
    return raw


def apply_sampling_strategy(items: list[dict], strategy: str, *, seed: int = 42) -> list[dict]:
    """Apply LLaVA-style sampling strategies to a sample list."""
    if not items or strategy == "all":
        return list(items)

    strategy_name = strategy
    count: int | None = None
    if ":" in strategy:
        strategy_name, raw_count = strategy.split(":", 1)
        if "%" in raw_count:
            pct = float(raw_count.replace("%", ""))
            count = max(1, math.ceil(len(items) * pct / 100.0))
        else:
            count = int(raw_count)

    if strategy_name == "first" and count is not None:
        return list(items[:count])
    if strategy_name == "end" and count is not None:
        return list(items[-count:])
    if strategy_name == "random" and count is not None:
        rng = random.Random(seed)
        picked = list(items)
        rng.shuffle(picked)
        return picked[:count]
    raise ValueError(f"Unsupported sampling_strategy {strategy!r}")


def summarize_datasets(entries: list[dict]) -> str:
    if len(entries) <= 1:
        return entries[0]["data_path"] if entries else ""
    lines = [f"mixed ({len(entries)} datasets, random shuffle each epoch)"]
    for entry in entries:
        name = entry["data_path"].rsplit("/", 1)[-1]
        strategy = entry.get("sampling_strategy", "all")
        lines.append(f"  - {name} [{strategy}]")
    return "\n".join(lines)
