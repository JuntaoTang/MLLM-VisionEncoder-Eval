#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods import METHODS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-global", type=Path, required=True)
    parser.add_argument("--visual-patches", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--config", default="{}", help="JSON mapping from method name to config")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configs = json.loads(args.config)
    if not isinstance(configs, dict):
        raise ValueError("--config must decode to a JSON object")
    global_visual = np.load(args.visual_global, mmap_mode="r", allow_pickle=False)
    patch_visual = np.load(args.visual_patches, mmap_mode="r", allow_pickle=False)
    text = np.load(args.text, mmap_mode="r", allow_pickle=False)
    results = {}
    for name, method in METHODS.items():
        visual = patch_visual if name == "ravel" else global_visual
        results[name] = method().evaluate(visual, text, dict(configs.get(name, {})))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "success", "methods": list(results), "output": str(args.output)}))


if __name__ == "__main__":
    main()
