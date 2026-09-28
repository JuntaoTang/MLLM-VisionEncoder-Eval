#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods import METHODS  # noqa: E402


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=tuple(METHODS))
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--config", default="{}", help="JSON object")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config)
    if not isinstance(config, dict):
        raise ValueError("--config must decode to a JSON object")
    visual = np.load(args.visual, mmap_mode="r", allow_pickle=False)
    text = np.load(args.text, mmap_mode="r", allow_pickle=False)
    result = METHODS[args.method]().evaluate(visual, text, config)
    payload = {"method": args.method, "result": result}
    rendered = json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
