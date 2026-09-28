#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods import METHODS  # noqa: E402
from src.neighbors import binary_overlap  # noqa: E402


def main() -> None:
    expected = {"rsa", "cca", "ravel", "gw", "mutualnn"}
    if set(METHODS) != expected:
        raise AssertionError(f"method registry mismatch: {set(METHODS)}")
    rng = np.random.default_rng(7)
    visual = rng.normal(size=(30, 8)).astype(np.float32)
    text = visual[:, :6] + 0.1 * rng.normal(size=(30, 6)).astype(np.float32)
    rsa = METHODS["rsa"]().evaluate(visual, text, {})
    mutualnn = METHODS["mutualnn"]().evaluate(visual, text, {"k": 5})
    cca = METHODS["cca"]().evaluate(
        visual,
        text,
        {
            "device": "cpu",
            "components": 3,
            "alphas": [0.1, 1.0],
            "visual_id": "synthetic_visual",
            "text_id": "synthetic_text",
        },
    )
    gw = METHODS["gw"]().evaluate(
        visual[:12], text[:12], {"backend": "numpy", "max_iter": 30}
    )
    patches = rng.normal(size=(20, 3, 6)).astype(np.float32)
    ravel = METHODS["ravel"]().evaluate(
        patches,
        text[:20],
        {"k": 4, "device": "cpu", "whitening_eps": 1e-4},
    )
    manual = binary_overlap(
        np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int32),
        np.asarray([[2, 1], [2, 0], [1, 0]], dtype=np.int32),
    )
    checks = {
        "registry": set(METHODS) == expected,
        "rsa_finite": np.isfinite(rsa["final_score"]),
        "mutualnn_finite": np.isfinite(mutualnn["final_score"]),
        "cca_finite": np.isfinite(cca["final_score"]),
        "gw_finite": np.isfinite(gw["final_score"]),
        "ravel_finite": np.isfinite(ravel["final_score"]),
        "ravel_no_truncation": ravel["pca_truncation"] is False,
        "ravel_binary_overlap": ravel["overlap"] == "binary_topk_intersection_divided_by_k",
        "binary_overlap_reference": manual == 1.0,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    payload = {"complete": all(checks.values()), "checks": checks}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
