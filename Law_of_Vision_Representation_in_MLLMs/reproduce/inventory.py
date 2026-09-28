#!/usr/bin/env python3
"""Dump finish.json inventory: recipes, unique encoders, pretrain checkpoint status."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, atomic_json, inventory_models, unique_vision_ids  # noqa: E402


def main():
    rows = inventory_models()
    vids = unique_vision_ids(rows)
    n_ok = sum(1 for r in rows if r.get("pretrain_ok"))
    n_err = sum(1 for r in rows if r.get("error"))
    llm_c = Counter(r.get("llm_id") for r in rows if r.get("llm_id"))
    summary = {
        "n_models": len(rows),
        "n_pretrain_ok": n_ok,
        "n_missing_recipe": n_err,
        "n_unique_vision": len(vids),
        "llm_counts": dict(llm_c),
        "vision_ids": vids,
        "missing_pretrain": [r["key"] for r in rows if not r.get("pretrain_ok") and not r.get("error")],
        "models": rows,
    }
    atomic_json(RESULTS_DIR / "inventory.json", summary)
    print(f"models={len(rows)} pretrain_ok={n_ok} unique_vision={len(vids)} llms={dict(llm_c)}")
    if summary["missing_pretrain"]:
        print("missing_pretrain:", len(summary["missing_pretrain"]))
        for k in summary["missing_pretrain"][:20]:
            print(" ", k)


if __name__ == "__main__":
    main()
