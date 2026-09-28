"""Track finished train+eval runs in results/finish.json."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Any

from src.utils.config import VTB_ROOT

FINISH_JSON = os.path.join(VTB_ROOT, "results", "finish.json")
_LOCK = threading.Lock()

# Canonical per-dataset order in finish.json / brief Avg (matches runtime eval.datasets).
FINISH_SCORE_ORDER: tuple[str, ...] = (
    "MMMU_TEST",
    "MMBench_TEST_EN_V11",
    "VQAv2_VAL",
    "ScienceQA_VAL",
    "ChartQA_TEST",
    "DocVQA_VAL",
    "TextVQA_VAL",
    "POPE",
    "GQA_TestDev_Balanced",
    "MSCOCO_KARPATHY_TEST",
    "FLICKR30K_KARPATHY_TEST",
)


def _finish_key(mode: str, recipe: str) -> str:
    recipe = str(recipe).removesuffix(".yaml").strip()
    return f"{mode}/{recipe}"


def _load() -> dict[str, Any]:
    if not os.path.isfile(FINISH_JSON):
        return {}
    try:
        data = json.loads(open(FINISH_JSON, encoding="utf-8").read())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(FINISH_JSON), exist_ok=True)
    tmp = FINISH_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, FINISH_JSON)


def _ordered_score_keys(names: set[str], order: list[str] | None = None) -> list[str]:
    """Return score keys in fixed order; unknown datasets keep stable sorted tail."""
    preferred = list(order) if order else list(FINISH_SCORE_ORDER)
    out: list[str] = [n for n in preferred if n in names]
    remaining = sorted(names - set(out) - {"average"})
    out.extend(remaining)
    return out


def concise_scores(
    dataset_summaries: dict[str, Any],
    *,
    dataset_order: list[str] | None = None,
) -> dict[str, float]:
    """Map dataset -> accuracy(%) / CIDEr plus mean average; skip failed entries.

    Keys are emitted in FINISH_SCORE_ORDER (or ``dataset_order``), with ``average`` last.
    MME (absolute perception+reasoning scale) is omitted entirely — not evaluated
    in the current suite and not mixed into average.
    """
    raw: dict[str, float] = {}
    for name, info in (dataset_summaries or {}).items():
        if not isinstance(info, dict):
            continue
        if info.get("status") and info.get("status") != "ok":
            continue
        # Skip MME: different scale; not in the current eval suite.
        if str(name).upper() == "MME" or info.get("metric") == "mme_perception_reasoning":
            continue
        acc = info.get("accuracy")
        if not isinstance(acc, (int, float)):
            continue
        value = float(acc)
        # Prefer percent for display; leave already-scaled (>1) metrics as-is.
        scored = round(value * 100.0, 2) if value <= 1.0 else round(value, 2)
        raw[str(name)] = scored

    scores: dict[str, float] = {}
    for name in _ordered_score_keys(set(raw), dataset_order):
        scores[name] = raw[name]
    if scores:
        scores["average"] = round(sum(scores.values()) / len(scores), 2)
    return scores


def order_scores_dict(scores: dict[str, Any], *, dataset_order: list[str] | None = None) -> dict[str, Any]:
    """Reorder an existing scores mapping (keeps average last)."""
    if not isinstance(scores, dict) or not scores:
        return scores
    avg = scores.get("average")
    names = {k for k in scores if k != "average"}
    ordered: dict[str, Any] = {}
    for name in _ordered_score_keys(names, dataset_order):
        if name in scores:
            ordered[name] = scores[name]
    if "average" in scores:
        ordered["average"] = avg
    return ordered


def is_finished(mode: str, recipe: str) -> bool:
    key = _finish_key(mode, recipe)
    with _LOCK:
        entry = _load().get(key)
    return isinstance(entry, dict) and bool(entry.get("scores"))


def record_finished(
    mode: str,
    recipe: str,
    dataset_summaries: dict[str, Any],
    *,
    run_slug: str | None = None,
    dataset_order: list[str] | None = None,
) -> str:
    """Write/replace a concise completion entry. Returns the finish.json path."""
    scores = concise_scores(dataset_summaries, dataset_order=dataset_order)
    key = _finish_key(mode, recipe)
    entry = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "scores": scores,
    }
    if run_slug:
        entry["run_slug"] = run_slug
    with _LOCK:
        data = _load()
        data[key] = entry
        _save(data)
    return FINISH_JSON


def rewrite_finish_scores_order(*, dataset_order: list[str] | None = None) -> str:
    """Rewrite all finish.json score dicts into the canonical key order."""
    with _LOCK:
        data = _load()
        for entry in data.values():
            if isinstance(entry, dict) and isinstance(entry.get("scores"), dict):
                entry["scores"] = order_scores_dict(entry["scores"], dataset_order=dataset_order)
        _save(data)
    return FINISH_JSON
