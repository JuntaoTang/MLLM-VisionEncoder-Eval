"""Format and print brief per-dataset eval scores (accuracy / Bleu0)."""

from __future__ import annotations

from typing import Callable


def format_dataset_score(summary: dict | None) -> str:
    """Return a short score string: accuracy %, Bleu0, or FAILED."""
    if not summary:
        return "n/a"

    if summary.get("status") == "failed":
        err = summary.get("error")
        return f"FAILED ({err})" if err else "FAILED"

    metric = summary.get("metric")
    primary = summary.get("primary_score")
    primary_metric = summary.get("primary_metric")

    if metric == "deferred_llm_judge":
        return "infer_ok (judge pending)"

    if metric == "mme_perception_reasoning":
        p = summary.get("perception")
        r = summary.get("reasoning")
        if isinstance(p, (int, float)) and isinstance(r, (int, float)):
            # Official MME reports AVG as perception + reasoning (max 2800).
            return f"AVG={p + r:.1f} (perception={p:.1f} reasoning={r:.1f})"
        return (
            f"perception={summary.get('perception', 'n/a')} "
            f"reasoning={summary.get('reasoning', 'n/a')}"
        )

    if metric == "caption_bleu0" or primary_metric == "bleu0":
        bleu0 = summary.get("bleu0")
        if isinstance(bleu0, (int, float)):
            return f"Bleu0={bleu0:.2f}"
        return summary.get("error") or "n/a"

    if metric == "caption_cider" or primary_metric == "cider":
        cider = summary.get("accuracy")
        if isinstance(cider, (int, float)):
            # Stored as fraction (CIDEr/100) or already percent-like.
            val = cider * 100.0 if cider <= 1.0 else cider
            return f"CIDEr={val:.2f}"
        return summary.get("error") or "n/a"

    if primary_metric == "accuracy" and isinstance(primary, (int, float)):
        return f"{primary:.2%}"

    acc = summary.get("accuracy")
    if isinstance(acc, (int, float)):
        return f"{acc:.2%}"

    return summary.get("error") or "n/a"


def _format_time_range(started_at: str | None, finished_at: str | None) -> str:
    if started_at and finished_at:
        return f"{started_at} ~ {finished_at}"
    return started_at or finished_at or "n/a"


def build_brief_summary(
    *,
    run_slug: str,
    dataset_summaries: dict[str, dict],
    started_at: str | None = None,
    finished_at: str | None = None,
    dataset_order: list[str] | None = None,
    failed: list[str] | None = None,
    extra: dict | None = None,
) -> dict:
    """Build a compact summary dict for summary.json (model, time, per-dataset scores)."""
    from vision_encoder_eval.mllm.utils.finish_tracker import concise_scores

    order = dataset_order or list(dataset_summaries.keys())
    scores: dict[str, str] = {}
    for name in order:
        scores[name] = format_dataset_score(dataset_summaries.get(name))

    numeric = concise_scores(dataset_summaries, dataset_order=order)
    avg = numeric.get("average")
    if isinstance(avg, (int, float)):
        scores["average"] = f"{avg:.2f}"

    time_range = _format_time_range(started_at, finished_at)
    lines = [
        f"model: {run_slug}",
        f"time: {time_range}",
    ]
    for name in order:
        lines.append(f"{name}: {scores[name]}")
    if isinstance(avg, (int, float)):
        lines.append(f"average: {avg:.2f}")
    if failed:
        lines.append(f"failed: {', '.join(failed)}")

    payload = {
        "model": run_slug,
        "started_at": started_at,
        "finished_at": finished_at,
        "scores": scores,
        "failed": list(failed or []),
        "text": "\n".join(lines),
    }
    if extra:
        payload.update(extra)
    return payload


def print_eval_summary(
    *,
    run_slug: str,
    dataset_summaries: dict[str, dict],
    dataset_order: list[str] | None = None,
    print_fn: Callable[..., None] = print,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> None:
    """Print a compact table of per-dataset scores after eval finishes."""
    order = dataset_order or list(dataset_summaries.keys())
    if not order:
        return

    print_fn("")
    print_fn("=" * 60)
    print_fn(f"Eval summary: {run_slug}")
    if started_at or finished_at:
        print_fn(f"Time: {_format_time_range(started_at, finished_at)}")
    print_fn("=" * 60)
    width = max(len(name) for name in order)
    for name in order:
        score = format_dataset_score(dataset_summaries.get(name))
        print_fn(f"  {name:<{width}}  {score}")
    print_fn("")
