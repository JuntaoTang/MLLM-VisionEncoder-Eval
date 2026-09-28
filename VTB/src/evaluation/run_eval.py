"""Unified multi-dataset evaluation entry point (continuous)."""

from __future__ import annotations

import json
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from src.evaluation.dataset_config import (
    DatasetEvalSpec,
    format_dataset_plan,
    parse_eval_dataset_specs,
    validate_dataset_specs,
)
from src.evaluation.judge_config import (
    format_judge_summary,
    resolve_judge_profile,
    uses_llm_judge,
)
from src.evaluation.judge_manager import (
    ensure_local_judge,
    needs_judge_server,
    release_eval_gpu_resources,
)
from src.evaluation.results import (
    dataset_dir,
    predictions_path,
    score_predictions,
    write_run_summary,
)
from src.evaluation.runner import evaluate as run_vtb_dataset
from src.evaluation.summary_format import build_brief_summary, format_dataset_score, print_eval_summary
from src.evaluation.vlmeval_runner import plan_infer_judge_gpus, run_vlmeval_dataset
from src.utils.config import load_config, load_run_context
from src.utils.finish_tracker import record_finished

_print_lock = threading.Lock()


def _locked_print(*args, **kwargs) -> None:
    with _print_lock:
        print(*args, **kwargs)


def _append_log(log_path: str | None, text: str) -> None:
    if not log_path:
        return
    with _print_lock:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text)


def _build_dataset_summary(spec: DatasetEvalSpec, results_root: str, summary: dict) -> dict:
    rel_xlsx = os.path.relpath(
        predictions_path(results_root, spec.name),
        results_root,
    )
    payload = {
        "status": "ok",
        "max_samples": summary.get("max_samples", spec.max_samples),
        "predictions": rel_xlsx,
        "metric": summary.get("metric"),
        "judge": summary.get("judge"),
        "accuracy": summary.get("accuracy"),
        "correct": summary.get("correct"),
        "total": summary.get("total"),
        "cuda_device": summary.get("cuda_device"),
    }
    for extra_key in ("perception", "reasoning", "caption_metrics", "vlmeval_acc"):
        if extra_key in summary:
            payload[extra_key] = summary[extra_key]
    return payload


def _run_one_dataset(
    *,
    config_path: str,
    spec: DatasetEvalSpec,
    results_root: str,
    backend: str,
    log_path: str | None,
    eval_model: str | None,
    cuda_device: str | None,
    skip_scoring: bool,
) -> tuple[str, int, dict | None, str | None]:
    out_dir = dataset_dir(results_root, spec.name)
    _locked_print(f"=== {spec.name} -> {out_dir}/predictions.xlsx (gpu={cuda_device}) ===")
    _append_log(log_path, f"dataset={spec.name} max_samples={spec.max_samples} gpu={cuda_device}\n")

    try:
        if backend == "vtb":
            summary = run_vtb_dataset(
                config_path,
                dataset_name=spec.name,
                max_samples=spec.max_samples,
                result_dir=out_dir,
            )
            code = 0
        else:
            code, summary = run_vlmeval_dataset(
                config_path,
                dataset_name=spec.name,
                max_samples=spec.max_samples,
                result_dir=out_dir,
                log_path=log_path,
                eval_model=eval_model,
                cuda_device=cuda_device,
                skip_scoring=skip_scoring,
            )
    except Exception as exc:
        return spec.name, 1, None, str(exc)

    if code != 0:
        return spec.name, code, summary, (summary or {}).get("error")

    _locked_print(f"[{spec.name}] done  {format_dataset_score(summary)}\n")
    return spec.name, 0, summary, None


def _run_datasets_parallel(
    *,
    config_path: str,
    specs: list[DatasetEvalSpec],
    results_root: str,
    backend: str,
    log_path: str | None,
    eval_model: str | None,
    gpu_ids: list[str],
    max_infer_workers: int,
    skip_scoring: bool,
) -> tuple[dict, list[str]]:
    gpu_pool: queue.Queue[str] = queue.Queue()
    for gpu_id in gpu_ids:
        gpu_pool.put(gpu_id)

    dataset_summaries: dict = {}
    failures: list[str] = []
    workers = min(len(gpu_ids), len(specs), max(1, max_infer_workers))

    def _worker(spec: DatasetEvalSpec) -> tuple[str, int, dict | None, str | None]:
        gpu_id = gpu_pool.get()
        try:
            return _run_one_dataset(
                config_path=config_path,
                spec=spec,
                results_root=results_root,
                backend=backend,
                log_path=log_path,
                eval_model=eval_model,
                cuda_device=gpu_id,
                skip_scoring=skip_scoring,
            )
        finally:
            gpu_pool.put(gpu_id)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, spec): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            name, code, summary, error = future.result()
            if code != 0 or summary is None:
                failures.append(name)
                err = error or "evaluation failed"
                print(f"[{name}] FAILED  {err}")
                dataset_summaries[name] = {
                    **(summary or {}),
                    "status": "failed",
                    "max_samples": spec.max_samples,
                    "error": err,
                }
                continue
            dataset_summaries[name] = _build_dataset_summary(spec, results_root, summary)

    return dataset_summaries, failures


def _score_deferred_llm_datasets(
    *,
    specs: list[DatasetEvalSpec],
    results_root: str,
    eval_cfg: dict,
    dataset_summaries: dict,
) -> None:
    """After infer frees GPUs, load local judge and score deferred MCQ sets."""
    deferred = [
        spec
        for spec in specs
        if uses_llm_judge(spec.name)
        and dataset_summaries.get(spec.name, {}).get("status") == "ok"
        and dataset_summaries[spec.name].get("metric") == "deferred_llm_judge"
    ]
    if not deferred:
        return

    print(
        f"\n=== LLM judge scoring phase ({len(deferred)} dataset(s), "
        f"{format_judge_summary(eval_cfg)}) ==="
    )
    ensure_local_judge(eval_cfg)
    for spec in deferred:
        xlsx = predictions_path(results_root, spec.name)
        if not os.path.isfile(xlsx):
            dataset_summaries[spec.name] = {
                **dataset_summaries.get(spec.name, {}),
                "status": "failed",
                "error": f"missing predictions for judge scoring: {xlsx}",
            }
            continue
        print(f"=== score {spec.name} with LLM judge ===")
        try:
            scored = score_predictions(xlsx, spec.name, eval_cfg=eval_cfg)
            dataset_summaries[spec.name].update(scored)
            dataset_summaries[spec.name]["status"] = "ok"
            print(f"[{spec.name}] scored  {format_dataset_score(scored)}\n")
        except Exception as exc:
            dataset_summaries[spec.name] = {
                **dataset_summaries.get(spec.name, {}),
                "status": "failed",
                "error": f"judge scoring failed: {exc}",
            }
            print(f"[{spec.name}] judge scoring FAILED: {exc}\n")


def run_evaluation(config_path: str, log_path: str | None = None, eval_model: str | None = None) -> int:
    eval_cfg: dict | None = None
    try:
        ctx = load_run_context(config_path, eval_model=eval_model)
        exp = load_config(config_path, eval_model=eval_model)
        eval_cfg = dict(ctx.eval or {})
        # Ensure judge scoring uses the same path-based LMUData as inference
        # (never ~/. /home/.../LMUData official downloads with numeric indices).
        eval_cfg.setdefault("lmudata_dir", exp.eval_data_dir)
        os.environ["LMUData"] = exp.eval_data_dir
        return _run_evaluation_inner(
            ctx=ctx,
            exp=exp,
            eval_cfg=eval_cfg,
            config_path=config_path,
            log_path=log_path,
            eval_model=eval_model,
        )
    finally:
        # Always free judge GPU before the next mllm's pretrain in run.py queue.
        release_eval_gpu_resources(eval_cfg)


def _run_evaluation_inner(
    *,
    ctx,
    exp,
    eval_cfg: dict,
    config_path: str,
    log_path: str | None,
    eval_model: str | None,
) -> int:
    backend = eval_cfg.get("backend", "vlmevalkit")

    specs = parse_eval_dataset_specs(eval_cfg)
    validate_dataset_specs(specs, exp.eval_data_dir)

    plan = format_dataset_plan(specs)
    infer_gpus, judge_gpus, schedule = plan_infer_judge_gpus(ctx, [s.name for s in specs])
    parallel = bool(eval_cfg.get("parallel_datasets", True))
    max_infer_workers = int(eval_cfg.get("max_infer_workers", len(infer_gpus) or 1))
    use_parallel = (
        parallel
        and backend == "vlmevalkit"
        and len(infer_gpus) > 1
        and len(specs) > 1
    )
    # Overlap: score LLM datasets as each finishes (judge GPU reserved).
    # Infer-then-judge: skip LLM scoring during infer, score after all GPUs free.
    skip_scoring = schedule == "infer_then_judge"

    ckpt_source = eval_cfg.get("checkpoint_source", "local_mllm")
    if ckpt_source == "registry":
        print(f"Eval checkpoint: registry model={ctx.registry_model_name}")
    else:
        use_ckpt = eval_cfg.get("use_checkpoint", "finetune")
        ckpt_root = ctx.finetune_dir if use_ckpt == "finetune" else ctx.pretrain_dir
        print(
            f"Eval checkpoint: local mllm={eval_cfg.get('checkpoint_mllm', ctx.mllm_recipe_name)} "
            f"({use_ckpt} -> {ckpt_root})"
        )
    print(f"Eval backend: {backend}")
    print(f"Datasets: {plan}")
    print(f"LMUData: {exp.eval_data_dir}")
    print(f"Results root: {exp.results_dir}")
    print(f"Judge (MCQ): {format_judge_summary(eval_cfg)}")
    print(f"GPU schedule: {schedule}")
    print(f"  Infer GPUs: [{', '.join(infer_gpus)}]")
    if judge_gpus:
        print(f"  Judge GPUs: [{', '.join(judge_gpus)}]")
    if use_parallel:
        workers = min(len(infer_gpus), len(specs), max(1, max_infer_workers))
        print(
            f"Eval parallel: up to {workers} concurrent datasets "
            f"(1 GPU each, max_infer_workers={max_infer_workers})"
        )
        if skip_scoring:
            print("  Mode infer_then_judge: run all infer first, then load judge for MMMU/MMBench.")
        elif needs_judge_server([s.name for s in specs], eval_cfg):
            print("  Mode overlap: judge reserved on leftover GPU(s); start lazily at score time.")
    else:
        print(f"Eval GPUs: sequential on {infer_gpus[0] if infer_gpus else '0'}")
    print()

    # Warm local judge early only when overlapping (GPU already reserved / idle).
    if (
        schedule == "overlap"
        and needs_judge_server([s.name for s in specs], eval_cfg)
        and bool(eval_cfg.get("preload_judge", False))
    ):
        ensure_local_judge(eval_cfg)

    started_at = datetime.now().isoformat()
    if use_parallel:
        dataset_summaries, failures = _run_datasets_parallel(
            config_path=config_path,
            specs=specs,
            results_root=exp.results_dir,
            backend=backend,
            log_path=log_path,
            eval_model=eval_model,
            gpu_ids=infer_gpus,
            max_infer_workers=max_infer_workers,
            skip_scoring=skip_scoring,
        )
    else:
        dataset_summaries = {}
        failures = []
        gpu_id = infer_gpus[0] if infer_gpus else "0"
        for spec in specs:
            name, code, summary, error = _run_one_dataset(
                config_path=config_path,
                spec=spec,
                results_root=exp.results_dir,
                backend=backend,
                log_path=log_path,
                eval_model=eval_model,
                cuda_device=gpu_id,
                skip_scoring=skip_scoring,
            )
            if code != 0 or summary is None:
                failures.append(name)
                dataset_summaries[name] = {
                    **(summary or {}),
                    "status": "failed",
                    "max_samples": spec.max_samples,
                    "error": error or "evaluation failed",
                }
                continue
            dataset_summaries[name] = _build_dataset_summary(spec, exp.results_dir, summary)

    # LLM-judge datasets are always deferred during infer; score them once GPUs free.
    _score_deferred_llm_datasets(
        specs=specs,
        results_root=exp.results_dir,
        eval_cfg=eval_cfg,
        dataset_summaries=dataset_summaries,
    )
    for name, payload in dataset_summaries.items():
        if payload.get("status") == "failed" and name not in failures:
            failures.append(name)

    finished_at = datetime.now().isoformat()
    judge_profile = resolve_judge_profile(eval_cfg)
    brief = build_brief_summary(
        run_slug=ctx.run_slug,
        dataset_summaries=dataset_summaries,
        started_at=started_at,
        finished_at=finished_at,
        dataset_order=[s.name for s in specs],
        failed=failures,
    )
    # Keep summary.json concise (model / time / per-dataset scores only).
    brief_out = {
        "model": brief["model"],
        "started_at": brief.get("started_at"),
        "finished_at": brief.get("finished_at"),
        "scores": brief["scores"],
        "failed": brief["failed"],
        "text": brief["text"],
    }
    detail = {
        "run_slug": ctx.run_slug,
        "backend": backend,
        "checkpoint": exp.checkpoint_dir,
        "lmudata_dir": exp.eval_data_dir,
        "started_at": started_at,
        "finished_at": finished_at,
        "parallel_datasets": use_parallel,
        "gpu_schedule": schedule,
        "eval_gpus": infer_gpus,
        "judge_gpus": judge_gpus,
        "judge": judge_profile.get("model"),
        "judge_model_path": judge_profile.get("model_path"),
        "failed": failures,
        "datasets": dataset_summaries,
    }
    os.makedirs(exp.results_dir, exist_ok=True)
    detail_path = os.path.join(exp.results_dir, "summary_detail.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)
    summary_file = write_run_summary(exp.results_dir, brief_out)
    print(f"Run summary: {summary_file}")
    print_eval_summary(
        run_slug=ctx.run_slug,
        dataset_summaries=dataset_summaries,
        dataset_order=[s.name for s in specs],
        started_at=started_at,
        finished_at=finished_at,
    )

    recipe = ctx.mllm_recipe_name or getattr(ctx, "run_slug", "")
    finish_path = record_finished(
        "continuous",
        recipe,
        dataset_summaries,
        run_slug=ctx.run_slug,
        dataset_order=[s.name for s in specs],
    )
    print(f"Recorded finish: {finish_path} ({recipe})")
    if failures:
        print(f"Evaluation failed for: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.evaluation.run_eval <config_path>")
        sys.exit(1)
    sys.exit(run_evaluation(sys.argv[1]))
