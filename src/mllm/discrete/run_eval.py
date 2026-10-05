"""Unified multi-dataset evaluation entry point."""

from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from vision_encoder_eval.mllm.evaluation.dataset_config import (
    DatasetEvalSpec,
    ensure_lmudata_image_paths,
    format_dataset_plan,
    parse_eval_dataset_specs,
    validate_dataset_specs,
)
from vision_encoder_eval.mllm.evaluation.results import (
    dataset_dir,
    predictions_path,
    score_predictions,
    write_run_summary,
)
from vision_encoder_eval.mllm.evaluation.judge_config import (
    format_judge_summary,
    parse_judge_gpu_ids,
    resolve_judge_profile,
    resolve_sample_num,
    resolve_sample_seed,
    resolve_sampling_mode,
    uses_api_judge,
    uses_llm_judge,
)
from vision_encoder_eval.mllm.evaluation.judge_manager import (
    ensure_local_judge,
    needs_judge_server,
    release_eval_gpu_resources,
)
from vision_encoder_eval.mllm.evaluation.summary_format import (
    build_brief_summary,
    format_dataset_score,
    print_eval_summary,
)
from vision_encoder_eval.mllm.discrete.vlmeval_runner import parse_all_eval_gpu_ids, parse_eval_gpu_ids, run_vlmeval_dataset
from vision_encoder_eval.mllm.discrete.config import load_config, load_run_context

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


def _build_dataset_summary(
    spec: DatasetEvalSpec,
    results_root: str,
    summary: dict,
) -> dict:
    rel_xlsx = os.path.relpath(
        predictions_path(results_root, spec.name),
        results_root,
    )
    payload = {
        "status": "ok",
        "max_samples": summary.get("max_samples", spec.max_samples),
        "predictions": rel_xlsx,
        "metric": summary.get("metric"),
        "primary_metric": summary.get("primary_metric"),
        "primary_score": summary.get("primary_score"),
        "judge": summary.get("judge"),
        "sampling": summary.get("sampling"),
        "sample_seed": summary.get("sample_seed"),
        "accuracy": summary.get("accuracy"),
        "correct": summary.get("correct"),
        "total": summary.get("total"),
    }
    for extra_key in (
        "perception",
        "reasoning",
        "caption_metrics",
        "vlmeval_acc",
        "cider_vlmeval",
        "bleu0",
    ):
        if extra_key in summary:
            payload[extra_key] = summary[extra_key]
    if isinstance(summary.get("caption_metrics"), dict):
        payload["caption_metrics"] = summary["caption_metrics"]
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
) -> tuple[str, int, dict | None, str | None]:
    """Run one dataset on one GPU. Returns (name, code, summary, error)."""
    out_dir = dataset_dir(results_root, spec.name)
    _locked_print(f"=== {spec.name} -> {out_dir}/predictions.xlsx ===")
    _append_log(log_path, f"dataset={spec.name} max_samples={spec.max_samples} gpu={cuda_device}\n")

    try:
        if backend == "vtb":
            raise ValueError(
                "VTB-Discrete only supports eval.backend=vlmevalkit "
                "(set vlm_class: VTB_Discrete_VLM in configs/runtime.yaml)."
            )
        code, summary = run_vlmeval_dataset(
                config_path,
                dataset_name=spec.name,
                max_samples=spec.max_samples,
                result_dir=out_dir,
                log_path=log_path,
                eval_model=eval_model,
                cuda_device=cuda_device,
        )
    except Exception as exc:
        return spec.name, 1, None, str(exc)

    if code != 0:
        return spec.name, code, summary, summary.get("error") if summary else None

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
    max_infer_workers: int = 2,
) -> tuple[dict, list[str]]:
    gpu_pool: queue.Queue[str] = queue.Queue()
    for gpu_id in gpu_ids:
        gpu_pool.put(gpu_id)

    dataset_summaries: dict = {}
    failures: list[str] = []

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
            )
        finally:
            gpu_pool.put(gpu_id)

    with ThreadPoolExecutor(
        max_workers=min(len(gpu_ids), len(specs), max(1, max_infer_workers))
    ) as executor:
        futures = {executor.submit(_worker, spec): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            name, code, summary, error = future.result()
            if code != 0 or summary is None:
                failures.append(name)
                dataset_summaries[name] = {
                    **(summary or {}),
                    "status": "failed",
                    "max_samples": spec.max_samples,
                    "error": error or "evaluation failed",
                }
                continue
            dataset_summaries[name] = _build_dataset_summary(spec, results_root, summary)

    return dataset_summaries, failures


def _run_datasets_sequential(
    *,
    config_path: str,
    specs: list[DatasetEvalSpec],
    results_root: str,
    backend: str,
    log_path: str | None,
    eval_model: str | None,
    gpu_ids: list[str],
) -> tuple[dict, list[str]]:
    dataset_summaries: dict = {}
    failures: list[str] = []
    gpu_id = gpu_ids[0] if gpu_ids else "0"

    for spec in specs:
        name, code, summary, error = _run_one_dataset(
            config_path=config_path,
            spec=spec,
            results_root=results_root,
            backend=backend,
            log_path=log_path,
            eval_model=eval_model,
            cuda_device=gpu_id,
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
        dataset_summaries[name] = _build_dataset_summary(spec, results_root, summary)

    return dataset_summaries, failures


def _score_deferred_llm_datasets(
    *,
    specs: list[DatasetEvalSpec],
    results_root: str,
    eval_cfg: dict,
    dataset_summaries: dict,
) -> None:
    """After infer, load local judge once and score deferred MMMU/MMBench."""
    deferred = [
        spec
        for spec in specs
        if uses_llm_judge(spec.name)
        and dataset_summaries.get(spec.name, {}).get("status") == "ok"
        and dataset_summaries[spec.name].get("metric") == "deferred_llm_judge"
    ]
    if not deferred:
        return

    _locked_print(
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
        _locked_print(f"=== score {spec.name} with LLM judge ===")
        try:
            scored = score_predictions(xlsx, spec.name, eval_cfg=eval_cfg)
            dataset_summaries[spec.name].update(scored)
            dataset_summaries[spec.name]["status"] = "ok"
            _locked_print(f"[{spec.name}] scored  {format_dataset_score(scored)}\n")
        except Exception as exc:
            dataset_summaries[spec.name] = {
                **dataset_summaries.get(spec.name, {}),
                "status": "failed",
                "error": f"judge scoring failed: {exc}",
            }
            _locked_print(f"[{spec.name}] judge scoring FAILED: {exc}\n")


def run_evaluation(config_path: str, log_path: str | None = None, eval_model: str | None = None) -> int:
    eval_cfg: dict | None = None
    try:
        ctx = load_run_context(config_path, eval_model=eval_model)
        exp = load_config(config_path, eval_model=eval_model)
        eval_cfg = dict(ctx.eval or {})
        return _run_evaluation_inner(
            ctx=ctx,
            exp=exp,
            eval_cfg=eval_cfg,
            config_path=config_path,
            log_path=log_path,
            eval_model=eval_model,
        )
    finally:
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
    gpu_ids = parse_eval_gpu_ids(ctx)
    parallel = eval_cfg.get("parallel_datasets", True)
    use_parallel = (
        parallel
        and backend == "vlmevalkit"
        and len(gpu_ids) > 1
        and len(specs) > 1
    )

    ckpt_source = eval_cfg.get("checkpoint_source", "local_mllm")
    if ckpt_source == "registry":
        print(f"Eval checkpoint: registry model={ctx.registry_model_name}")
    else:
        use_ckpt = eval_cfg.get("use_checkpoint", "finetune")
        ckpt_root = ctx.finetune_dir if use_ckpt == "finetune" else ctx.pretrain_dir
        print(
            f"Eval checkpoint: local recipe={eval_cfg.get('checkpoint_recipe', ctx.recipe_name)} "
            f"({use_ckpt} -> {ckpt_root})"
        )
    print(f"Eval backend: {backend}")
    print(f"Datasets: {plan}")
    sampling_mode = resolve_sampling_mode(eval_cfg)
    sample_seed = resolve_sample_seed(eval_cfg)
    global_samples = resolve_sample_num(eval_cfg, None)
    judge_profile = resolve_judge_profile(eval_cfg)
    judge_gpu_ids = parse_judge_gpu_ids(eval_cfg, all_eval_gpus=parse_all_eval_gpu_ids(ctx))
    print(f"Sampling: {sampling_mode}, max_samples={global_samples}, seed={sample_seed}")
    print(f"Judge (MCQ): {format_judge_summary(eval_cfg)}")
    if needs_judge_server([spec.name for spec in specs], eval_cfg):
        if uses_api_judge(eval_cfg):
            print("  MCQ judge: cloud API (starts on first MMMU/MMBench score).")
        else:
            print(
                f"  MCQ judge: local {judge_profile['model']} on GPUs "
                f"[{', '.join(judge_gpu_ids)}] "
                f"(deferred until after parallel infer)."
            )
            print(f"  Model infer GPUs: [{', '.join(gpu_ids)}]")
    ensure_lmudata_image_paths(exp.eval_data_dir, [spec.name for spec in specs])
    print(f"LMUData: {exp.eval_data_dir}")
    print(f"Results root: {exp.results_dir}")
    max_infer_workers = int(eval_cfg.get("max_infer_workers", 2))
    if use_parallel:
        workers = min(len(gpu_ids), len(specs), max(1, max_infer_workers))
        print(
            f"Eval parallel: up to {workers} concurrent model loads on GPUs "
            f"[{', '.join(gpu_ids)}] (max_infer_workers={max_infer_workers})"
        )
    else:
        print(f"Eval GPUs: {gpu_ids[0] if gpu_ids else '0'} (sequential)\n")

    if use_parallel:
        print()
    started_at = datetime.now().isoformat()

    def _run_datasets() -> tuple[dict, list[str]]:
        if use_parallel:
            return _run_datasets_parallel(
                config_path=config_path,
                specs=specs,
                results_root=exp.results_dir,
                backend=backend,
                log_path=log_path,
                eval_model=eval_model,
                gpu_ids=gpu_ids,
                max_infer_workers=max_infer_workers,
            )
        return _run_datasets_sequential(
            config_path=config_path,
            specs=specs,
            results_root=exp.results_dir,
            backend=backend,
            log_path=log_path,
            eval_model=eval_model,
            gpu_ids=gpu_ids,
        )

    dataset_summaries, failures = _run_datasets()
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
    dataset_order = [spec.name for spec in specs]
    brief_summary = build_brief_summary(
        run_slug=ctx.run_slug,
        dataset_summaries=dataset_summaries,
        started_at=started_at,
        finished_at=finished_at,
        dataset_order=dataset_order,
        failed=failures,
    )
    payload = {
        "run_slug": ctx.run_slug,
        "brief_summary": brief_summary,
        "backend": backend,
        "checkpoint": exp.checkpoint_dir,
        "lmudata_dir": exp.eval_data_dir,
        "eval_protocol": "reference_benchmark",
        "sampling": sampling_mode,
        "max_samples": global_samples,
        "sample_seed": sample_seed,
        "judge_mode": "local",
        "judge": judge_profile["model"],
        "judge_base_url": judge_profile.get("base_url"),
        "judge_model_path": judge_profile.get("model_path"),
        "started_at": started_at,
        "finished_at": finished_at,
        "parallel_datasets": use_parallel,
        "eval_gpus": gpu_ids,
        "judge_gpus": judge_gpu_ids if needs_judge_server([spec.name for spec in specs], eval_cfg) else [],
        "failed": failures,
        "datasets": dataset_summaries,
    }
    summary_file = write_run_summary(exp.results_dir, payload)
    print_eval_summary(
        run_slug=ctx.run_slug,
        dataset_summaries=dataset_summaries,
        dataset_order=dataset_order,
        print_fn=_locked_print,
        started_at=started_at,
        finished_at=finished_at,
    )
    print(f"Run summary: {summary_file}")

    try:
        from vision_encoder_eval.mllm.utils.wandb_eval import maybe_log_eval_wandb

        maybe_log_eval_wandb(ctx, payload)
    except ImportError:
        pass

    if failures:
        print(f"Evaluation failed for: {', '.join(failures)}")
        return 1

    from vision_encoder_eval.mllm.utils.finish_tracker import record_finished

    recipe = ctx.recipe_name or getattr(ctx, "run_slug", "")
    finish_path = record_finished(
        "discrete",
        recipe,
        dataset_summaries,
        run_slug=ctx.run_slug,
        dataset_order=[spec.name for spec in specs],
    )
    print(f"Recorded finish: {finish_path} ({recipe})")
    return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m vision_encoder_eval.mllm.evaluation.run_eval <config_path>")
        sys.exit(1)
    sys.exit(run_evaluation(sys.argv[1]))
