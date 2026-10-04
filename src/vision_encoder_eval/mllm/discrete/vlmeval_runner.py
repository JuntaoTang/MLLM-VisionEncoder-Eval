"""Run multi-benchmark evaluation via bundled VLMEvalKit."""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile

from vision_encoder_eval.mllm.evaluation.checkpoint import pick_eval_checkpoint_dir
from vision_encoder_eval.mllm.evaluation.judge_config import (
    apply_judge_env,
    build_judge_kwargs,
    resolve_judge_profile,
    resolve_sample_num,
    resolve_sample_seed,
    resolve_sampling_mode,
)
from vision_encoder_eval.mllm.evaluation.results import (
    cleanup_temp_dir,
    dataset_dir,
    find_vlmeval_xlsx,
    merge_prediction_xlsx,
    prepare_eval_lmudata,
    publish_predictions,
    score_predictions,
)
from vision_encoder_eval.mllm.discrete.config import (
    PROJECT_ROOT,
    load_config,
    load_run_context,
    resolve_vlmeval_root,
)


def parse_all_eval_gpu_ids(ctx) -> list[str]:
    eval_cfg = ctx.eval or {}
    raw = eval_cfg.get("cuda_visible_devices")
    if raw is None:
        runtime = ctx.runtime or {}
        raw = runtime.get("cuda_visible_devices", "0")
    return [part.strip() for part in str(raw).split(",") if part.strip()] or ["0"]


def parse_eval_gpu_ids(ctx, *, reserve_judge_gpu: bool = True) -> list[str]:
    from vision_encoder_eval.mllm.evaluation.judge_config import parse_judge_gpu_ids, uses_api_judge

    eval_cfg = ctx.eval or {}
    gpus = parse_all_eval_gpu_ids(ctx)
    if reserve_judge_gpu and not uses_api_judge(eval_cfg):
        reserved = set(parse_judge_gpu_ids(eval_cfg, all_eval_gpus=gpus))
        gpus = [gpu for gpu in gpus if gpu not in reserved]
    return gpus or ["0"]


def _eval_cuda_devices(ctx, cuda_device: str | None = None) -> str:
    if cuda_device is not None:
        return str(cuda_device)
    gpus = parse_eval_gpu_ids(ctx)
    return gpus[0] if gpus else "0"


def _vlmeval_root() -> str:
    return resolve_vlmeval_root()


def _ensure_vlmevalkit() -> None:
    root = _vlmeval_root()
    if not os.path.isfile(os.path.join(root, "run.py")):
        raise RuntimeError(f"VLMEvalKit not found at {root}")
    try:
        import vlmeval  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("VLMEvalKit is not installed.") from exc


def _model_key(run_slug: str) -> str:
    return run_slug.replace("/", "__")


def _append_vlmeval_logs(work_dir: str, dataset_name: str) -> None:
    log_dir = os.path.join(work_dir, "logs")
    if not os.path.isdir(log_dir):
        return
    for lf in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
        print(f"--- VLMEvalKit log ({dataset_name}): {os.path.basename(lf)} ---", flush=True)
        with open(lf, encoding="utf-8", errors="replace") as f:
            print(f.read(), flush=True)


from vision_encoder_eval.mllm.discrete.eval_utils import build_vlmeval_model_entry


def _build_vlmeval_config(ctx, checkpoint_dir: str, dataset_name: str) -> tuple[dict, str]:
    model_key = _model_key(ctx.run_slug)
    model_entry = build_vlmeval_model_entry(ctx, checkpoint_dir)
    return (
        {
            "model": {model_key: model_entry},
            "data": {dataset_name: {}},
        },
        model_key,
    )


def run_vlmeval_dataset(
    config_path: str,
    dataset_name: str,
    max_samples: int | None = None,
    result_dir: str | None = None,
    log_path: str | None = None,
    eval_model: str | None = None,
    cuda_device: str | None = None,
    lmudata_dir_override: str | None = None,
    skip_vlmeval_sampling: bool = False,
    skip_scoring: bool = False,
) -> tuple[int, dict]:
    _ensure_vlmevalkit()
    ctx = load_run_context(config_path, eval_model=eval_model)
    exp = load_config(config_path, eval_model=eval_model)
    resolved = pick_eval_checkpoint_dir(ctx, ctx.llm["model_name_or_path"])
    vlmeval_cfg, model_key = _build_vlmeval_config(ctx, resolved.path, dataset_name)

    results_root = exp.results_dir
    out_dir = result_dir or dataset_dir(results_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    eval_cfg = ctx.eval or {}
    sampling_mode = resolve_sampling_mode(eval_cfg)
    sample_num = resolve_sample_num(eval_cfg, max_samples)
    sample_seed = resolve_sample_seed(eval_cfg) if sampling_mode != "tsv_head" else None

    if lmudata_dir_override:
        lmudata_dir = lmudata_dir_override
    elif sample_num is not None:
        # Seeded random TSV (vlmevalkit) or first-N head (tsv_head).
        lmudata_dir = prepare_eval_lmudata(
            exp.eval_data_dir, dataset_name, sample_num, sample_seed=sample_seed
        )
    else:
        lmudata_dir = exp.eval_data_dir
    work_dir = os.path.join(out_dir, "vlmeval_work")
    os.makedirs(work_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(vlmeval_cfg, f, indent=2)
        cfg_path = f.name

    env = os.environ.copy()
    env["LMUData"] = lmudata_dir
    env["PYTHONUNBUFFERED"] = "1"
    env["VTB_EVAL_LOG_EVERY"] = str(eval_cfg.get("log_every", 50))
    env["VTB_ALLOW_SUBSAMPLED_TSV"] = "1"
    if log_path or not sys.stdout.isatty():
        env["TQDM_DISABLE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [PROJECT_ROOT, _vlmeval_root(), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)

    cuda_devices = _eval_cuda_devices(ctx, cuda_device=cuda_device)
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    env["VTB_EVAL_DEVICE_INDEX"] = "0"
    env["SPLIT_THINK"] = "1"
    env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    vlmeval_mode = str(ctx.eval.get("vlmeval_mode", "infer"))
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "vision_encoder_eval.mllm.evaluation.vlmeval_launcher",
        "--config",
        cfg_path,
        "--work-dir",
        work_dir,
        "--mode",
        vlmeval_mode,
    ]
    if ctx.eval.get("reuse", True):
        cmd.append("--reuse")
    # Response preview is printed in discrete_vlm when verbose_responses is set;
    # do not also pass VLMEvalKit --verbose (would duplicate every answer in eval.log).

    # Judge is only needed during local MCQ scoring (parent process), not infer.
    if vlmeval_mode != "infer":
        apply_judge_env(eval_cfg, env)
        judge_profile = resolve_judge_profile(eval_cfg)
        judge_kwargs = build_judge_kwargs(eval_cfg)
        judge = judge_kwargs.get("model")
        if judge:
            cmd.extend(["--judge", str(judge)])
        if judge_profile.get("base_url"):
            cmd.extend(["--judge-base-url", str(judge_profile["base_url"])])
        judge_args = judge_profile.get("args")
        if isinstance(judge_args, dict) and judge_args:
            cmd.extend(["--judge-args", json.dumps(judge_args)])
    else:
        judge = None

    if sampling_mode != "tsv_head" and sample_num is not None and sample_seed is not None:
        sample_note = f"{sample_num}(seed={sample_seed},random)"
    else:
        sample_note = (
            "all" if sample_num is None else f"{sample_num}(tsv_head)"
        )
    gpu_note = f" gpu={cuda_devices}" if cuda_device is not None else ""
    print(f"  dataset={dataset_name} samples={sample_note}{gpu_note}")
    print(f"  output={out_dir}/predictions.xlsx")

    print(f"=== vlmeval {dataset_name} gpu={cuda_devices} ===")
    print(" ".join(cmd))

    # Inherit stdout/stderr when pipeline redirected output to eval.log;
    # otherwise append subprocess output directly to log_path.
    if log_path and sys.stdout.isatty():
        with open(log_path, "a", encoding="utf-8") as log_f:
            proc = subprocess.run(
                cmd, env=env, cwd=PROJECT_ROOT,
                stdout=log_f, stderr=subprocess.STDOUT,
            )
    else:
        proc = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT)

    _append_vlmeval_logs(work_dir, dataset_name)

    summary: dict = {
        "dataset": dataset_name,
        "max_samples": sample_num,
        "sample_seed": sample_seed if sampling_mode == "vlmevalkit" else None,
        "sampling": sampling_mode,
        "judge": judge,
        "judge_mode": "local",
        "backend": "vlmevalkit",
        "checkpoint": exp.checkpoint_dir,
        "status": "failed" if proc.returncode != 0 else "ok",
    }

    try:
        if proc.returncode == 0:
            shard_patterns = [
                os.path.join(work_dir, model_key, "*", f"*_{dataset_name}*.xlsx"),
                os.path.join(work_dir, model_key, "*", f"{model_key}_{dataset_name}*.xlsx"),
            ]
            shards: list[str] = []
            for pattern in shard_patterns:
                shards.extend(sorted(set(glob.glob(pattern))))
            shards = sorted(set(shards))

            src_xlsx = shards[-1] if len(shards) == 1 else None
            if len(shards) > 1:
                merged_tmp = os.path.join(work_dir, f"{model_key}_{dataset_name}_merged.xlsx")
                merge_prediction_xlsx(shards, merged_tmp)
                src_xlsx = merged_tmp
            if src_xlsx is None:
                src_xlsx = find_vlmeval_xlsx(work_dir, model_key, dataset_name)
            if not src_xlsx or not os.path.isfile(src_xlsx):
                summary["error"] = "prediction xlsx not found after inference"
                proc = subprocess.CompletedProcess(cmd, 1)
            else:
                publish_root = results_root
                published = publish_predictions(src_xlsx, publish_root, dataset_name)
                summary["predictions_file"] = published
                summary["total"] = sample_num
                if skip_scoring:
                    summary["status"] = "ok"
                else:
                    # Defer LLM-judge MCQ scoring so parallel MMMU/MMBench do not
                    # race-start two Qwen3-32B loads on the reserved judge GPU.
                    from vision_encoder_eval.mllm.evaluation.judge_config import uses_llm_judge

                    if uses_llm_judge(dataset_name):
                        summary["metric"] = "deferred_llm_judge"
                        summary["accuracy"] = None
                        summary["status"] = "ok"
                        print(
                            f"  {dataset_name}: infer done "
                            f"(scoring deferred for LLM judge)"
                        )
                    else:
                        summary.update(
                            score_predictions(
                                published,
                                dataset_name,
                                eval_cfg=eval_cfg,
                            )
                        )
                        metric = summary.get("metric")
                        acc = summary.get("accuracy")
                        primary = summary.get("primary_score")
                        primary_metric = summary.get("primary_metric")
                        if metric == "mme_perception_reasoning":
                            acc_note = (
                                f"perception={summary.get('perception')} "
                                f"reasoning={summary.get('reasoning')}"
                            )
                        elif metric == "caption_bleu0" or primary_metric == "bleu0":
                            bleu0 = summary.get("bleu0")
                            acc_note = (
                                f"Bleu0={bleu0:.2f}"
                                if isinstance(bleu0, (int, float))
                                else "n/a"
                            )
                        elif primary_metric == "accuracy" and isinstance(
                            primary, (int, float)
                        ):
                            acc_note = f"{primary:.2%}"
                        elif isinstance(acc, (int, float)):
                            acc_note = f"{acc:.2%}"
                        else:
                            acc_note = "n/a"
                        print(
                            f"  {dataset_name}: {metric}={acc_note} "
                            f"({summary.get('total')} samples)"
                        )
    finally:
        try:
            os.unlink(cfg_path)
        except OSError:
            pass
        if sampling_mode == "tsv_head" and sample_num is not None:
            cleanup_temp_dir(lmudata_dir)

    if proc.returncode != 0:
        print(f"  VLMEvalKit failed for {dataset_name} (exit {proc.returncode})")
        if log_path:
            print(f"  See log: {log_path}")
        return proc.returncode, summary

    return 0, summary
