"""Run multi-benchmark evaluation via bundled VLMEvalKit."""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile

from src.evaluation.checkpoint import pick_eval_checkpoint_dir
from src.evaluation.judge_config import (
    parse_judge_gpu_ids,
    resolve_sample_num,
    resolve_sample_seed,
    resolve_sampling_mode,
    uses_api_judge,
    uses_llm_judge,
)
from src.evaluation.results import (
    cleanup_temp_dir,
    dataset_dir,
    find_vlmeval_xlsx,
    make_vlmeval_work_dir,
    merge_prediction_xlsx,
    prepare_eval_lmudata,
    publish_predictions,
    score_predictions,
)
from src.utils.config import (
    VTB_ROOT,
    apply_cuda_stub_env,
    apply_offline_hf_env,
    install_offline_hf_env,
    load_config,
    load_run_context,
    resolve_llava_project,
)


def parse_all_eval_gpu_ids(ctx) -> list[str]:
    eval_cfg = ctx.eval or {}
    raw = eval_cfg.get("cuda_visible_devices")
    if raw is None:
        runtime = ctx.runtime or {}
        raw = runtime.get("cuda_visible_devices", "0")
    return [part.strip() for part in str(raw).split(",") if part.strip()] or ["0"]


def plan_infer_judge_gpus(
    ctx,
    datasets: list[str],
) -> tuple[list[str], list[str], str]:
    """Split GPUs into infer pool vs judge leftover.

    Returns (infer_gpus, judge_gpus, mode) where mode is:
      - ``overlap``: enough cards for concurrent infer + reserved judge
      - ``infer_then_judge``: use all cards for infer first, then free them for judge
    """
    eval_cfg = ctx.eval or {}
    all_gpus = parse_all_eval_gpu_ids(ctx)
    need_judge = (
        str(eval_cfg.get("judge") or "exact_matching") != "exact_matching"
        and any(uses_llm_judge(name) for name in datasets)
        and not uses_api_judge(eval_cfg)
    )
    if not need_judge:
        return all_gpus, [], "infer_only"

    judge_gpus = parse_judge_gpu_ids(eval_cfg, all_eval_gpus=all_gpus)
    max_workers = int(eval_cfg.get("max_infer_workers", len(all_gpus) or 1))
    # One dataset per GPU ideally; need room for leftover judge cards.
    if len(all_gpus) >= min(len(datasets), max_workers) + len(judge_gpus):
        infer = [g for g in all_gpus if g not in set(judge_gpus)] or all_gpus[:1]
        return infer, judge_gpus, "overlap"

    # Not enough free cards → run all infer in parallel, then load judge alone.
    return all_gpus, judge_gpus, "infer_then_judge"


def _vlmeval_root() -> str:
    return os.path.join(VTB_ROOT, "third_party", "VLMEvalKit")


def _ensure_vlmevalkit() -> None:
    root = _vlmeval_root()
    if not os.path.isfile(os.path.join(root, "run.py")):
        raise RuntimeError(
            f"VLMEvalKit not found at {root}. Run: bash scripts/setup_env.sh"
        )
    try:
        import vlmeval  # noqa: F401
    except ImportError as exc:
        hint = "VLMEvalKit is not installed. Run: bash scripts/setup_env.sh"
        msg = str(exc)
        if "libGL" in msg or "libgl" in msg.lower():
            hint = (
                "OpenCV failed to load libGL.so.1 (headless host). "
                "Install: pip install opencv-python-headless "
                "&& pip uninstall -y opencv-python"
            )
        raise RuntimeError(f"{hint} (import error: {exc})") from exc


def _model_key(run_slug: str) -> str:
    return run_slug.replace("/", "__")


def _build_vlmeval_config(
    ctx,
    checkpoint_dir: str,
    dataset_name: str,
    model_base: str | None = None,
    llm_path: str | None = None,
) -> tuple[dict, str]:
    eval_cfg = ctx.eval or {}
    model_key = _model_key(ctx.run_slug)
    conv_mode = eval_cfg.get("conv_mode")
    if not conv_mode:
        version = ctx.llm.get("model_version", "")
        if "qwen_3" in version or version == "qwen3":
            conv_mode = "qwen_3"
        elif "qwen" in version:
            conv_mode = "qwen_2_5"
        elif "smol" in version:
            conv_mode = "smollm2"
        else:
            conv_mode = "llava_llama_3"

    ve = ctx.vision_encoder or {}
    ve_type = ve.get("type", "open_clip_hub")
    vision_tower = ve.get("vision_tower")
    if not vision_tower and ve_type in ("dinov3", "raev2", "ijepa", "pe", "eupe", "pixio"):
        vision_tower = f"vtb_ssl:{ve_type}"
    elif not vision_tower and ve_type == "open_clip_hub" and ve.get("model_name"):
        vision_tower = f"open_clip_hub:{ve['model_name']}"
    elif not vision_tower and ve_type == "hf_clip":
        vision_tower = ve.get("model_name_or_path") or ve.get("weights_path")
    elif not vision_tower and ve_type in ("hf", "hf_vision"):
        path = ve.get("model_name_or_path") or ve.get("vision_tower") or ve.get("weights_path")
        vision_tower = f"hf:{path}" if path else None

    model_entry = {
        "class": "VTB_LLaVA",
        "model_path": checkpoint_dir,
        "model_base": model_base,
        "llm_path": llm_path or model_base,
        "conv_mode": conv_mode,
        "vision_weights": ve.get("weights_path"),
        "vision_tower": vision_tower,
        "processor_path": ve.get("processor_path"),
        "attn_implementation": eval_cfg.get("attn_implementation", "sdpa"),
        "max_new_tokens": int(eval_cfg.get("max_new_tokens", 2048)),
    }

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
    skip_scoring: bool = False,
) -> tuple[int, dict]:
    _ensure_vlmevalkit()
    install_offline_hf_env()

    ctx = load_run_context(config_path, eval_model=eval_model)
    exp = load_config(config_path, eval_model=eval_model)
    resolved = pick_eval_checkpoint_dir(ctx, ctx.llm["model_name_or_path"])
    model_base = ctx.llm["model_name_or_path"] if resolved.kind == "adapter" else None
    vlmeval_cfg, model_key = _build_vlmeval_config(
        ctx,
        resolved.path,
        dataset_name,
        model_base=model_base,
        llm_path=ctx.llm["model_name_or_path"],
    )

    results_root = exp.results_dir
    out_dir = result_dir or dataset_dir(results_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    eval_cfg = ctx.eval or {}
    sampling_mode = resolve_sampling_mode(eval_cfg)
    sample_num = resolve_sample_num(eval_cfg, max_samples)
    sample_seed = resolve_sample_seed(eval_cfg) if sampling_mode != "tsv_head" else None

    if sample_num is not None:
        lmudata_dir = prepare_eval_lmudata(
            exp.eval_data_dir, dataset_name, sample_num, sample_seed=sample_seed
        )
    else:
        lmudata_dir = exp.eval_data_dir
    work_dir = make_vlmeval_work_dir()

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(vlmeval_cfg, f, indent=2)
        cfg_path = f.name

    llava_root = resolve_llava_project(ctx.paths)
    env = apply_cuda_stub_env(os.environ.copy())
    apply_offline_hf_env(env)
    env["LMUData"] = lmudata_dir
    if sample_num is not None:
        env["VTB_ALLOW_SUBSAMPLED_TSV"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [VTB_ROOT, llava_root, _vlmeval_root(), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)

    # One physical GPU per worker (remapped to cuda:0 inside the process).
    if cuda_device is not None:
        cuda_devices = str(cuda_device)
    else:
        infer_gpus, _, _ = plan_infer_judge_gpus(ctx, [dataset_name])
        cuda_devices = infer_gpus[0]
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    env["VTB_EVAL_DEVICE_INDEX"] = "0"

    ve = ctx.vision_encoder or {}
    weights = ve.get("weights_path")
    ve_type = ve.get("type", "open_clip_hub")
    if weights and ve_type != "hf_clip":
        env["VTB_VISION_WEIGHTS"] = weights
    force_image_size = ve.get("force_image_size")
    if force_image_size is not None:
        env["VTB_FORCE_IMAGE_SIZE"] = str(int(force_image_size))
    force_qg = ve.get("force_quick_gelu")
    model_name = str(ve.get("model_name") or ve.get("vision_tower") or "")
    if force_qg is None and "siglip" in model_name.lower():
        force_qg = False
    if force_qg is True:
        env["VTB_FORCE_QUICK_GELU"] = "1"
    elif force_qg is False:
        env["VTB_FORCE_QUICK_GELU"] = "0"
    clip_processor = ve.get("processor_path")
    if clip_processor and os.path.isdir(str(clip_processor)):
        env["VTB_CLIP_IMAGE_PROCESSOR"] = str(clip_processor)
    if ve_type in ("dinov3", "raev2", "ijepa"):
        env["VTB_ROOT"] = VTB_ROOT
        image_size = ve.get("image_size") or force_image_size
        if image_size is not None:
            env["VTB_SSL_IMAGE_SIZE"] = str(int(image_size))
        layers = ve.get("layers")
        if layers is not None:
            if isinstance(layers, (list, tuple)):
                env["VTB_SSL_LAYERS"] = ".".join(str(int(x)) for x in layers)
            else:
                env["VTB_SSL_LAYERS"] = str(layers)
        dinov3_repo = ve.get("dinov3_repo") or os.path.join(VTB_ROOT, "third_party", "dinov3")
        if os.path.isdir(dinov3_repo):
            env["VTB_DINOV3_REPO_DIR"] = dinov3_repo
            env["DINOV3_REPO_DIR"] = dinov3_repo
    if ve_type in ("pe", "eupe", "pixio"):
        env["VTB_ROOT"] = VTB_ROOT
        image_size = ve.get("image_size") or force_image_size
        if image_size is not None:
            env["VTB_SSL_IMAGE_SIZE"] = str(int(image_size))
        pe_repo = ve.get("pe_repo") or os.path.join(VTB_ROOT, "third_party", "perception_models")
        if os.path.isdir(pe_repo):
            env["VTB_PE_REPO_DIR"] = pe_repo
        pe_config = ve.get("pe_config") or ve.get("model_name")
        if pe_config:
            env["VTB_PE_CONFIG"] = str(pe_config)
        eupe_repo = ve.get("eupe_repo") or os.path.join(VTB_ROOT, "third_party", "eupe")
        if os.path.isdir(eupe_repo):
            env["VTB_EUPE_REPO_DIR"] = eupe_repo
        if ve.get("eupe_hub"):
            env["VTB_EUPE_HUB"] = str(ve["eupe_hub"])
        pixio_repo = ve.get("pixio_repo") or os.path.join(VTB_ROOT, "third_party", "pixio")
        if os.path.isdir(pixio_repo):
            env["VTB_PIXIO_REPO_DIR"] = pixio_repo
        if ve.get("pixio_hub"):
            env["VTB_PIXIO_HUB"] = str(ve["pixio_hub"])

    cmd = [
        sys.executable,
        os.path.join(VTB_ROOT, "scripts", "run_vlmeval.py"),
        "--config",
        cfg_path,
        "--work-dir",
        work_dir,
        "--mode",
        str(ctx.eval.get("vlmeval_mode", "infer")),
    ]
    if ctx.eval.get("reuse", True):
        cmd.append("--reuse")
    # Infer-only: VLMEvalKit --judge is unused for our post scoring path.
    cmd.extend(["--judge", "exact_matching"])

    if sampling_mode != "tsv_head" and sample_num is not None and sample_seed is not None:
        sample_note = f"{sample_num}(seed={sample_seed},random)"
    else:
        sample_note = "all" if sample_num is None else f"{sample_num}(tsv_head)"
    print(f"  dataset={dataset_name} samples={sample_note} gpu={cuda_devices}")
    print(f"  output={out_dir}/predictions.xlsx")

    if log_path:
        with open(log_path, "a", encoding="utf-8") as log_f:
            log_f.write(f"=== vlmeval {dataset_name} gpu={cuda_devices} {cfg_path} ===\n")
            log_f.write(" ".join(cmd) + "\n")
            proc = subprocess.run(cmd, env=env, cwd=VTB_ROOT, stdout=log_f, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run(cmd, env=env, cwd=VTB_ROOT)

    summary: dict = {
        "dataset": dataset_name,
        "max_samples": sample_num,
        "sample_seed": sample_seed,
        "backend": "vlmevalkit",
        "checkpoint": exp.checkpoint_dir,
        "cuda_device": cuda_devices,
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
                published = publish_predictions(src_xlsx, results_root, dataset_name)
                summary["predictions_file"] = published
                # Always defer LLM-judge scoring so parallel infer is not blocked
                # waiting on the judge server (and can keep reserved GPUs free).
                defer = uses_llm_judge(dataset_name)
                if defer:
                    summary["metric"] = "deferred_llm_judge"
                    summary["accuracy"] = None
                    print(f"  {dataset_name}: infer done (scoring deferred for LLM judge)")
                else:
                    summary.update(score_predictions(published, dataset_name, eval_cfg=eval_cfg))
                    metric = summary.get("metric")
                    acc = summary.get("accuracy")
                    if metric == "mme_perception_reasoning":
                        acc_note = (
                            f"perception={summary.get('perception')} "
                            f"reasoning={summary.get('reasoning')}"
                        )
                    elif metric == "caption_cider" and isinstance(acc, (int, float)):
                        acc_note = f"CIDEr={acc * 100:.2f}"
                    elif isinstance(acc, (int, float)):
                        acc_note = f"{acc:.2%}"
                    else:
                        acc_note = "n/a"
                    print(
                        f"  {dataset_name}: {metric}={acc_note} ({summary.get('total')} samples)"
                    )
    finally:
        try:
            os.unlink(cfg_path)
        except OSError:
            pass
        cleanup_temp_dir(lmudata_dir)
        cleanup_temp_dir(work_dir)

    if proc.returncode != 0:
        print(f"  VLMEvalKit failed for {dataset_name} (exit {proc.returncode})")
        if log_path:
            print(f"  See log: {log_path}")
        return proc.returncode, summary

    return 0, summary


def run_vlmeval(config_path: str, log_path: str | None = None) -> int:
    """Backward-compatible wrapper: run all datasets from config."""
    from src.evaluation.run_eval import run_evaluation

    return run_evaluation(config_path, log_path=log_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.evaluation.vlmeval_runner <config_path>")
        sys.exit(1)
    sys.exit(run_vlmeval(sys.argv[1]))
