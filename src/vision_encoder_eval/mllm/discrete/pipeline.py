"""Pipeline orchestrator for discrete MLLM training."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

import yaml

from vision_encoder_eval.mllm.discrete.run_eval import run_evaluation
from vision_encoder_eval.mllm.discrete.checkpoint import finalize_stage_checkpoint, prepare_stage_output_dir
from vision_encoder_eval.mllm.discrete.config import PROJECT_ROOT, RunContext, load_run_context
from vision_encoder_eval.mllm.discrete.log_utils import redirect_output_to_file
from vision_encoder_eval.mllm.discrete.registry import register_trained_model
from vision_encoder_eval.mllm.discrete.wandb_config import (
    apply_wandb_env,
    format_wandb_summary,
    resolve_wandb_settings,
)


def allocate_stage_log_dir(output_root: str, slug: str, stage: str) -> str:
    stage_root = os.path.join(output_root, slug, stage)
    os.makedirs(stage_root, exist_ok=True)
    run_dir = os.path.join(stage_root, datetime.now().strftime("%m_%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _make_stage_log_dir(ctx: RunContext, stage: str) -> str:
    log_dir = ctx.runtime.get("log_dir", "logs/discrete")
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(PROJECT_ROOT, log_dir)
    return allocate_stage_log_dir(log_dir, ctx.run_slug, stage)


def _resolve_stage_data(ctx: RunContext, stage: str) -> dict:
    root = ctx.data.get("datasets_root", "")
    if stage == "pretrain":
        entries = normalize_stage_datasets(ctx.data.get("pretrain", {}), root, _join_data_root)
        return {"datasets": entries}
    if stage == "finetune":
        tag = ctx.finetune_tag or "mix"
        presets = ctx.data.get("finetune_presets", {})
        if tag not in presets:
            raise ValueError(
                f"Unknown finetune_tag {tag!r}. Available: {list(presets)}"
            )
        entries = normalize_stage_datasets(presets[tag], root, _join_data_root)
        return {"datasets": entries}
    return ctx.data.get(stage, {})


def _join_data_root(root: str, value: str) -> str:
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.join(root, value)


def _default_wandb_run_name(ctx: RunContext, stage: str) -> str:
    return f"{ctx.run_slug}/{stage}"


def _resolve_training_wandb(ctx: RunContext, stage: str, training: dict) -> dict:
    """Align report_to with available wandb credentials (VTB-style secrets.yaml)."""
    wandb_settings = resolve_wandb_settings(
        ctx.train.get(stage, {}),
        global_wandb=ctx.train.get("wandb"),
        default_run_name=_default_wandb_run_name(ctx, stage),
    )
    wants_wandb = str(training.get("report_to", "none")).strip().lower() == "wandb"
    if wants_wandb and not wandb_settings.get("enabled"):
        print(
            "Warning: report_to=wandb but no wandb api_key found. "
            "Set configs/secrets.yaml wandb.api_key, WANDB_API_KEY, or use ../VTB/configs/secrets.yaml. "
            "Falling back to report_to=none.",
            flush=True,
        )
        training["report_to"] = "none"
    elif wandb_settings.get("enabled"):
        print(f"W&B: {format_wandb_summary(wandb_settings)}", flush=True)
    return wandb_settings


from vision_encoder_eval.mllm.discrete.eval_utils import build_tokenizer_cfg
from vision_encoder_eval.mllm.utils.data_mix import normalize_stage_datasets, summarize_datasets
from vision_encoder_eval.mllm.discrete.model.vision_config import resolve_vis_mode
from vision_encoder_eval.mllm.discrete.train.pretrain_settings import resolve_pretrain_settings


def _build_phase_config(ctx: RunContext, stage: str, output_dir: str) -> dict:
    data_cfg = _resolve_stage_data(ctx, stage)
    datasets = data_cfg.get("datasets") or []
    arch_cfg = dict(ctx.arch or {})
    projector_cfg = dict(ctx.projector or {})
    tokenizer_cfg = build_tokenizer_cfg(ctx.tokenizer)
    arch_cfg.setdefault(
        "vis_mode",
        resolve_vis_mode({"arch": arch_cfg, "tokenizer": tokenizer_cfg, "projector": projector_cfg}),
    )
    root = ctx.data.get("datasets_root", "")
    train = ctx.train[stage]
    batch = ctx.stage_batch(stage)
    cfg = {
        "llm": {
            "model_name_or_path": ctx.llm["model_name_or_path"],
            "hidden_size": ctx.llm.get("hidden_size", 2048),
        },
        "tokenizer": tokenizer_cfg,
        "projector": projector_cfg,
        "arch": arch_cfg,
        "data": {
            "datasets": datasets,
            "max_length": train.get("model_max_length", 8192),
        },
        "training": {
            "batch_size": batch["per_device_train_batch_size"],
            "grad_accum": batch["gradient_accumulation_steps"],
            "learning_rate": train.get("learning_rate", 1e-4 if stage == "pretrain" else 1e-5),
            "mm_projector_lr": train.get("mm_projector_lr"),
            "num_epochs": train.get("num_train_epochs", 1),
            "warmup_ratio": train.get("warmup_ratio", 0.03),
            "weight_decay": train.get("weight_decay", 0.0),
            "max_grad_norm": train.get("max_grad_norm", 1.0),
            "lr_scheduler_type": train.get("lr_scheduler_type", "cosine"),
            "seed": ctx.train.get("seed", 42),
            "gradient_checkpointing": train.get("gradient_checkpointing", stage == "finetune"),
            "gradient_checkpointing_kwargs": train.get("gradient_checkpointing_kwargs"),
            "save_strategy": (
                "no" if train.get("save_strategy") is False else str(train.get("save_strategy", "no"))
            ),
            "save_total_limit": train.get("save_total_limit", 1),
            "logging_steps": train.get("logging_steps", 10),
            "report_to": train.get("report_to", "none"),
            "bf16": train.get("bf16", True),
            "tf32": train.get("tf32", False),
            "dataloader_drop_last": train.get("dataloader_drop_last", False),
            "num_workers": ctx.runtime.get("dataloader_num_workers", 8),
            "model_max_length": train.get("model_max_length", 8192),
            "max_steps": train.get("max_steps"),
            "save_steps": train.get("save_steps"),
        },
        "output_dir": output_dir,
    }
    if stage == "pretrain":
        pretrain = resolve_pretrain_settings({"arch": arch_cfg})
        cfg["training"].update(pretrain.training_overrides)
    return cfg


def _launch_training_subprocess(
    ctx: RunContext,
    stage: str,
    cfg_path: str,
    output_dir: str,
    *,
    wandb_settings: dict | None = None,
) -> int:
    """Launch training via torchrun when num_gpus > 1 (aligned with VTB)."""
    num_gpus = int(ctx.batch.get("num_gpus", 1))
    runtime = ctx.runtime or {}
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [PROJECT_ROOT, env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    env["CUDA_VISIBLE_DEVICES"] = str(runtime.get("cuda_visible_devices", "0"))
    if wandb_settings:
        apply_wandb_env(wandb_settings, env)

    if stage == "pretrain":
        module = "vision_encoder_eval.mllm.discrete.train.train"
        extra = [cfg_path, "--output-dir", output_dir]
        resume = ctx.checkpoints.get("pretrain_resume")
    elif stage == "finetune":
        module = "vision_encoder_eval.mllm.discrete.train.train_phase2"
        extra = [cfg_path, "--output-dir", output_dir, "--phase1-checkpoint", ctx.pretrain_dir]
        resume = ctx.checkpoints.get("finetune_resume")
    else:
        raise ValueError(f"Unknown training stage: {stage}")

    if resume:
        extra.extend(["--resume-from-checkpoint", resume])

    if num_gpus > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            f"--master_port={runtime.get('master_port', 29501)}",
            "-m",
            module,
            *extra,
        ]
        print(f"[{stage}] torchrun x{num_gpus}: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT)
        return proc.returncode

    if stage == "pretrain":
        from vision_encoder_eval.mllm.discrete.train.train import train as train_phase1
        train_phase1(cfg_path, output_dir, ctx.checkpoints.get("pretrain_resume"))
    else:
        from vision_encoder_eval.mllm.discrete.train.train_phase2 import train as train_phase2
        train_phase2(
            cfg_path,
            output_dir,
            ctx.pretrain_dir,
            ctx.checkpoints.get("finetune_resume"),
        )
    return 0


def run_training_stage(ctx: RunContext, stage: str, log_path: str | None = None) -> int:
    output_dir = prepare_stage_output_dir(ctx, stage)
    log_dir = os.path.dirname(log_path) if log_path else _make_stage_log_dir(ctx, stage)
    os.makedirs(log_dir, exist_ok=True)

    cfg = _build_phase_config(ctx, stage, output_dir)
    wandb_settings = _resolve_training_wandb(ctx, stage, cfg["training"])
    cfg_path = os.path.join(log_dir, f"{stage}_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    with redirect_output_to_file(log_path):
        print(f"=== {stage.upper()} {datetime.now().isoformat()} ===")
        print(f"Experiment: {ctx.run_slug}")
        print(f"Recipe: {ctx.recipe_name}")
        print(f"[{stage}] Log dir: {log_dir}")
        print(f"[{stage}] Output: {output_dir}")
        print(f"[{stage}] Data: {summarize_datasets(cfg['data']['datasets'])}")
        print(f"[{stage}] Config: {cfg_path}")

        if stage == "pretrain":
            resume = ctx.checkpoints.get("pretrain_resume")
            if resume:
                cfg["resume_from_checkpoint"] = resume
                print(f"[{stage}] Resume: {resume}", flush=True)
                with open(cfg_path, "w") as f:
                    yaml.dump(cfg, f, sort_keys=False)
            code = _launch_training_subprocess(
                ctx, stage, cfg_path, output_dir, wandb_settings=wandb_settings
            )
        elif stage == "finetune":
            cfg["phase1_checkpoint"] = ctx.pretrain_dir
            resume = ctx.checkpoints.get("finetune_resume")
            if resume:
                cfg["resume_from_checkpoint"] = resume
                print(f"[{stage}] Resume: {resume}", flush=True)
            with open(cfg_path, "w") as f:
                yaml.dump(cfg, f, sort_keys=False)
            code = _launch_training_subprocess(
                ctx, stage, cfg_path, output_dir, wandb_settings=wandb_settings
            )
        else:
            print(f"Unknown training stage: {stage}", file=sys.stderr)
            return 1
        if code != 0:
            print(f"{stage} failed with exit code {code}", file=sys.stderr)
            return code

        finalize_stage_checkpoint(output_dir)
        registered = register_trained_model(ctx, stage, log_dir=log_dir)
        print(f"Registered model: {registered}")
        print(f"=== {stage} completed {datetime.now().isoformat()} ===")
    return 0


def run_pipeline(
    config_path: str,
    train_config_path: str | None = None,
    data_config_path: str | None = None,
    stages_override: list[str] | None = None,
    recipe_override: str | None = None,
    finetune_tag_override: str | None = None,
    eval_model: str | None = None,
) -> int:
    from vision_encoder_eval.mllm.utils.config import materialize_single_model_config, normalize_model_list, resolve_mode_runtime

    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)

    runtime_cfg, _ = resolve_mode_runtime(config_path, "discrete")
    stages = stages_override or runtime_cfg.get("stages") or []
    if not stages:
        print("No stages configured. Set `stages` in configs/runtime.yaml (discrete).", file=sys.stderr)
        return 1

    if recipe_override:
        recipes = [recipe_override]
    else:
        recipes = normalize_model_list(runtime_cfg, "recipe", "recipes")
    if not recipes and not eval_model:
        print("No recipe configured. Set `recipe:` (string or list) in configs/runtime.yaml.", file=sys.stderr)
        return 1
    if not recipes:
        recipes = [None]

    print(f"[discrete] queue={len(recipes)} recipe(s)  stages={', '.join(stages)}")
    from vision_encoder_eval.mllm.utils.finish_tracker import is_finished

    for i, recipe in enumerate(recipes, 1):
        label = recipe or eval_model or "(eval)"
        if recipe and is_finished("discrete", recipe) and set(stages) >= {"pretrain", "finetune", "test"}:
            print(f"\n===== [{i}/{len(recipes)}] skip {label} (in results/finish.json) =====")
            continue
        print(f"\n===== [{i}/{len(recipes)}] {label} =====")
        try:
            from vision_encoder_eval.mllm.evaluation.judge_manager import release_eval_gpu_resources

            release_eval_gpu_resources(runtime_cfg.get("eval"))
        except Exception as exc:
            print(f"[warn] release_eval_gpu_resources failed: {exc}", file=sys.stderr)
        one_config = (
            materialize_single_model_config(config_path, "discrete", recipe, "recipe")
            if recipe
            else config_path
        )
        code = _run_one_discrete(
            config_path=one_config,
            train_config_path=train_config_path,
            data_config_path=data_config_path,
            stages=stages,
            recipe_override=recipe,
            finetune_tag_override=finetune_tag_override,
            eval_model=eval_model,
        )
        if code != 0:
            print(f"Failed on {label} (exit={code})", file=sys.stderr)
            return code

    print("\nAll discrete jobs done.")
    return 0


def _run_one_discrete(
    config_path: str,
    train_config_path: str | None,
    data_config_path: str | None,
    stages: list[str],
    recipe_override: str | None,
    finetune_tag_override: str | None,
    eval_model: str | None,
) -> int:
    kwargs = {"config_path": config_path}
    if train_config_path:
        kwargs["train_config_path"] = train_config_path
    if data_config_path:
        kwargs["data_config_path"] = data_config_path
    if recipe_override:
        kwargs["recipe_override"] = recipe_override
    if finetune_tag_override:
        kwargs["finetune_tag_override"] = finetune_tag_override

    ctx = load_run_context(**kwargs)

    print(f"[discrete] {ctx.run_slug}  stages={', '.join(stages)}")

    for stage in stages:
        if stage in ("pretrain", "finetune"):
            run_log_dir = _make_stage_log_dir(ctx, stage)
            log_path = os.path.join(run_log_dir, f"{stage}.log")
            code = run_training_stage(ctx, stage, log_path=log_path)
            if code != 0:
                print(f"{stage} failed. See {log_path}", file=sys.stderr)
                return code
        elif stage in ("test", "eval"):
            run_log_dir = _make_stage_log_dir(ctx, stage)
            eval_log = os.path.join(run_log_dir, "eval.log")
            with redirect_output_to_file(eval_log):
                code = run_evaluation(config_path, log_path=eval_log, eval_model=eval_model)
            if code != 0:
                print(f"Evaluation failed. See {eval_log}", file=sys.stderr)
                return code
        else:
            print(f"Unknown stage: {stage}", file=sys.stderr)
            return 1

    print(f"Done: {ctx.run_slug}")
    return 0
