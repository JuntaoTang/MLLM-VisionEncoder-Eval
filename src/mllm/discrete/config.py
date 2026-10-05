"""Config loading and RunContext for discrete VTB runs."""

from __future__ import annotations
from vision_encoder_eval.core.runtime import mllm_root, mllm_configs_root

from vision_encoder_eval.core.runtime import asset_path

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import yaml

from vision_encoder_eval.mllm.utils.config import (
    DISCRETE_CHECKPOINTS_ROOT,
    DEFAULT_DATA_CONFIG,
    SHARED_DATA_CONFIG,
    SHARED_RUNTIME_CONFIG,
    merge_recipe_batch,
    resolve_mllm_recipe_path,
    resolve_mode_runtime,
    _recipe_name_from_path,
)

VTB_ROOT = mllm_root()
PROJECT_ROOT = VTB_ROOT  # backward-compatible alias
CONFIGS_ROOT = mllm_configs_root()
DISCRETE_CONFIGS = os.path.join(CONFIGS_ROOT, "discrete")
DEFAULT_RUNTIME_CONFIG = SHARED_RUNTIME_CONFIG
DEFAULT_TRAIN_CONFIG = os.path.join(DISCRETE_CONFIGS, "train.yaml")
DEFAULT_DATA_CONFIG = SHARED_DATA_CONFIG

CKPT_ROOT = asset_path('download', '')
CACHE_ROOT_DEFAULT = asset_path('runtime', '')
DATA_ROOT = asset_path('datasets', '')
LOGS_ROOT = os.path.join(VTB_ROOT, "logs", "discrete")
RESULTS_ROOT = os.path.join(VTB_ROOT, "results", "discrete")
MODEL_ROOT = os.path.join(CKPT_ROOT, "tokenizer", "discrete")
LLM_ROOT = os.path.join(CKPT_ROOT, "llm")


def resolve_model_path(path: str) -> str:
    """Resolve checkpoint paths: absolute, under MODEL_ROOT, or under VTB_ROOT."""
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.abspath(path)
    vtb_candidate = os.path.abspath(os.path.join(VTB_ROOT, path))
    if os.path.exists(vtb_candidate):
        return vtb_candidate
    return os.path.abspath(os.path.join(MODEL_ROOT, path))


def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(path: str, root: str | None = None) -> str:
    if not path:
        return path
    if not os.path.isabs(path):
        base = root or PROJECT_ROOT
        path = os.path.join(base, path)
    return os.path.abspath(path)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_preset(value: Any, subdir: str) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        if subdir == "tokenizer":
            candidates = [
                os.path.join(DISCRETE_CONFIGS, "tokenizer", f"{value}.yaml"),
            ]
        else:
            candidates = [os.path.join(CONFIGS_ROOT, subdir, f"{value}.yaml")]
        path = next((p for p in candidates if os.path.isfile(p)), None)
        if not path:
            raise FileNotFoundError(f"Preset not found: {value!r} in {subdir}")
        cfg = _load_yaml(path)
        key = subdir.rstrip("s")
        entry = cfg.get(key, cfg)
        if isinstance(entry, dict) and "id" not in entry:
            entry = {**entry, "id": value}
        return entry
    raise ValueError(f"Invalid preset reference: {value!r}")


def experiment_slug(
    llm: dict,
    tokenizer: dict,
    projector: dict | None = None,
    finetune_tag: str | None = None,
) -> str:
    """Checkpoint/output slug: {llm}/{tokenizer}[/{finetune_tag}].

    Projector is part of the mllm recipe, not a separate checkpoint axis.
    Use runtime.experiment_slug to override (e.g. arch ablations).
    """
    del projector  # kept for call-site compatibility
    llm_id = llm.get("id")
    if not llm_id:
        raise ValueError("LLM preset must define `id`")
    tokenizer_id = tokenizer.get("id")
    if not tokenizer_id:
        raise ValueError("Tokenizer preset must define `id`")
    base = os.path.join(llm_id, tokenizer_id)
    if finetune_tag:
        base = os.path.join(base, finetune_tag)
    return base


def resolve_vlmeval_root() -> str:
    root = os.path.join(VTB_ROOT, "third_party", "VLMEvalKit")
    if os.path.isfile(os.path.join(root, "run.py")):
        return root
    raise FileNotFoundError(f"VLMEvalKit not found at {root}")


def get_default_lmudata_dir() -> str:
    from vision_encoder_eval.mllm.discrete.data_layout import ensure_lmudata_view

    return ensure_lmudata_view(DATA_ROOT)


def _join_data_root(root: str, rel: str) -> str:
    if os.path.isabs(rel):
        return rel
    return os.path.join(root, rel)


def load_data_config(
    data_config_path: str = DEFAULT_DATA_CONFIG,
    *,
    finetune_tag: str | None = None,
) -> dict:
    cfg = _load_yaml(_resolve_path(data_config_path))
    root = cfg.get("datasets_root", DATA_ROOT)
    finetune_presets = {}
    for tag, preset in (cfg.get("finetune_presets") or {}).items():
        finetune_presets[tag] = preset
    pretrain = cfg.get("pretrain", {})
    resolved = {
        "datasets_root": root,
        "finetune_presets": finetune_presets,
        "pretrain": pretrain,
        "finetune_tag": finetune_tag or "",
        "eval": {},
    }
    eval_cfg = cfg.get("eval", {})
    instructions_dir = _join_data_root(root, eval_cfg.get("instructions_dir", "instructions/test"))
    images_dir = _join_data_root(root, eval_cfg.get("images_dir", "images/test"))
    from vision_encoder_eval.mllm.discrete.data_layout import ensure_lmudata_view

    resolved["eval"] = {
        "instructions_dir": instructions_dir,
        "images_dir": images_dir,
        "lmudata_dir": ensure_lmudata_view(
            root,
            instructions_dir=instructions_dir,
            images_dir=images_dir,
        ),
    }
    return resolved


@dataclass
class ExperimentConfig:
    name: str = ""
    description: str = ""
    llm: dict = field(default_factory=dict)
    tokenizer: dict = field(default_factory=dict)
    projector: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    run_dir: str = ""
    checkpoint_dir: str = ""
    output_dir: str = ""
    results_dir: str = ""
    eval_data_dir: str = ""

    def resolve(self):
        for d in [self.run_dir, self.checkpoint_dir, self.output_dir, self.results_dir]:
            os.makedirs(d, exist_ok=True)
        return self


def load_config(run_config_path: str, eval_model: str | None = None) -> ExperimentConfig:
    from vision_encoder_eval.mllm.evaluation.checkpoint import pick_eval_checkpoint_dir

    ctx = load_run_context(run_config_path, eval_model=eval_model)
    resolved = pick_eval_checkpoint_dir(ctx, ctx.llm["model_name_or_path"])

    exp = ExperimentConfig()
    exp.name = ctx.registry_model_name or ctx.run_slug
    exp.description = ctx.experiment.get("description", "")
    exp.llm = ctx.llm
    exp.tokenizer = ctx.tokenizer
    exp.projector = ctx.projector
    exp.training = ctx.train.get("finetune", {})
    exp.run_dir = os.path.dirname(ctx.finetune_dir)
    exp.checkpoint_dir = resolved.path
    exp.output_dir = ctx.output_dir
    exp.results_dir = ctx.results_dir
    exp.eval_data_dir = ctx.data.get("eval", {}).get("lmudata_dir") or get_default_lmudata_dir()
    return exp.resolve()


@dataclass
class RunContext:
    experiment: dict = field(default_factory=dict)
    stages: list = field(default_factory=list)
    llm: dict = field(default_factory=dict)
    tokenizer: dict = field(default_factory=dict)
    projector: dict = field(default_factory=dict)
    checkpoints: dict = field(default_factory=dict)
    batch: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    runtime: dict = field(default_factory=dict)
    eval: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    arch: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)
    recipe_path: str = ""
    recipe_name: str = ""
    run_config_path: str = ""
    runtime_config_path: str = ""
    train_config_path: str = ""
    data_config_path: str = ""
    output_dir: str = ""
    results_dir: str = ""
    pretrain_dir: str = ""
    finetune_dir: str = ""
    checkpoint_base: str = ""
    finetune_tag: str = ""
    slug_override: str = ""
    registry_model_name: str = ""

    @property
    def run_slug(self) -> str:
        if self.slug_override:
            return self.slug_override
        return experiment_slug(self.llm, self.tokenizer, self.projector, self.finetune_tag)

    def stage_output_dir(self, stage: str) -> str:
        if stage == "pretrain":
            return self.pretrain_dir
        if stage == "finetune":
            return self.finetune_dir
        raise ValueError(f"Unknown stage: {stage}")

    def stage_batch(self, stage: str) -> dict:
        num_gpus = int(self.batch.get("num_gpus", 1))
        stage_cfg = self.batch.get(stage, {})
        if isinstance(stage_cfg, dict) and stage_cfg:
            micro = int(stage_cfg.get("per_device_train_batch_size", 8))
            accum = int(stage_cfg.get("gradient_accumulation_steps", 1))
        else:
            micro = int(self.batch.get("per_device_train_batch_size", 8))
            accum = int(self.batch.get("gradient_accumulation_steps", 2))
        return {
            "per_device_train_batch_size": micro,
            "gradient_accumulation_steps": accum,
            "num_gpus": num_gpus,
            "global_batch_size": micro * accum * num_gpus,
        }

    def resolve_paths(self):
        from vision_encoder_eval.mllm.discrete.checkpoint import (
            checkpoint_stage_has_content,
            checkpoint_stage_root,
            resolve_latest_stage_dir,
        )

        slug = self.run_slug
        base = self.output.get("base_dir") or DISCRETE_CHECKPOINTS_ROOT
        if not os.path.isabs(base):
            base = os.path.join(PROJECT_ROOT, base)
        self.checkpoint_base = base
        if checkpoint_stage_has_content(base, slug, "pretrain"):
            self.pretrain_dir = resolve_latest_stage_dir(base, slug, "pretrain")
        else:
            self.pretrain_dir = checkpoint_stage_root(base, slug, "pretrain")
        if checkpoint_stage_has_content(base, slug, "finetune"):
            self.finetune_dir = resolve_latest_stage_dir(base, slug, "finetune")
        else:
            self.finetune_dir = checkpoint_stage_root(base, slug, "finetune")
        self.output_dir = os.path.join(LOGS_ROOT, slug)
        self.results_dir = os.path.join(RESULTS_ROOT, slug)
        for d in [self.checkpoint_base, self.pretrain_dir, self.finetune_dir, self.output_dir, self.results_dir]:
            os.makedirs(d, exist_ok=True)
        return self

    def resolve_eval_stage_dir(self, stage: str) -> str:
        from vision_encoder_eval.mllm.discrete.checkpoint import resolve_latest_stage_dir
        return resolve_latest_stage_dir(self.checkpoint_base, self.run_slug, stage)


def load_run_context(
    config_path: str = DEFAULT_RUNTIME_CONFIG,
    train_config_path: str = DEFAULT_TRAIN_CONFIG,
    data_config_path: Optional[str] = None,
    *,
    recipe_override: str | None = None,
    finetune_tag_override: str | None = None,
    eval_model: str | None = None,
) -> RunContext:
    config_path = _resolve_path(config_path)
    runtime_cfg, _ = resolve_mode_runtime(config_path, "discrete")
    recipe_name = recipe_override
    if recipe_name is None:
        from vision_encoder_eval.mllm.utils.config import normalize_model_list

        names = normalize_model_list(runtime_cfg, "recipe", "recipes")
        if len(names) > 1:
            raise ValueError(
                "Multiple recipe entries require the pipeline multi-model loop "
                f"(got {names}). Pass recipe_override for a single recipe."
            )
        recipe_name = names[0] if names else None
    if not recipe_name:
        raise ValueError(f"Config {config_path} must set `recipe: <name>` under discrete section")
    recipe_path = resolve_mllm_recipe_path(DISCRETE_CONFIGS, recipe_name)
    recipe = _load_yaml(recipe_path)
    llm = _resolve_preset(recipe.get("llm"), "llm")
    tokenizer = _resolve_preset(recipe.get("tokenizer"), "tokenizer")
    projector = _resolve_preset(recipe.get("projector"), "projector")
    default_train = runtime_cfg.get("train_config") or DEFAULT_TRAIN_CONFIG
    train_path = train_config_path or default_train
    train_base = _load_yaml(_resolve_path(train_path))
    train_override = recipe.get("train", {})
    train = _deep_merge(train_base, train_override) if train_override else train_base
    data_cfg_path = data_config_path or runtime_cfg.get("data_config", DEFAULT_DATA_CONFIG)
    ctx = RunContext()
    ctx.run_config_path = config_path
    ctx.runtime_config_path = config_path
    ctx.train_config_path = _resolve_path(train_path)
    ctx.data_config_path = _resolve_path(data_cfg_path)
    ctx.recipe_path = recipe_path
    ctx.recipe_name = _recipe_name_from_path(recipe_path, os.path.join(DISCRETE_CONFIGS, "mllm"))
    ctx.experiment = recipe.get("experiment", {})
    ctx.stages = runtime_cfg.get("stages", [])
    ctx.llm = llm
    ctx.tokenizer = tokenizer
    ctx.projector = projector
    ctx.checkpoints = runtime_cfg.get("checkpoints", {})
    ctx.batch = merge_recipe_batch(runtime_cfg.get("batch", {}), recipe.get("batch"))
    ctx.output = runtime_cfg.get("output", {})
    ctx.runtime = runtime_cfg.get("runtime", {})
    runtime_arch = (runtime_cfg.get("runtime") or {}).get("arch", {})
    recipe_arch = recipe.get("arch", {})
    ctx.arch = _deep_merge(recipe_arch, runtime_arch)
    ctx.eval = _deep_merge(runtime_cfg.get("eval", {}), recipe.get("eval", {}))
    ctx.train = {
        "seed": train.get("seed", 42),
        "pretrain": train.get("pretrain", {}),
        "finetune": train.get("finetune", {}),
        "wandb": train.get("wandb", {}),
        "deepspeed": train.get("deepspeed", {}),
    }
    ctx.paths = runtime_cfg.get("paths", train.get("paths", {}))
    ctx.finetune_tag = str(finetune_tag_override or runtime_cfg.get("finetune_tag", ""))
    ctx.slug_override = runtime_cfg.get("experiment_slug")
    ctx.data = load_data_config(ctx.data_config_path, finetune_tag=ctx.finetune_tag)
    ctx = ctx.resolve_paths()
    if eval_model:
        ctx.eval["model"] = eval_model
    from vision_encoder_eval.mllm.discrete.registry import apply_default_eval_from_train_recipe, apply_eval_model_registry

    apply_default_eval_from_train_recipe(ctx)
    apply_eval_model_registry(ctx)
    if not ctx.checkpoints.get("pretrain_resume") and not ctx.checkpoints.get("finetune_resume"):
        ctx.pretrain_dir = ctx.resolve_eval_stage_dir("pretrain")
        ctx.finetune_dir = ctx.resolve_eval_stage_dir("finetune")
    return ctx
