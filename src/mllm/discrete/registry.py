"""Model registry for VTB-Discrete training runs."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import yaml

from vision_encoder_eval.mllm.discrete.config import LOGS_ROOT, RESULTS_ROOT

REGISTRY_YAML = os.path.join(LOGS_ROOT, "mllm_list.yaml")
REGISTRY_TXT = os.path.join(LOGS_ROOT, "mllm_list")


def _load_yaml(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)


def _sync_name_list(registry: dict[str, Any]) -> None:
    models = registry.get("models") or {}
    lines = sorted(models.keys())
    os.makedirs(os.path.dirname(REGISTRY_TXT), exist_ok=True)
    with open(REGISTRY_TXT, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines))
            f.write("\n")


def canonical_model_name(
    llm: dict,
    tokenizer: dict,
    projector: dict | None,
    finetune_tag: str | None = None,
) -> str:
    llm_id = llm.get("id")
    tokenizer_id = tokenizer.get("id")
    projector_id = (projector or {}).get("id") or "mlp2x"
    if not llm_id or not tokenizer_id:
        raise ValueError("LLM and tokenizer presets must define `id`")
    name = f"{llm_id}__{tokenizer_id}__{projector_id}"
    if finetune_tag:
        name = f"{name}__{finetune_tag}"
    return name


def register_trained_model(ctx, stage: str, *, log_dir: str | None = None) -> str:
    registry = _load_yaml(REGISTRY_YAML)
    models: dict[str, Any] = registry.setdefault("models", {})
    base_name = canonical_model_name(ctx.llm, ctx.tokenizer, ctx.projector, ctx.finetune_tag)
    name = base_name
    if name in models:
        idx = 1
        while f"{name}_{idx}" in models:
            idx += 1
        name = f"{name}_{idx}"
    checkpoint_dir = ctx.stage_output_dir(stage)
    entry = {
        "recipe": ctx.recipe_name,
        "slug": ctx.run_slug,
        "finetune_tag": ctx.finetune_tag,
        "description": ctx.experiment.get("description", ""),
        "checkpoints": {stage: checkpoint_dir},
        "stages_completed": [stage],
        "use_checkpoint": stage,
        "registered_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if log_dir:
        entry["last_log_dir"] = os.path.relpath(log_dir, LOGS_ROOT) if log_dir.startswith(LOGS_ROOT) else log_dir
    models[name] = entry
    _save_yaml(REGISTRY_YAML, registry)
    _sync_name_list(registry)
    return name


def load_registry() -> dict[str, Any]:
    return _load_yaml(REGISTRY_YAML)


def get_registry_entry(name: str) -> dict[str, Any]:
    registry = _load_yaml(REGISTRY_YAML)
    models = registry.get("models") or {}
    if name not in models:
        available = ", ".join(sorted(models.keys())) or "(empty)"
        raise KeyError(
            f"Model {name!r} not found in {REGISTRY_TXT}. Available: {available}"
        )
    return models[name]


def apply_eval_model_registry(ctx) -> None:
    model_name = ctx.eval.get("model")
    if not model_name:
        return

    entry = get_registry_entry(model_name)
    checkpoints = entry.get("checkpoints") or {}
    if checkpoints.get("pretrain"):
        ctx.pretrain_dir = checkpoints["pretrain"]
    if checkpoints.get("finetune"):
        ctx.finetune_dir = checkpoints["finetune"]
    if entry.get("use_checkpoint"):
        ctx.eval["use_checkpoint"] = entry["use_checkpoint"]
    ctx.registry_model_name = model_name
    ctx.eval["checkpoint_source"] = "registry"
    ctx.results_dir = os.path.join(RESULTS_ROOT, model_name)
    os.makedirs(ctx.results_dir, exist_ok=True)
    if not ctx.eval.get("recipe"):
        ctx.eval.setdefault("recipe", entry.get("recipe"))


def apply_default_eval_from_train_recipe(ctx) -> None:
    if ctx.eval.get("model") not in (None, ""):
        return
    if ctx.eval.get("recipe") not in (None, ""):
        return
    ctx.eval["checkpoint_source"] = "local_recipe"
    ctx.eval.setdefault("checkpoint_recipe", ctx.recipe_name)
