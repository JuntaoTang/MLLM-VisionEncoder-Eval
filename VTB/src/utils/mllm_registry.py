"""Registry of finished training runs under logs/mllm_list."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

import yaml

from src.utils.config import LOGS_ROOT, RESULTS_ROOT, VTB_ROOT

MLLM_LIST_YAML = os.path.join(LOGS_ROOT, "mllm_list.yaml")
MLLM_LIST_TXT = os.path.join(LOGS_ROOT, "mllm_list")


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
    os.makedirs(os.path.dirname(MLLM_LIST_TXT), exist_ok=True)
    with open(MLLM_LIST_TXT, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines))
            f.write("\n")


def load_registry() -> dict[str, Any]:
    return _load_yaml(MLLM_LIST_YAML)


def save_registry(registry: dict[str, Any]) -> None:
    registry.setdefault("models", {})
    _save_yaml(MLLM_LIST_YAML, registry)
    _sync_name_list(registry)


def list_registered_models() -> list[str]:
    return sorted((load_registry().get("models") or {}).keys())


def get_registry_entry(name: str) -> dict[str, Any]:
    registry = load_registry()
    models = registry.get("models") or {}
    if name not in models:
        available = ", ".join(sorted(models.keys())) or "(empty)"
        raise KeyError(
            f"Model {name!r} not found in {MLLM_LIST_TXT}. Available: {available}"
        )
    return models[name]


def canonical_model_name(llm: dict, vision_encoder: dict, projector: dict | None) -> str:
    llm_id = llm.get("id")
    vision_id = vision_encoder.get("id")
    projector_id = (projector or {}).get("id") or (projector or {}).get("type") or "mlp2x"
    if not llm_id or not vision_id:
        raise ValueError("LLM and vision presets must define `id` for registry naming")
    return f"{llm_id}__{vision_id}__{projector_id}"


def _run_suffix_from_log_dir(log_dir: str | None) -> str | None:
    if not log_dir:
        return None
    base = os.path.basename(os.path.normpath(log_dir))
    if base in ("latest",):
        return None
    return base


def _choose_registry_name(
    base_name: str,
    slug: str,
    mllm_recipe: str,
    log_dir: str | None,
    models: dict[str, Any],
    forced_name: str | None,
) -> tuple[str, bool]:
    """Return (registry_name, update_existing)."""
    if forced_name:
        if forced_name in models:
            entry = models[forced_name]
            same = entry.get("slug") == slug and entry.get("mllm_recipe") == mllm_recipe
            return forced_name, same
        return forced_name, False

    if base_name not in models:
        return base_name, False

    entry = models[base_name]
    if entry.get("slug") == slug and entry.get("mllm_recipe") == mllm_recipe:
        return base_name, True

    suffix = _run_suffix_from_log_dir(log_dir) or datetime.now().strftime("%m_%d_%H%M")
    candidate = f"{base_name}__{suffix}"
    if candidate not in models:
        return candidate, False

    idx = 1
    while f"{candidate}_{idx}" in models:
        idx += 1
    return f"{candidate}_{idx}", False


def register_trained_model(
    ctx,
    stage: str,
    *,
    log_dir: str | None = None,
    forced_name: str | None = None,
) -> str:
    """Append or update logs/mllm_list after a successful training stage."""
    registry = load_registry()
    models: dict[str, Any] = registry.setdefault("models", {})

    base_name = canonical_model_name(ctx.llm, ctx.vision_encoder, ctx.projector)
    forced = forced_name or ctx.output.get("register_name")
    if forced:
        name, update_existing = _choose_registry_name(
            base_name, ctx.run_slug, ctx.mllm_recipe_name, log_dir, models, forced
        )
    else:
        name, update_existing = _choose_registry_name(
            base_name, ctx.run_slug, ctx.mllm_recipe_name, log_dir, models, None
        )

    checkpoint_dir = ctx.stage_output_dir(stage)
    if update_existing and name in models:
        entry = dict(models[name])
    else:
        entry = {
            "mllm_recipe": ctx.mllm_recipe_name,
            "slug": ctx.run_slug,
            "description": ctx.experiment.get("description", ""),
            "checkpoints": {},
            "stages_completed": [],
            "use_checkpoint": stage,
            "registered_at": datetime.now().isoformat(timespec="seconds"),
        }

    checkpoints = dict(entry.get("checkpoints") or {})
    checkpoints[stage] = checkpoint_dir
    entry["checkpoints"] = checkpoints

    stages = list(entry.get("stages_completed") or [])
    if stage not in stages:
        stages.append(stage)
    entry["stages_completed"] = stages

    if "finetune" in stages:
        entry["use_checkpoint"] = "finetune"
    elif "pretrain" in stages:
        entry["use_checkpoint"] = "pretrain"

    entry["updated_at"] = datetime.now().isoformat(timespec="seconds")
    if log_dir:
        entry["last_log_dir"] = os.path.relpath(log_dir, VTB_ROOT) if log_dir.startswith(VTB_ROOT) else log_dir

    models[name] = entry
    save_registry(registry)
    return name


def apply_eval_model_registry(ctx) -> None:
    """Resolve eval.model from logs/mllm_list into checkpoint paths and defaults."""
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

    if not ctx.eval.get("mllm"):
        ctx.eval.setdefault("mllm", entry.get("mllm_recipe"))


def _eval_field_unset(value: Any) -> bool:
    return value is None or value == ""


def apply_default_eval_from_train_mllm(ctx) -> None:
    """When eval.model and eval.mllm are unset, use checkpoint from runtime `mllm`."""
    if not _eval_field_unset(ctx.eval.get("model")):
        return
    if not _eval_field_unset(ctx.eval.get("mllm")):
        return
    if not ctx.mllm_recipe_name:
        raise ValueError(
            "Both eval.model and eval.mllm are unset. Set top-level `mllm:` to the recipe "
            "whose trained checkpoint should be evaluated, or set eval.model from logs/mllm_list."
        )
    ctx.eval["checkpoint_source"] = "local_mllm"
    ctx.eval.setdefault("checkpoint_mllm", ctx.mllm_recipe_name)

