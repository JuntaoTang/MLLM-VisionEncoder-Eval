"""Resolve Weights & Biases settings from configs/secrets.yaml and configs/train.yaml."""

from __future__ import annotations

import os
from typing import Any

from vision_encoder_eval.mllm.discrete.config import CONFIGS_ROOT

SECRETS_PATH = os.path.join(CONFIGS_ROOT, "secrets.yaml")


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _secrets_path() -> str | None:
    if os.path.isfile(SECRETS_PATH):
        return SECRETS_PATH
    return None


def _load_secrets() -> dict[str, Any]:
    path = _secrets_path()
    if path is None:
        return {}
    return _load_yaml(path)


def _non_empty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("null", "none", ""):
        return None
    return text


def resolve_wandb_api_key() -> str | None:
    secrets = _load_secrets()
    wandb_secrets = secrets.get("wandb") if isinstance(secrets.get("wandb"), dict) else {}
    for candidate in (
        wandb_secrets.get("api_key"),
        secrets.get("wandb_api_key"),
        secrets.get("WANDB_API_KEY"),
        os.environ.get("WANDB_API_KEY"),
    ):
        key = _non_empty(candidate)
        if key:
            return key
    return None


def resolve_wandb_settings(
    train_stage_cfg: dict[str, Any] | None,
    *,
    global_wandb: dict[str, Any] | None = None,
    default_run_name: str | None = None,
) -> dict[str, Any]:
    """Build wandb env/training kwargs for one stage."""
    stage_cfg = train_stage_cfg or {}
    global_cfg = global_wandb if isinstance(global_wandb, dict) else {}
    secrets = _load_secrets()
    secret_cfg = secrets.get("wandb") if isinstance(secrets.get("wandb"), dict) else {}

    report_to = str(stage_cfg.get("report_to", "none")).strip().lower()
    enabled = report_to == "wandb" or bool(global_cfg.get("enabled"))
    api_key = resolve_wandb_api_key()
    if enabled and not api_key:
        enabled = False

    def pick(*candidates: Any) -> str | None:
        for candidate in candidates:
            value = _non_empty(candidate)
            if value:
                return value
        return None

    run_name = pick(
        stage_cfg.get("wandb_run_name"),
        global_cfg.get("run_name"),
        secret_cfg.get("run_name"),
        default_run_name,
        os.environ.get("WANDB_NAME"),
        os.environ.get("WANDB_RUN_NAME"),
    )

    return {
        "enabled": enabled,
        "api_key": api_key,
        "entity": pick(global_cfg.get("entity"), secret_cfg.get("entity"), os.environ.get("WANDB_ENTITY")),
        "project": pick(global_cfg.get("project"), secret_cfg.get("project"), os.environ.get("WANDB_PROJECT")),
        "mode": pick(global_cfg.get("mode"), secret_cfg.get("mode"), os.environ.get("WANDB_MODE")) or "online",
        "run_name": run_name,
        "group": pick(global_cfg.get("run_group")),
        "job_type": pick(global_cfg.get("job_type"), stage_cfg.get("wandb_job_type")),
    }


def resolve_eval_wandb_settings(ctx) -> dict[str, Any]:
    """Wandb settings for benchmark eval logging."""
    global_cfg = ctx.train.get("wandb") if isinstance(ctx.train.get("wandb"), dict) else {}
    secrets = _load_secrets()
    secret_cfg = secrets.get("wandb") if isinstance(secrets.get("wandb"), dict) else {}
    if not global_cfg.get("log_eval", True):
        return {"enabled": False}

    def pick(*candidates: Any) -> str | None:
        for candidate in candidates:
            value = _non_empty(candidate)
            if value:
                return value
        return None

    project = pick(global_cfg.get("project"), secret_cfg.get("project"), os.environ.get("WANDB_PROJECT"))
    api_key = resolve_wandb_api_key()
    run_name = pick(global_cfg.get("eval_run_name"), global_cfg.get("run_name"), f"{ctx.run_slug}/eval")

    return {
        "enabled": bool(project and api_key),
        "api_key": api_key,
        "entity": pick(global_cfg.get("entity"), secret_cfg.get("entity"), os.environ.get("WANDB_ENTITY")),
        "project": project,
        "mode": pick(global_cfg.get("mode"), secret_cfg.get("mode"), os.environ.get("WANDB_MODE")) or "online",
        "run_name": run_name,
        "group": pick(global_cfg.get("run_group")),
        "job_type": pick(global_cfg.get("job_type")),
    }


def apply_wandb_env(
    settings: dict[str, Any],
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Inject WANDB_* variables into a subprocess env dict."""
    target = env if env is not None else os.environ
    if not settings.get("enabled"):
        return target

    api_key = settings.get("api_key")
    if api_key:
        target["WANDB_API_KEY"] = str(api_key)

    for key, setting_key in (
        ("WANDB_ENTITY", "entity"),
        ("WANDB_PROJECT", "project"),
        ("WANDB_MODE", "mode"),
        ("WANDB_NAME", "run_name"),
        ("WANDB_RUN_GROUP", "group"),
        ("WANDB_JOB_TYPE", "job_type"),
    ):
        value = _non_empty(settings.get(setting_key))
        if value:
            target[key] = value

    return target


def format_wandb_summary(settings: dict[str, Any]) -> str:
    if not settings.get("enabled"):
        return "disabled (report_to=none or missing api_key)"
    parts = ["enabled"]
    if settings.get("project"):
        parts.append(f"project={settings['project']}")
    if settings.get("entity"):
        parts.append(f"entity={settings['entity']}")
    if settings.get("group"):
        parts.append(f"group={settings['group']}")
    if settings.get("run_name"):
        parts.append(f"run={settings['run_name']}")
    if settings.get("job_type"):
        parts.append(f"job={settings['job_type']}")
    parts.append(f"mode={settings.get('mode', 'online')}")
    parts.append("api_key=set")
    return ", ".join(parts)
