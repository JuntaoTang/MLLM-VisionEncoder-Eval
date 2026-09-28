"""Build VLMEvalKit judge kwargs from runtime eval config (local or DashScope API)."""

from __future__ import annotations

import json
import os
from typing import Any

from src.utils.config import CONFIGS_ROOT

# Datasets where LLM judge extracts MCQ letter from verbose Qwen3 output.
LLM_JUDGE_DATASETS = frozenset(
    {
        "MMMU_TEST",
        "MMMU_DEV_VAL",
        "MMBench_TEST_EN_V11",
        "MMBench_DEV_EN_V11",
    }
)

DEFAULT_LOCAL_JUDGE = "Qwen3-32B"
DEFAULT_API_JUDGE = "qwen3.7-plus"
DEFAULT_LOCAL_JUDGE_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_LOCAL_JUDGE_MODEL_PATH = "/cache/ckpt/download/llm/Qwen3-32B"
DEFAULT_LOCAL_JUDGE_ARGS: dict[str, Any] = {
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 32,
    "enable_thinking": False,
    "chat_template_kwargs": {"enable_thinking": False},
}
DEFAULT_API_JUDGE_ARGS: dict[str, Any] = {
    "temperature": 0.01,
    "top_p": 0.001,
    "top_k": 1,
    "max_length": 128,
    "retry": 6,
    "timeout": 600,
    "verbose": False,
}
SECRETS_PATH = os.path.join(CONFIGS_ROOT, "secrets.yaml")


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _load_secrets() -> dict[str, Any]:
    if not os.path.isfile(SECRETS_PATH):
        return {}
    return _load_yaml(SECRETS_PATH)


def resolve_judge_api_key(eval_cfg: dict[str, Any] | None) -> str | None:
    cfg = eval_cfg or {}
    for candidate in (
        cfg.get("judge_api_key"),
        _load_secrets().get("judge_api_key"),
        _load_secrets().get("DASHSCOPE_API_KEY"),
        os.environ.get("DASHSCOPE_API_KEY"),
    ):
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text and text.lower() not in ("null", "none", ""):
            return text
    return None


def uses_api_judge(eval_cfg: dict[str, Any] | None) -> bool:
    return resolve_judge_api_key(eval_cfg) is not None


def resolve_judge_model_path(eval_cfg: dict[str, Any] | None) -> str:
    cfg = eval_cfg or {}
    return str(cfg.get("judge_model_path") or DEFAULT_LOCAL_JUDGE_MODEL_PATH)


def resolve_judge_profile(eval_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = eval_cfg or {}
    api_key = resolve_judge_api_key(cfg)
    if api_key:
        model = str(cfg.get("judge_api_model") or cfg.get("judge") or DEFAULT_API_JUDGE)
        extra = cfg.get("judge_args")
        if isinstance(extra, str) and extra.strip():
            extra = json.loads(extra)
        elif not isinstance(extra, dict):
            extra = None
        args = dict(DEFAULT_API_JUDGE_ARGS)
        if extra:
            args.update(extra)
        return {
            "mode": "api",
            "model": model,
            "api_key": api_key,
            "args": args,
        }

    extra = cfg.get("judge_args")
    if isinstance(extra, str) and extra.strip():
        extra = json.loads(extra)
    elif not isinstance(extra, dict):
        extra = None

    local_args = dict(DEFAULT_LOCAL_JUDGE_ARGS)
    if extra:
        local_args.update(extra)

    return {
        "mode": "local",
        "model": str(cfg.get("judge") or DEFAULT_LOCAL_JUDGE),
        "base_url": str(cfg.get("judge_base_url") or DEFAULT_LOCAL_JUDGE_BASE_URL),
        "model_path": resolve_judge_model_path(cfg),
        "args": local_args,
    }


def parse_judge_gpu_ids(
    eval_cfg: dict[str, Any] | None,
    *,
    all_eval_gpus: list[str] | None = None,
) -> list[str]:
    """Physical GPU ids for the local judge pool (default: first two eval GPUs)."""
    cfg = eval_cfg or {}
    for key in ("judge_cuda_devices", "judge_cuda_device", "judge_gpu"):
        raw = cfg.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        gpus = [part.strip() for part in text.split(",") if part.strip()]
        if gpus:
            return gpus

    if all_eval_gpus is None:
        all_eval_gpus = [
            part.strip()
            for part in str(cfg.get("cuda_visible_devices") or "0,1,2,3").split(",")
            if part.strip()
        ]
    count = max(1, int(cfg.get("judge_num_gpus", 1)))
    if not all_eval_gpus:
        return ["0"]
    # Prefer trailing GPUs so infer can take 0..N-1 (one task per card).
    placement = str(cfg.get("judge_gpu_placement") or "last").strip().lower()
    if placement in ("first", "head"):
        return all_eval_gpus[:count]
    return all_eval_gpus[-count:]


def uses_llm_judge(dataset_name: str) -> bool:
    """Only MMMU / MMBench need LLM answer extraction; VQA/caption never use judge."""
    return dataset_name in LLM_JUDGE_DATASETS


def apply_judge_env(eval_cfg: dict[str, Any] | None, env: dict[str, str] | None = None) -> None:
    """Set DASHSCOPE_API_KEY when using cloud judge."""
    profile = resolve_judge_profile(eval_cfg)
    if profile.get("mode") != "api":
        return
    key = str(profile.get("api_key") or "")
    if not key:
        return
    target = env if env is not None else os.environ
    target["DASHSCOPE_API_KEY"] = key


def build_judge_kwargs(
    eval_cfg: dict[str, Any] | None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Return kwargs for VLMEvalKit dataset.evaluate(..., **judge_kwargs)."""
    cfg = eval_cfg or {}
    nproc = int(cfg.get("judge_nproc", 32 if uses_api_judge(cfg) else 4))

    if dataset_name is not None and not uses_llm_judge(dataset_name):
        return {"model": "exact_matching", "nproc": nproc}

    profile = resolve_judge_profile(cfg)
    kwargs: dict[str, Any] = {
        "model": profile["model"],
        "nproc": nproc,
    }

    if profile.get("mode") == "api":
        kwargs["key"] = profile["api_key"]
    else:
        base_url = profile.get("base_url")
        if base_url:
            kwargs["api_base"] = f"{str(base_url).rstrip('/')}/chat/completions"

    args = profile.get("args")
    if isinstance(args, dict):
        kwargs.update(args)

    return kwargs


def format_judge_summary(eval_cfg: dict[str, Any] | None) -> str:
    profile = resolve_judge_profile(eval_cfg)
    if profile.get("mode") == "api":
        return f"api ({profile['model']} @ DashScope)"
    return f"local ({profile['model']} @ {profile['base_url']})"


def resolve_sampling_mode(eval_cfg: dict[str, Any] | None) -> str:
    """Return 'vlmevalkit' (random sample in VLMEvalKit) or 'tsv_head' (legacy)."""
    cfg = eval_cfg or {}
    mode = str(cfg.get("sampling") or "vlmevalkit").strip().lower()
    if mode not in ("vlmevalkit", "tsv_head"):
        raise ValueError(f"Unsupported eval.sampling: {mode!r} (use vlmevalkit or tsv_head)")
    return mode


def resolve_sample_num(eval_cfg: dict[str, Any] | None, per_dataset: int | None) -> int | None:
    """Effective sample count for one dataset job."""
    if per_dataset is not None:
        return per_dataset
    cfg = eval_cfg or {}
    raw = cfg.get("max_samples")
    if raw is None:
        return None
    value = int(raw)
    return value if value > 0 else None


def resolve_sample_seed(eval_cfg: dict[str, Any] | None) -> int:
    cfg = eval_cfg or {}
    return int(cfg.get("sample_seed", 42))
