from vision_encoder_eval.core.runtime import mllm_root, mllm_configs_root

from vision_encoder_eval.core.runtime import asset_path
import glob
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import yaml

from vision_encoder_eval.mllm.evaluation.checkpoint import pick_eval_checkpoint_dir

VTB_ROOT = mllm_root()
CONFIGS_ROOT = mllm_configs_root()
CONTINUOUS_CONFIGS = os.path.join(CONFIGS_ROOT, "continuous")
DISCRETE_CONFIGS = os.path.join(CONFIGS_ROOT, "discrete")
SHARED_RUNTIME_CONFIG = os.path.join(CONFIGS_ROOT, "runtime.yaml")
SHARED_DATA_CONFIG = os.path.join(CONFIGS_ROOT, "data.yaml")
DEFAULT_LLAVA_PROJECT = os.path.join(VTB_ROOT, "third_party", "LLaVA-NeXT")
DEFAULT_RUNTIME_CONFIG = SHARED_RUNTIME_CONFIG
DEFAULT_TRAIN_CONFIG = os.path.join(CONTINUOUS_CONFIGS, "train.yaml")
DEFAULT_DATA_CONFIG = SHARED_DATA_CONFIG
VTB_CACHE_ROOT = asset_path('runtime', '')
TRAINED_CKPT_ROOT = asset_path('trained', '')
CUDA_STUB_HOME = os.path.join(VTB_ROOT, "scripts", "cuda_stub")
if not os.path.isfile(os.path.join(CUDA_STUB_HOME, "bin", "nvcc")):
    CUDA_STUB_HOME = asset_path('package', 'resources/cuda_stub')
DATA_ROOT = os.path.join(VTB_CACHE_ROOT, "data")
CHECKPOINTS_ROOT = os.path.join(TRAINED_CKPT_ROOT, "continuous")
DISCRETE_CHECKPOINTS_ROOT = os.path.join(TRAINED_CKPT_ROOT, "discrete")
PRETRAINED_ROOT = os.path.join(DATA_ROOT, "models")
DATASETS_ROOT = asset_path('datasets', '')
HF_HOME_DEFAULT = os.path.join(DATA_ROOT, "cache", "huggingface")
MANIFESTS_ROOT = os.path.join(DATA_ROOT, "manifests")
LOGS_ROOT = os.path.join(VTB_ROOT, "logs", "continuous")
RESULTS_ROOT = os.path.join(VTB_ROOT, "results", "continuous")

_RUNTIME_KEYS = (
    "stages",
    "batch",
    "runtime",
    "eval",
    "checkpoints",
    "output",
    "data_config",
)


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_vtb_path(path: str) -> str:
    if not path:
        return path
    if not os.path.isabs(path):
        path = os.path.join(VTB_ROOT, path)
    return os.path.abspath(path)


def resolve_llava_project(paths: dict | None = None) -> str:
    """Return the bundled LLaVA-NeXT path inside VTB."""
    llava = (paths or {}).get("llava_project") or DEFAULT_LLAVA_PROJECT
    return _resolve_vtb_path(llava)


_OFFLINE_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "HF_ENDPOINT",
)


def apply_cuda_stub_env(base: dict | None = None) -> dict:
    """Point CUDA_HOME at the nvcc stub (machines without CUDA toolkit / nvcc)."""
    env = dict(os.environ if base is None else base)
    env["CUDA_HOME"] = CUDA_STUB_HOME
    env["DS_SKIP_CUDA_CHECK"] = "1"
    return env


def install_cuda_stub_env() -> None:
    os.environ["CUDA_HOME"] = CUDA_STUB_HOME
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"


def apply_offline_hf_env(base: dict | None = None) -> dict:
    """Build a subprocess env dict with offline HuggingFace settings and no shell proxy."""
    env = dict(os.environ if base is None else base)
    for key in _OFFLINE_ENV_KEYS:
        env.pop(key, None)
    env.update(
        {
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def install_offline_hf_env() -> None:
    """Apply offline HuggingFace settings to the current Python process."""
    install_cuda_stub_env()
    for key in _OFFLINE_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(
        {
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_mode_runtime(config_path: str, mode: str) -> tuple[dict, str]:
    """Load runtime yaml; if it has ``continuous``/``discrete`` sections, merge shared + mode block."""
    path = _resolve_vtb_path(config_path)
    cfg = _load_yaml(path)
    if mode in cfg and isinstance(cfg.get(mode), dict):
        shared = {k: v for k, v in cfg.items() if k not in ("continuous", "discrete")}
        merged = _deep_merge(shared, cfg[mode])
        return merged, path
    return cfg, path


def normalize_model_list(runtime_cfg: dict, *keys: str) -> list[str]:
    """Accept a scalar or list under the first present key (e.g. mllm / mllms)."""
    val = None
    found = None
    for key in keys:
        if key in runtime_cfg and runtime_cfg[key] is not None:
            val = runtime_cfg[key]
            found = key
            break
    if val is None:
        return []
    if isinstance(val, str):
        name = val.strip()
        return [name] if name else []
    if isinstance(val, (list, tuple)):
        out = [str(x).strip() for x in val if str(x).strip()]
        if not out:
            raise ValueError(f"`{found}` list is empty")
        return out
    raise ValueError(f"`{found}` must be a string or list of strings, got {type(val)!r}")


def materialize_single_model_config(
    config_path: str, mode: str, model_name: str, key: str
) -> str:
    """Write a temp runtime yaml with a single mllm/recipe so loaders stay single-model."""
    import tempfile
    from pathlib import Path

    path = _resolve_vtb_path(config_path)
    cfg = _load_yaml(path)
    if mode in cfg and isinstance(cfg.get(mode), dict):
        block = dict(cfg[mode])
        block[key] = model_name
        block.pop(f"{key}s", None)
        cfg = {**cfg, mode: block}
    else:
        cfg = dict(cfg)
        cfg[key] = model_name
        cfg.pop(f"{key}s", None)

    slug = model_name.replace("/", "__")
    out = Path(tempfile.mkdtemp(prefix="vtb_runtime_")) / f"{mode}_{slug}.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(out)


def merge_recipe_batch(runtime_batch: dict | None, recipe_batch: dict | None) -> dict:
    """Merge batch settings: recipe fills stages that runtime did not set.

    Normal training leaves stage micro/accum unset in runtime so the mllm recipe
    wins. Smoke/lifecycle runtimes set explicit per-stage micros and must not be
    overwritten by the probed optimal batch (which can OOM on full-model save).
    """
    merged = dict(runtime_batch or {})
    if not recipe_batch:
        return merged
    for stage in ("pretrain", "finetune"):
        if isinstance(merged.get(stage), dict) and merged[stage]:
            continue
        stage_batch = recipe_batch.get(stage)
        if isinstance(stage_batch, dict) and stage_batch:
            merged[stage] = dict(stage_batch)
    return merged


def divisors(n: int) -> list[int]:
    out = []
    for m in range(1, n + 1):
        if n % m == 0:
            out.append(m)
    return sorted(out, reverse=True)


PRETRAIN_EFFECTIVE_BATCH = 128
FINETUNE_EFFECTIVE_BATCH = 32


def batch_from_micro(micro: int, *, stage: str) -> dict:
    target = PRETRAIN_EFFECTIVE_BATCH if stage == "pretrain" else FINETUNE_EFFECTIVE_BATCH
    if target % micro != 0:
        raise ValueError(f"effective batch {target} not divisible by micro={micro}")
    return {
        "per_device_train_batch_size": micro,
        "gradient_accumulation_steps": target // micro,
    }


def _join_data_root(root: str, value: str) -> str:
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.join(root, value)


def _normalize_stage_datasets(stage_cfg: dict) -> list[dict]:
    if "datasets" in stage_cfg:
        entries = stage_cfg["datasets"]
        if not entries:
            raise ValueError("datasets must not be empty")
        return list(entries)
    if "data_path" in stage_cfg:
        return [{
            "data_path": stage_cfg["data_path"],
            "image_folder": stage_cfg.get("image_folder", ""),
            "sampling_strategy": stage_cfg.get("sampling_strategy", "all"),
        }]
    raise ValueError("Stage config must define `datasets` or `data_path`")


def _resolve_dataset_entry(root: str, entry: dict) -> dict:
    data_path = entry.get("data_path")
    if not data_path:
        raise ValueError("Each dataset entry must define `data_path`")
    image_folder = entry.get("image_folder", "")
    if not image_folder:
        raise ValueError(f"Dataset {data_path!r} must define `image_folder`")
    return {
        "data_path": _join_data_root(root, data_path),
        "image_folder": _join_data_root(root, image_folder),
        "sampling_strategy": str(entry.get("sampling_strategy", "all")),
    }


def _stage_needs_manifest(entries: list[dict]) -> bool:
    if len(entries) > 1:
        return True
    return len(entries) == 1 and entries[0]["sampling_strategy"] != "all"


def _dataset_mix_fingerprint(stage: str, entries: list[dict]) -> str:
    payload = {
        "stage": stage,
        "datasets": [
            {
                "data_path": entry["data_path"],
                "image_folder": entry["image_folder"],
                "sampling_strategy": entry["sampling_strategy"],
            }
            for entry in entries
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _manifest_path(stage: str, fingerprint: str) -> str:
    return os.path.join(MANIFESTS_ROOT, stage, f"{fingerprint[:16]}.yaml")


def _manifest_registry_path() -> str:
    return os.path.join(MANIFESTS_ROOT, "registry.yaml")


def _load_manifest_registry() -> dict:
    path = _manifest_registry_path()
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _save_manifest_registry(registry: dict) -> None:
    os.makedirs(MANIFESTS_ROOT, exist_ok=True)
    with open(_manifest_registry_path(), "w") as f:
        yaml.dump(registry, f, sort_keys=False, allow_unicode=True)


def _read_manifest_fingerprint(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    vtb = data.get("vtb") or {}
    return vtb.get("fingerprint")


def _write_llava_manifest(stage: str, entries: list[dict], fingerprint: str) -> str:
    path = _manifest_path(stage, fingerprint)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    manifest = {
        "vtb": {
            "fingerprint": fingerprint,
            "stage": stage,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sources": [
                {
                    "data_path": entry["data_path"],
                    "image_folder": entry["image_folder"],
                    "sampling_strategy": entry["sampling_strategy"],
                }
                for entry in entries
            ],
        },
        "datasets": [
            {
                "json_path": entry["data_path"],
                "image_folder": entry["image_folder"],
                "sampling_strategy": entry["sampling_strategy"],
            }
            for entry in entries
        ],
    }
    with open(path, "w") as f:
        yaml.dump(manifest, f, sort_keys=False, allow_unicode=True)

    registry = _load_manifest_registry()
    stage_registry = registry.setdefault(stage, {})
    stage_registry[fingerprint] = {
        "path": path,
        "created_at": manifest["vtb"]["created_at"],
        "sources": manifest["vtb"]["sources"],
    }
    _save_manifest_registry(registry)
    return path


def _get_or_create_llava_manifest(stage: str, entries: list[dict]) -> tuple[str, str, bool]:
    """Return manifest path, fingerprint, and whether an existing mix was reused."""
    fingerprint = _dataset_mix_fingerprint(stage, entries)
    path = _manifest_path(stage, fingerprint)
    if os.path.isfile(path) and _read_manifest_fingerprint(path) == fingerprint:
        return path, fingerprint, True
    path = _write_llava_manifest(stage, entries, fingerprint)
    return path, fingerprint, False


def _resolve_stage_data(stage: str, stage_cfg: dict, root: str) -> dict:
    entries = [_resolve_dataset_entry(root, entry) for entry in _normalize_stage_datasets(stage_cfg)]
    manifest_fingerprint = None
    manifest_reused = None
    if _stage_needs_manifest(entries):
        data_path, manifest_fingerprint, manifest_reused = _get_or_create_llava_manifest(stage, entries)
        image_folder = entries[0]["image_folder"]
    else:
        data_path = entries[0]["data_path"]
        image_folder = entries[0]["image_folder"]
    return {
        "data_path": data_path,
        "image_folder": image_folder,
        "datasets": entries,
        "manifest_fingerprint": manifest_fingerprint,
        "manifest_reused": manifest_reused,
    }


def format_stage_data_summary(stage_data: dict) -> str:
    datasets = stage_data.get("datasets") or []
    if len(datasets) <= 1 and not stage_data.get("manifest_fingerprint"):
        return stage_data.get("data_path", "")

    lines: list[str] = []
    fingerprint = stage_data.get("manifest_fingerprint")
    if fingerprint:
        status = "reused" if stage_data.get("manifest_reused") else "created"
        lines.append(f"manifest {status} (mix_id={fingerprint[:12]})")
    else:
        lines.append(f"mixed ({len(datasets)} datasets, random shuffle each epoch)")

    for entry in datasets:
        name = os.path.basename(entry["data_path"])
        strategy = entry.get("sampling_strategy", "all")
        lines.append(f"  - {name} [{strategy}]")

    data_path = stage_data.get("data_path", "")
    if data_path.endswith(".yaml"):
        lines.append(f"  -> {data_path}")
    return "\n".join(lines)


def load_data_config(data_config_path: str = DEFAULT_DATA_CONFIG) -> dict:
    cfg = _load_yaml(_resolve_vtb_path(data_config_path))
    root = cfg.get("datasets_root", DATASETS_ROOT)
    resolved = {"datasets_root": root, "eval": {}}

    for stage in ("pretrain", "finetune"):
        if stage not in cfg:
            continue
        resolved[stage] = _resolve_stage_data(stage, cfg[stage], root)

    eval_cfg = cfg.get("eval", {})
    instructions_dir = eval_cfg.get("instructions_dir")
    images_dir = eval_cfg.get("images_dir")
    if instructions_dir or images_dir:
        from vision_encoder_eval.mllm.discrete.data_layout import ensure_lmudata_view

        resolved["eval"] = {
            "instructions_dir": _join_data_root(root, instructions_dir or "instructions/test"),
            "images_dir": _join_data_root(root, images_dir or "images/test"),
            "lmudata_dir": ensure_lmudata_view(
                root,
                instructions_dir=_join_data_root(root, instructions_dir or "instructions/test"),
                images_dir=_join_data_root(root, images_dir or "images/test"),
            ),
        }
    else:
        resolved["eval"] = {
            "lmudata_dir": _join_data_root(root, eval_cfg.get("lmudata_dir", ".lmudata")),
        }
    return resolved


# Retired continuous recipe / vision-encoder aliases (old -> current).
_RECIPE_ALIASES = {
    "clip_vit_l14_hf_mlp2x": "clip_openai__l14_mlp2x",
    "vit_l14_openai_clip_mlp2x": "clip_openai__l14_mlp2x",
}
_VISION_ENCODER_ALIASES = {
    "clip_vit_l14_hf": "clip_openai__l14",
    "vit_l14_openai_clip": "clip_openai__l14",
}


def _normalize_recipe_name(name: str) -> str:
    """Map legacy names to llm/recipe and mlp2x layout."""
    if name.endswith("_mlp3x_gelu"):
        name = f"{name[:-11]}_mlp2x"
    for llm_id in ("qwen25", "qwen3", "smollm2"):
        prefix = f"{llm_id}_"
        if name.startswith(prefix):
            name = f"{llm_id}/{name[len(prefix):]}"
            break
    if "/" in name:
        llm_id, recipe = name.split("/", 1)
        recipe = _RECIPE_ALIASES.get(recipe, recipe)
        return f"{llm_id}/{recipe}"
    return _RECIPE_ALIASES.get(name, name)


def resolve_mllm_recipe_path(configs_root: str, name: str) -> str:
    """Resolve recipe yaml under configs/{mode}/mllm/{llm}/{recipe}.yaml."""
    name = _normalize_recipe_name(name)
    flat = os.path.join(configs_root, "mllm", f"{name}.yaml")
    if os.path.isfile(flat):
        return flat
    if "/" in name:
        llm_id, recipe = name.split("/", 1)
        nested = os.path.join(configs_root, "mllm", llm_id, f"{recipe}.yaml")
        if os.path.isfile(nested):
            return nested
    else:
        matches = glob.glob(os.path.join(configs_root, "mllm", "*", f"{name}.yaml"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            llm_dirs = sorted({os.path.basename(os.path.dirname(p)) for p in matches})
            raise ValueError(
                f"Ambiguous recipe {name!r}; found in {llm_dirs}. "
                f"Use llm/recipe format, e.g. {llm_dirs[0]}/{name}"
            )
    raise FileNotFoundError(f"MLLM recipe not found: {name!r} under {configs_root}/mllm")


def _vision_encoder_preset_paths(value: str) -> list[str]:
    base = os.path.join(CONTINUOUS_CONFIGS, "vision_encoder")
    paths = [os.path.join(base, f"{value}.yaml")]
    for sub in (
        "mc1",
        "mc2",
        "siglip2",
        "dinov3",
        "raev2",
        "ijepa",
        "pe",
        "dinov2",
        "dino",
        "webssl",
        "eupe",
        "pixio",
        "dinov3_hf",
    ):
        paths.append(os.path.join(base, sub, f"{value}.yaml"))
    paths.append(os.path.join(base, f"{value}.yaml"))
    paths.append(os.path.join(CONFIGS_ROOT, "vision_encoder", f"{value}.yaml"))
    return paths


def _resolve_preset(value: Any, subdir: str) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        if subdir == "vision_encoder":
            value = _VISION_ENCODER_ALIASES.get(value, value)
            candidates = _vision_encoder_preset_paths(value)
        else:
            candidates = [os.path.join(CONFIGS_ROOT, subdir, f"{value}.yaml")]
        path = next((p for p in candidates if os.path.isfile(p)), None)
        if not path:
            raise FileNotFoundError(f"Preset not found: {value!r} in {subdir}")
        preset = _load_yaml(path)
        key = subdir.rstrip("s") if subdir.endswith("s") else subdir
        if subdir == "vision_encoder":
            key = "vision_encoder"
        elif subdir == "llm":
            key = "llm"
        elif subdir == "projector":
            key = "projector"
        cfg = preset.get(key, preset)
        if isinstance(cfg, dict) and "id" not in cfg:
            cfg = {**cfg, "id": value}
        return cfg
    raise ValueError(f"Invalid preset reference: {value!r}")


def _is_mllm_recipe(cfg: dict) -> bool:
    return bool(cfg.get("llm"))


def _recipe_name_from_path(recipe_path: str, mllm_root: str) -> str:
    rel = os.path.relpath(recipe_path, mllm_root)
    if rel.endswith(".yaml"):
        rel = rel[:-5]
    return rel.replace(os.sep, "/")


def _load_mllm_recipe_by_name(name: str) -> tuple[dict, str]:
    recipe_path = resolve_mllm_recipe_path(CONTINUOUS_CONFIGS, name)
    return _load_yaml(recipe_path), recipe_path


def _merge_legacy_runtime(recipe: dict, runtime_cfg: dict) -> dict:
    legacy = {k: recipe.get(k) for k in _RUNTIME_KEYS if k in recipe}
    if legacy:
        return _deep_merge(runtime_cfg, legacy)
    return runtime_cfg


def _load_train_config(
    entry: dict,
    recipe: dict,
    train_config_path: str,
) -> tuple[dict, str]:
    train_path = (
        entry.get("train")
        or recipe.get("train_config")
        or recipe.get("train_base")
        or train_config_path
    )
    train_path = _resolve_vtb_path(train_path)
    train_base = _load_yaml(train_path)
    train_override = recipe.get("train", {})
    if train_override:
        train_base = _deep_merge(train_base, train_override)
    return train_base, train_path


def experiment_slug(
    llm: dict,
    vision_encoder: dict,
    projector: dict | None = None,
) -> str:
    """Directory key: {llm_id}/{vision_encoder_id}[/{projector_id}]."""
    llm_id = llm.get("id")
    if not llm_id:
        raise ValueError("LLM preset must define `id` (e.g. qwen3)")
    vision_id = vision_encoder.get("id")
    if not vision_id:
        raise ValueError("Vision encoder preset must define `id` (e.g. vit_l14_metaclip)")
    base = os.path.join(llm_id, vision_id)
    if not projector:
        return base
    projector_id = projector.get("id") or projector.get("type", "mlp2x")
    if projector_id in ("mlp2x", "mlp2x_gelu", "mlp3x_gelu"):
        return base
    return os.path.join(base, projector_id)


def stage_log_stage_name(stage: str) -> str:
    if stage in ("test", "eval"):
        return "test"
    return stage


def next_stage_run_id(stage_root: str, date_prefix: str) -> int:
    max_id = -1
    prefix = f"{date_prefix}_"
    if os.path.isdir(stage_root):
        for name in os.listdir(stage_root):
            if name in ("latest",):
                continue
            if name.startswith(prefix) and name[len(prefix) :].isdigit():
                max_id = max(max_id, int(name[len(prefix) :]))
    return max_id + 1


def allocate_stage_log_dir(
    output_root: str,
    slug: str,
    stage: str,
    when: datetime | None = None,
) -> str:
    when = when or datetime.now()
    stage_name = stage_log_stage_name(stage)
    stage_root = os.path.join(output_root, slug, stage_name)
    os.makedirs(stage_root, exist_ok=True)

    date_prefix = when.strftime("%m_%d")
    run_id = next_stage_run_id(stage_root, date_prefix)
    run_dir = os.path.join(stage_root, f"{date_prefix}_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    latest_link = os.path.join(stage_root, "latest")
    if os.path.islink(latest_link):
        os.unlink(latest_link)
    elif os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(run_dir, latest_link)

    return run_dir


@dataclass
class RunContext:
    experiment: dict = field(default_factory=dict)
    stages: list = field(default_factory=list)
    llm: dict = field(default_factory=dict)
    vision_encoder: dict = field(default_factory=dict)
    projector: dict = field(default_factory=dict)
    checkpoints: dict = field(default_factory=dict)
    batch: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    runtime: dict = field(default_factory=dict)
    eval: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)

    mllm_recipe_path: str = ""
    mllm_recipe_name: str = ""
    registry_model_name: str = ""
    run_config_path: str = ""
    runtime_config_path: str = ""
    train_config_path: str = ""
    data_config_path: str = ""
    output_dir: str = ""
    results_dir: str = ""
    pretrain_dir: str = ""
    finetune_dir: str = ""
    checkpoint_base: str = ""

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
            micro = int(stage_cfg.get("per_device_train_batch_size", 16))
            accum = int(stage_cfg.get("gradient_accumulation_steps", 1))
        else:
            micro = int(self.batch.get("per_device_train_batch_size", 16))
            accum = int(self.batch.get("gradient_accumulation_steps", 1))
        return {
            "per_device_train_batch_size": micro,
            "gradient_accumulation_steps": accum,
            "num_gpus": num_gpus,
            "global_batch_size": micro * accum * num_gpus,
        }

    @property
    def run_slug(self) -> str:
        return experiment_slug(self.llm, self.vision_encoder, self.projector)

    def resolve_paths(self):
        slug = self.run_slug
        base = self.output.get("base_dir") or self.paths.get("checkpoints_root") or CHECKPOINTS_ROOT
        if not os.path.isabs(base):
            base = os.path.join(VTB_ROOT, base)
        data_ckpt = os.path.join(PRETRAINED_ROOT, "checkpoints")
        trained_root = os.path.normpath(TRAINED_CKPT_ROOT)
        norm_base = os.path.normpath(base)
        if norm_base.startswith(os.path.normpath(data_ckpt)) and not norm_base.startswith(trained_root):
            raise ValueError(
                f"Training output must not go under {data_ckpt} "
                f"(reserved for downloaded weights). Use {CHECKPOINTS_ROOT} instead."
            )
        self.checkpoint_base = base
        self.pretrain_dir = self._resolve_stage_dir(base, slug, "pretrain")
        self.finetune_dir = self._resolve_stage_dir(base, slug, "finetune")
        self.output_dir = os.path.join(LOGS_ROOT, slug)
        self.results_dir = os.path.join(RESULTS_ROOT, slug)
        for d in [self.checkpoint_base, self.pretrain_dir, self.finetune_dir, self.output_dir, self.results_dir]:
            os.makedirs(d, exist_ok=True)
        return self

    def _resolve_stage_dir(self, base: str, slug: str, stage: str) -> str:
        from vision_encoder_eval.mllm.utils.checkpoint_layout import resolve_latest_stage_dir

        return resolve_latest_stage_dir(base, slug, stage)


def load_run_context(
    config_path: str = DEFAULT_RUNTIME_CONFIG,
    train_config_path: str = DEFAULT_TRAIN_CONFIG,
    data_config_path: Optional[str] = None,
    *,
    run_config_path: str | None = None,
    runtime_config_path: str | None = None,
    eval_model: str | None = None,
    mode: str = "continuous",
    mllm_override: str | None = None,
) -> RunContext:
    # Backward-compatible kwargs from older call sites
    if run_config_path is not None:
        config_path = run_config_path
    if runtime_config_path is not None and run_config_path is None:
        config_path = runtime_config_path

    config_path = _resolve_vtb_path(config_path)
    runtime_cfg, runtime_path = resolve_mode_runtime(config_path, mode)

    if _is_mllm_recipe(_load_yaml(config_path)):
        recipe = _load_yaml(config_path)
        recipe_path = config_path
        runtime_cfg = _merge_legacy_runtime(recipe, runtime_cfg)
        runtime_path = config_path
    else:
        recipe = {}
        recipe_path = ""
        mllm_name = mllm_override
        if mllm_name is None:
            names = normalize_model_list(runtime_cfg, "mllm", "mllms")
            if len(names) > 1:
                raise ValueError(
                    "Multiple mllm entries require the pipeline multi-model loop "
                    f"(got {names}). Pass mllm_override for a single recipe."
                )
            mllm_name = names[0] if names else None
        eval_model_name = eval_model or (runtime_cfg.get("eval") or {}).get("model")
        if not mllm_name and not eval_model_name:
            raise ValueError(
                f"Config {config_path} must set `mllm: <recipe_name>` (or a list) for training, "
                "or `eval.model: <name>` from logs/mllm_list for eval-only runs."
            )
        if mllm_name:
            recipe, recipe_path = _load_mllm_recipe_by_name(mllm_name)

    default_train = runtime_cfg.get("train_config") or DEFAULT_TRAIN_CONFIG
    train_cfg_path = train_config_path or default_train
    train_cfg, resolved_train_path = _load_train_config({}, recipe, train_cfg_path)
    data_cfg_path = data_config_path or runtime_cfg.get("data_config", DEFAULT_DATA_CONFIG)

    ctx = RunContext()
    ctx.run_config_path = config_path
    ctx.runtime_config_path = _resolve_vtb_path(runtime_path)
    ctx.mllm_recipe_path = os.path.abspath(recipe_path) if recipe_path else ""
    ctx.mllm_recipe_name = (
        _recipe_name_from_path(recipe_path, os.path.join(CONTINUOUS_CONFIGS, "mllm"))
        if recipe_path
        else ""
    )
    ctx.train_config_path = resolved_train_path
    ctx.data_config_path = _resolve_vtb_path(data_cfg_path)

    ctx.experiment = recipe.get("experiment", {})
    ctx.stages = runtime_cfg.get("stages", [])
    if recipe:
        ctx.llm = _resolve_preset(recipe.get("llm"), "llm")
        ctx.vision_encoder = _resolve_preset(recipe.get("vision_encoder"), "vision_encoder")
        ctx.projector = _resolve_preset(recipe.get("projector"), "projector")
    ctx.checkpoints = runtime_cfg.get("checkpoints", {})
    ctx.batch = merge_recipe_batch(runtime_cfg.get("batch", {}), recipe.get("batch"))
    ctx.data = load_data_config(ctx.data_config_path)
    ctx.output = runtime_cfg.get("output", {})
    ctx.runtime = runtime_cfg.get("runtime", {})
    ctx.eval = _deep_merge(runtime_cfg.get("eval", {}), recipe.get("eval", {}))
    if eval_model:
        ctx.eval["model"] = eval_model
    ctx.train = {
        "pretrain": dict(train_cfg.get("pretrain") or {}),
        "finetune": dict(train_cfg.get("finetune") or {}),
    }
    runtime_train = runtime_cfg.get("train") or {}
    if runtime_train:
        ctx.train = _deep_merge(ctx.train, runtime_train)
    ctx.paths = train_cfg.get("paths", {})
    ctx = ctx.resolve_paths()

    from vision_encoder_eval.mllm.utils.mllm_registry import (
        apply_default_eval_from_train_mllm,
        apply_eval_model_registry,
        get_registry_entry,
    )

    if not ctx.llm and ctx.eval.get("model"):
        registry_entry = get_registry_entry(ctx.eval["model"])
        train_recipe_name = ctx.eval.get("mllm") or registry_entry.get("mllm_recipe")
        if train_recipe_name:
            bootstrap_recipe, bootstrap_path = _load_mllm_recipe_by_name(train_recipe_name)
            ctx.llm = _resolve_preset(bootstrap_recipe.get("llm"), "llm")
            ctx.vision_encoder = _resolve_preset(
                bootstrap_recipe.get("vision_encoder"), "vision_encoder"
            )
            ctx.projector = _resolve_preset(bootstrap_recipe.get("projector"), "projector")
            ctx.mllm_recipe_name = train_recipe_name
            ctx.mllm_recipe_path = os.path.abspath(bootstrap_path)
            ctx.experiment = bootstrap_recipe.get("experiment", ctx.experiment)
            ctx.eval = _deep_merge(ctx.eval, bootstrap_recipe.get("eval", {}))

    eval_mllm_name = ctx.eval.get("mllm")
    if eval_mllm_name and eval_mllm_name != ctx.mllm_recipe_name:
        eval_recipe, eval_recipe_path = _load_mllm_recipe_by_name(eval_mllm_name)
        ctx.llm = _resolve_preset(eval_recipe.get("llm"), "llm")
        ctx.vision_encoder = _resolve_preset(eval_recipe.get("vision_encoder"), "vision_encoder")
        ctx.projector = _resolve_preset(eval_recipe.get("projector"), "projector")
        ctx.eval = _deep_merge(ctx.eval, eval_recipe.get("eval", {}))
        if not ctx.experiment:
            ctx.experiment = eval_recipe.get("experiment", {})
        ctx.mllm_recipe_path = os.path.abspath(eval_recipe_path)

    apply_default_eval_from_train_mllm(ctx)
    apply_eval_model_registry(ctx)
    return ctx


@dataclass
class ExperimentConfig:
    name: str = ""
    description: str = ""
    llm: dict = field(default_factory=dict)
    vision_encoder: dict = field(default_factory=dict)
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
    ctx = load_run_context(run_config_path, eval_model=eval_model)
    resolved = pick_eval_checkpoint_dir(ctx, ctx.llm["model_name_or_path"])

    exp = ExperimentConfig()
    exp.name = ctx.registry_model_name or ctx.run_slug
    exp.description = ctx.experiment.get("description", "")
    exp.llm = ctx.llm
    exp.vision_encoder = ctx.vision_encoder
    exp.projector = ctx.projector
    exp.training = ctx.train.get("finetune", {})
    exp.run_dir = os.path.dirname(ctx.finetune_dir)
    exp.checkpoint_dir = resolved.path
    exp.output_dir = ctx.output_dir
    exp.results_dir = ctx.results_dir
    exp.eval_data_dir = ctx.data.get("eval", {}).get("lmudata_dir", os.path.join(DATASETS_ROOT, "LMUData"))
    return exp.resolve()
