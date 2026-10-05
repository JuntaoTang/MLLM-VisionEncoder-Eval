"""Shared helpers for reproducing Law of Vision Representation on VTB models."""

from __future__ import annotations

from vision_encoder_eval.core.runtime import asset_path, mllm_configs_root

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

REPRO_ROOT = Path(__file__).resolve().parent
LAW_ROOT = REPRO_ROOT.parent
VTB_ROOT = Path(asset_path('mllm', ''))
LLAVA_ROOT = Path(asset_path('third_party', 'LLaVA-NeXT'))
FINISH_JSON = VTB_ROOT / "results" / "finish.json"
MLLM_CFG_ROOT = Path(mllm_configs_root()) / "continuous" / "mllm"
VISION_CFG_ROOT = Path(mllm_configs_root()) / "continuous" / "vision_encoder"
LLM_CFG_ROOT = Path(mllm_configs_root()) / "llm"
CKPT_ROOT = Path(asset_path('trained', 'continuous'))
PRETRAIN_JSON = asset_path('datasets', 'instructions/pretrain/blip_laion_cc_sbu_558k.json')
PRETRAIN_IMG = asset_path('datasets', 'images/pretrain')
CLIP_PROC = asset_path('download', 'tokenizer/continuous/clip-vit-large-patch14')

RESULTS_DIR = Path(os.environ.get('VEE_LAW_RESULTS') or asset_path('runtime','law/results'))
LOGS_DIR = Path(os.environ.get('VEE_LAW_LOGS') or asset_path('runtime','law/logs'))
CACHE_LAW = Path(asset_path('runtime', 'law_ac'))
FEAT_DIR = CACHE_LAW / "features"
DATA_DIR = Path(asset_path('datasets', ''))
SPAIR_DIR = DATA_DIR / "SPair-71k"
SPAIR_TAR = DATA_DIR / "SPair-71k.tar.gz"

A_N_DATAPOINTS = 100
ANNO_SIZE = 840
SOFT_EVAL_WINDOW = 5


def ensure_vtb_path() -> None:
    for p in (str(LLAVA_ROOT), str(VTB_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("VTB_ROOT", str(VTB_ROOT))
    os.environ.setdefault("VTB_CLIP_IMAGE_PROCESSOR", CLIP_PROC)
    from vision_encoder_eval.mllm.utils.config import CUDA_STUB_HOME
    os.environ.setdefault("CUDA_HOME", CUDA_STUB_HOME)
    os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_yaml(path: Path | str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_finish() -> dict[str, Any]:
    with open(FINISH_JSON) as f:
        return json.load(f)


def _find_preset_yaml(root: Path, name: str) -> Path:
    direct = root / f"{name}.yaml"
    if direct.is_file():
        return direct
    hits = list(root.rglob(f"{name}.yaml"))
    if not hits:
        raise FileNotFoundError(f"preset yaml not found: {root} / {name}")
    return hits[0]


def load_vision_cfg(vision_id: str) -> dict:
    data = load_yaml(_find_preset_yaml(VISION_CFG_ROOT, vision_id))
    return data.get("vision_encoder") or data


def load_llm_cfg(llm_id: str) -> dict:
    data = load_yaml(_find_preset_yaml(LLM_CFG_ROOT, llm_id))
    return data.get("llm") or data


def recipe_path_from_finish_key(key: str) -> Path:
    # continuous/qwen3/clip_openai__l14_mlp2x -> configs/continuous/mllm/qwen3/clip_openai__l14_mlp2x.yaml
    rel = key.split("/", 1)[1] if key.startswith("continuous/") else key
    path = MLLM_CFG_ROOT / f"{rel}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"recipe yaml missing: {path}")
    return path


def ckpt_has_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    markers = (
        "mm_projector.bin",
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    if any((path / m).is_file() for m in markers):
        return True
    ckpts = sorted(
        [p for p in path.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-", 1)[1]) if p.name.split("-", 1)[1].isdigit() else -1,
    )
    return any(ckpt_has_weights(p) for p in ckpts)


def resolve_stage_dir(stage_dir: Path) -> Path | None:
    if not stage_dir.is_dir():
        return None
    if ckpt_has_weights(stage_dir):
        ckpts = sorted(
            [p for p in stage_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
            key=lambda p: int(p.name.split("-", 1)[1]) if p.name.split("-", 1)[1].isdigit() else -1,
        )
        if ckpts and ckpt_has_weights(ckpts[-1]) and not (stage_dir / "mm_projector.bin").is_file() and not (
            stage_dir / "model.safetensors"
        ).is_file():
            return ckpts[-1]
        return stage_dir
    ckpts = sorted(
        [p for p in stage_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-", 1)[1]) if p.name.split("-", 1)[1].isdigit() else -1,
    )
    for p in reversed(ckpts):
        if ckpt_has_weights(p):
            return p
    return None


def inventory_models() -> list[dict[str, Any]]:
    finish = load_finish()
    rows = []
    for key, rec in finish.items():
        if not key.startswith("continuous/"):
            continue
        try:
            rpath = recipe_path_from_finish_key(key)
            recipe = load_yaml(rpath)
        except FileNotFoundError:
            rows.append({"key": key, "error": "missing_recipe", "scores": rec.get("scores")})
            continue
        vision_id = recipe.get("vision_encoder")
        llm_name = recipe.get("llm")
        llm_cfg = load_llm_cfg(llm_name)
        vision_cfg = load_vision_cfg(vision_id)
        run_slug = rec.get("run_slug") or f"{llm_cfg.get('id')}/{vision_cfg.get('id')}"
        pretrain_dir = CKPT_ROOT / run_slug / "pretrain"
        finetune_dir = CKPT_ROOT / run_slug / "finetune"
        pretrain_resolved = resolve_stage_dir(pretrain_dir)
        rows.append(
            {
                "key": key,
                "recipe_yaml": str(rpath),
                "recipe_name": key.split("/", 1)[1],
                "llm_id": llm_cfg.get("id"),
                "llm_path": llm_cfg.get("model_name_or_path"),
                "llm_version": llm_cfg.get("model_version"),
                "vision_id": vision_cfg.get("id") or vision_id,
                "run_slug": run_slug,
                "pretrain_dir": str(pretrain_resolved) if pretrain_resolved else str(pretrain_dir),
                "pretrain_ok": pretrain_resolved is not None,
                "finetune_ok": resolve_stage_dir(finetune_dir) is not None,
                "scores": rec.get("scores") or {},
                "average": (rec.get("scores") or {}).get("average"),
            }
        )
    return rows


def unique_vision_ids(rows: list[dict[str, Any]]) -> list[str]:
    seen = []
    for r in rows:
        vid = r.get("vision_id")
        if vid and vid not in seen:
            seen.append(vid)
    return seen


def apply_vision_env(vision: dict) -> None:
    ensure_vtb_path()
    for k in (
        "VTB_VISION_WEIGHTS",
        "VTB_FORCE_QUICK_GELU",
        "VTB_OPENCLIP_PRETRAINED_TAG",
        "VTB_FORCE_IMAGE_SIZE",
        "VTB_SSL_IMAGE_SIZE",
        "VTB_SSL_LAYERS",
        "VTB_HF_SELECT_FEATURE",
        "VTB_PE_CONFIG",
        "VTB_EUPE_HUB",
        "VTB_PIXIO_HUB",
        "VTB_DINOV3_BACKBONE",
    ):
        os.environ.pop(k, None)

    os.environ.setdefault("VTB_CLIP_IMAGE_PROCESSOR", CLIP_PROC)
    proc = vision.get("processor_path")
    if proc and os.path.isdir(str(proc)):
        os.environ["VTB_CLIP_IMAGE_PROCESSOR"] = str(proc)

    tower_type = vision.get("type", "open_clip_hub")
    weights = vision.get("weights_path")
    model_name = str(vision.get("model_name") or vision.get("vision_tower") or "")
    force_qg = vision.get("force_quick_gelu")
    if force_qg is None and "siglip" in model_name.lower():
        force_qg = False
    if force_qg is True:
        os.environ["VTB_FORCE_QUICK_GELU"] = "1"
    elif force_qg is False:
        os.environ["VTB_FORCE_QUICK_GELU"] = "0"
    if "siglip" in model_name.lower():
        os.environ["VTB_OPENCLIP_PRETRAINED_TAG"] = vision.get("pretrained") or "webli"

    if weights and tower_type != "hf_clip":
        os.environ["VTB_VISION_WEIGHTS"] = str(weights)

    force_image_size = vision.get("force_image_size")
    if force_image_size is not None:
        os.environ["VTB_FORCE_IMAGE_SIZE"] = str(int(force_image_size))

    if tower_type in ("dinov3", "raev2", "ijepa", "pe", "eupe", "pixio"):
        os.environ["VTB_ROOT"] = str(VTB_ROOT)
        image_size = vision.get("image_size") or force_image_size
        if image_size is not None:
            os.environ["VTB_SSL_IMAGE_SIZE"] = str(int(image_size))
        layers = vision.get("layers")
        if layers is not None:
            if isinstance(layers, (list, tuple)):
                os.environ["VTB_SSL_LAYERS"] = ".".join(str(int(x)) for x in layers)
            else:
                os.environ["VTB_SSL_LAYERS"] = str(layers)
        mapping = {
            "dinov3_repo": ("VTB_DINOV3_REPO_DIR", VTB_ROOT / "third_party" / "dinov3"),
            "pe_repo": ("VTB_PE_REPO_DIR", VTB_ROOT / "third_party" / "perception_models"),
            "eupe_repo": ("VTB_EUPE_REPO_DIR", VTB_ROOT / "third_party" / "eupe"),
            "pixio_repo": ("VTB_PIXIO_REPO_DIR", VTB_ROOT / "third_party" / "pixio"),
        }
        for key, (env_k, default) in mapping.items():
            path = vision.get(key) or str(default)
            if os.path.isdir(path):
                os.environ[env_k] = path
                if env_k == "VTB_DINOV3_REPO_DIR":
                    os.environ["DINOV3_REPO_DIR"] = path
        pe_config = vision.get("pe_config") or vision.get("model_name")
        if pe_config:
            os.environ["VTB_PE_CONFIG"] = str(pe_config)
        if vision.get("eupe_hub"):
            os.environ["VTB_EUPE_HUB"] = str(vision["eupe_hub"])
        if vision.get("pixio_hub"):
            os.environ["VTB_PIXIO_HUB"] = str(vision["pixio_hub"])
        if vision.get("dinov3_backbone"):
            os.environ["VTB_DINOV3_BACKBONE"] = str(vision["dinov3_backbone"])
    if tower_type in ("hf", "hf_vision") and vision.get("select_feature"):
        os.environ["VTB_HF_SELECT_FEATURE"] = str(vision["select_feature"])


def build_vision_tower(vision: dict, device: str):
    from vision_encoder_eval.mllm.runner.llava_train import _vision_tower_args
    from llava.model.multimodal_encoder.builder import build_vision_tower

    apply_vision_env(vision)
    tower_name, pretrained, select_layer = _vision_tower_args(vision)
    args = SimpleNamespace(
        vision_tower=tower_name,
        vision_tower_pretrained=pretrained,
        mm_vision_select_layer=int(select_layer),
        mm_vision_select_feature=vision.get("select_feature", "patch"),
        mm_vision_image_size=vision.get("image_size"),
        unfreeze_mm_vision_tower=False,
    )
    tower = build_vision_tower(args)
    if hasattr(tower, "load_model") and not getattr(tower, "is_loaded", True):
        tower.load_model()
    tower = tower.to(device)
    tower.eval()
    for p in tower.parameters():
        p.requires_grad_(False)
    return tower


def tokens_to_grid(feat):
    """(B, N, C) or (B, C, H, W) -> (B, C, H, W) square grid."""
    import math
    import torch
    import torch.nn.functional as F

    if feat.dim() == 4:
        _b, _c, h, w = feat.shape
        if h == w:
            return feat
        side = int(round(math.sqrt(h * w)))
        return F.interpolate(feat, size=(side, side), mode="bilinear", align_corners=False)

    if feat.dim() != 3:
        raise ValueError(f"unexpected feature rank {feat.dim()} shape={tuple(feat.shape)}")
    b, n, c = feat.shape
    side = int(math.sqrt(n))
    if side * side == n:
        return feat.permute(0, 2, 1).reshape(b, c, side, side)
    if (side := int(math.sqrt(n - 1))) * side == n - 1:
        return feat[:, 1:].permute(0, 2, 1).reshape(b, c, side, side)
    # non-square token count: interpolate to nearest square
    side = max(1, int(round(math.sqrt(n))))
    grid = feat.permute(0, 2, 1).reshape(b, c, n, 1)
    return F.interpolate(grid, size=(side, side), mode="bilinear", align_corners=False)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def shard_list(items: list, rank: int, world: int) -> list:
    if world <= 0:
        return list(items)
    rank = int(rank) % int(world)
    return [x for i, x in enumerate(items) if i % world == rank]
