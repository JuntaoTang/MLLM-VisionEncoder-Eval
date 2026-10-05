"""Shared plumbing for the COCO-2K alignment-probing control experiment.

Mirrors the way VTB builds vision towers / discrete tokenizers so that the
features fed to the alignment layer are exactly the features the MLLM pipeline
would see, and mirrors SAIL's alignment layer + SigLIP objective on the other
side.
"""

from __future__ import annotations

from vision_encoder_eval.core.runtime import asset_path, mllm_configs_root

import ast
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROBE_ROOT = Path(__file__).resolve().parent
VTB_ROOT = Path(os.environ.get("VTB_ROOT") or asset_path('mllm', '')).resolve()
TOKENIZER_LIST = Path(os.environ.get("ALIGN_TOKENIZER_LIST") or asset_path('tokenizer_list', ''))

COCO_TSV = asset_path('datasets', 'instructions/test/MSCOCO_KARPATHY_TEST.tsv')
CLIP_PROCESSOR = asset_path('download', 'tokenizer/continuous/clip-vit-large-patch14')

DATA_DIR = Path(os.environ.get("ALIGN_DATA_DIR") or asset_path('runtime','alignment/data'))
CACHE_DIR = Path(os.environ.get("ALIGN_CACHE_DIR") or asset_path('runtime','alignment/cache'))
RESULTS_DIR = Path(os.environ.get("ALIGN_RESULTS_DIR") or asset_path('runtime','alignment/results'))
LOGS_DIR = Path(asset_path('runtime','alignment/logs'))

COCO_JSON = DATA_DIR / "coco2k.json"
CC3M_ROOT = asset_path('cc3m', '')
CC3M_CSV = f"{CC3M_ROOT}/cc3m_3long_3short_1raw_captions_url.csv"

# Datasets the probe can encode. "captions" lists the caption fields in the
# order they are stored; index 0 is the main caption, 1+ are extra positives.
DATASETS = {
    "coco2k": {"json": COCO_JSON, "captions": ["c0", "c1", "c2", "c3", "c4"]},
    "cc3m2k": {"json": DATA_DIR / "cc3m2k.json",
               "captions": ["raw_caption", "longSV_captions"]},
    "cc3m15k": {"json": DATA_DIR / "cc3m15k.json",
                "captions": ["raw_caption", "longSV_captions"]},
    "cc3m20k": {"json": DATA_DIR / "cc3m20k.json",
                "captions": ["raw_caption", "longSV_captions"]},
    "cc3m25k": {"json": DATA_DIR / "cc3m25k.json",
                "captions": ["raw_caption", "longSV_captions"]},
    "cc3m10k": {"json": DATA_DIR / "cc3m10k.json",
                "captions": ["raw_caption", "longSV_captions"]},
}


def dataset_json(name: str) -> Path:
    if name not in DATASETS:
        raise SystemExit(f"unknown dataset {name!r}; have {sorted(DATASETS)}")
    return DATASETS[name]["json"]


def load_dataset(name: str) -> dict:
    p = dataset_json(name)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - build it first")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def vision_cache(dataset: str, slug: str) -> Path:
    return CACHE_DIR / "vision" / dataset / slug


def text_cache(dataset: str, llm: str) -> Path:
    return CACHE_DIR / "text" / dataset / llm

# The three LLMs under comparison (paths taken from VTB configs/llm/*.yaml).
LLMS: dict[str, dict[str, Any]] = {
    "qwen25": {
        "path": asset_path('download', 'llm/Qwen2.5-1.5B-Instruct'),
        "display": "Qwen2.5-1.5B-Instruct",
    },
    "qwen3": {
        "path": asset_path('download', 'llm/Qwen3-1.7B'),
        "display": "Qwen3-1.7B",
    },
    "smollm2": {
        "path": asset_path('download', 'llm/SmolLM2-1.7B-Instruct'),
        "display": "SmolLM2-1.7B-Instruct",
    },
}

VISION_CFG_ROOT = Path(mllm_configs_root()) / "continuous" / "vision_encoder"
DISCRETE_CFG_ROOT = Path(mllm_configs_root()) / "discrete" / "tokenizer"


# --------------------------------------------------------------------------- #
# process setup
# --------------------------------------------------------------------------- #
def bootstrap() -> None:
    """Put VTB + its vendored LLaVA on sys.path and apply its offline env."""
    for p in (asset_path('third_party', 'LLaVA-NeXT'), str(VTB_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from vision_encoder_eval.mllm.utils.config import install_offline_hf_env  # noqa: E402

    install_offline_hf_env()
    os.environ.setdefault("VTB_ROOT", str(VTB_ROOT))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# --------------------------------------------------------------------------- #
# tokenizer registry
# --------------------------------------------------------------------------- #
def read_tokenizer_list(path: Path | str = TOKENIZER_LIST) -> list[str]:
    slugs: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # entries look like "12. toklip_s_256"
            slug = line.split(".", 1)[1].strip() if line[0].isdigit() and "." in line else line
            slugs.append(slug)
    return slugs


def resolve_slug(slug: str) -> tuple[str, Path]:
    """Return ("continuous"|"discrete", config path) for a tokenizer slug."""
    cont = sorted(VISION_CFG_ROOT.rglob(f"{slug}.yaml"))
    if cont:
        return "continuous", cont[0]
    disc = DISCRETE_CFG_ROOT / f"{slug}.yaml"
    if disc.exists():
        return "discrete", disc
    raise FileNotFoundError(f"no vision/tokenizer config found for slug {slug!r}")


def load_cfg(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# COCO-2K split
# --------------------------------------------------------------------------- #
def load_coco2k() -> dict:
    if not COCO_JSON.exists():
        raise FileNotFoundError(
            f"{COCO_JSON} missing — run `python build_data.py` first."
        )
    with open(COCO_JSON, encoding="utf-8") as f:
        return json.load(f)


def read_karpathy_tsv(path: str = COCO_TSV) -> list[dict]:
    csv.field_size_limit(10**7)
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = []
    for r in rows:
        caps = ast.literal_eval(r["answer"])
        caps = [c.strip() for c in caps if isinstance(c, str) and c.strip()]
        if len(caps) < 5 or not os.path.exists(r["image_path"]):
            continue
        out.append({"id": r["index"], "image_path": r["image_path"], "captions": caps[:5]})
    return out


# --------------------------------------------------------------------------- #
# continuous vision towers (replicates src/runner/llava_train.py:setup_env)
# --------------------------------------------------------------------------- #
def _install_vision_env(vision: dict) -> None:
    tower_type = vision.get("type", "open_clip_hub")
    weights = vision.get("weights_path")

    processor = vision.get("processor_path") or CLIP_PROCESSOR
    if processor and os.path.isdir(processor):
        os.environ["VTB_CLIP_IMAGE_PROCESSOR"] = processor

    force_qg = vision.get("force_quick_gelu")
    model_name = str(vision.get("model_name") or vision.get("vision_tower") or "")
    if force_qg is None and "siglip" in model_name.lower():
        force_qg = False
    if force_qg is True:
        os.environ["VTB_FORCE_QUICK_GELU"] = "1"
    elif force_qg is False:
        os.environ["VTB_FORCE_QUICK_GELU"] = "0"
    else:
        os.environ.pop("VTB_FORCE_QUICK_GELU", None)

    if weights and tower_type != "hf_clip":
        os.environ["VTB_VISION_WEIGHTS"] = weights
    else:
        os.environ.pop("VTB_VISION_WEIGHTS", None)

    force_image_size = vision.get("force_image_size")
    if force_image_size is not None:
        os.environ["VTB_FORCE_IMAGE_SIZE"] = str(int(force_image_size))
    else:
        os.environ.pop("VTB_FORCE_IMAGE_SIZE", None)

    for key in (
        "VTB_SSL_IMAGE_SIZE",
        "VTB_SSL_LAYERS",
        "VTB_PE_CONFIG",
        "VTB_EUPE_HUB",
        "VTB_PIXIO_HUB",
        "VTB_DINOV3_BACKBONE",
        "VTB_HF_SELECT_FEATURE",
    ):
        os.environ.pop(key, None)

    if tower_type in ("dinov3", "raev2", "ijepa", "pe", "eupe", "pixio"):
        os.environ["VTB_ROOT"] = str(VTB_ROOT)
        image_size = vision.get("image_size") or force_image_size
        if image_size is not None:
            os.environ["VTB_SSL_IMAGE_SIZE"] = str(int(image_size))
        layers = vision.get("layers")
        if layers is not None:
            os.environ["VTB_SSL_LAYERS"] = (
                ".".join(str(int(x)) for x in layers)
                if isinstance(layers, (list, tuple))
                else str(layers)
            )
        # repo roots recorded in the yaml may point at another checkout; only
        # honour them when they actually exist, else fall back to VTB_ROOT.
        for cfg_key, env_keys, sub in (
            ("dinov3_repo", ("VTB_DINOV3_REPO_DIR", "DINOV3_REPO_DIR"), "dinov3"),
            ("pe_repo", ("VTB_PE_REPO_DIR",), "perception_models"),
            ("eupe_repo", ("VTB_EUPE_REPO_DIR",), "eupe"),
            ("pixio_repo", ("VTB_PIXIO_REPO_DIR",), "pixio"),
        ):
            repo = vision.get(cfg_key)
            if not repo or not os.path.isdir(repo):
                repo = str(VTB_ROOT / "third_party" / sub)
            if os.path.isdir(repo):
                for env_key in env_keys:
                    os.environ[env_key] = repo
        pe_config = vision.get("pe_config") or vision.get("model_name")
        if tower_type == "pe" and pe_config:
            os.environ["VTB_PE_CONFIG"] = str(pe_config)
        if vision.get("eupe_hub"):
            os.environ["VTB_EUPE_HUB"] = str(vision["eupe_hub"])
        if vision.get("pixio_hub"):
            os.environ["VTB_PIXIO_HUB"] = str(vision["pixio_hub"])
        if vision.get("dinov3_backbone"):
            os.environ["VTB_DINOV3_BACKBONE"] = str(vision["dinov3_backbone"])

    if tower_type in ("hf", "hf_vision") and vision.get("select_feature"):
        os.environ["VTB_HF_SELECT_FEATURE"] = str(vision["select_feature"])

    if "siglip" in model_name.lower() and tower_type == "open_clip_hub":
        os.environ["VTB_OPENCLIP_PRETRAINED_TAG"] = vision.get("pretrained") or "webli"
    else:
        os.environ.pop("VTB_OPENCLIP_PRETRAINED_TAG", None)


TOKLIP_ROOT = Path(asset_path('package', 'mllm/discrete/model/tokenizers/toklip'))


def _reset_open_clip_namespace(want_toklip: bool) -> None:
    """Make the next `import open_clip` resolve to the right copy.

    TokLIP ships a vendored open_clip (the only one carrying the `*-toklip`
    model configs); every `open_clip_hub` tower needs the site-packages one.
    Both import under the name `open_clip`, so inside one process whichever
    loads first wins for everything after it. Drop the cached modules — and
    TokLIP's sys.path entry — so each build starts from a clean slate.
    """
    root = str(TOKLIP_ROOT)
    for name in [m for m in sys.modules
                 if m == "open_clip" or m.startswith("open_clip.")]:
        del sys.modules[name]
    if want_toklip:
        if root not in sys.path:
            sys.path.insert(0, root)
    else:
        while root in sys.path:
            sys.path.remove(root)


def build_continuous_tower(vision: dict, device: str):
    _reset_open_clip_namespace(want_toklip=False)
    from llava.model.multimodal_encoder.builder import build_vision_tower
    from vision_encoder_eval.mllm.runner.llava_train import _vision_tower_args

    _install_vision_env(vision)
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
    tower = tower.to(device)
    tower.eval()
    tower.requires_grad_(False)
    return tower


def build_discrete_tokenizer(cfg: dict, device: str):
    from vision_encoder_eval.mllm.discrete.model.tokenizers.factory import build_visual_tokenizer
    from vision_encoder_eval.mllm.discrete.model.vision_config import resolve_vis_mode

    _reset_open_clip_namespace(
        want_toklip=(cfg.get("tokenizer") or {}).get("type") == "toklip")
    vis_mode = resolve_vis_mode(cfg)
    tok = build_visual_tokenizer(cfg)
    tok = tok.to(device)
    tok.eval()
    tok.requires_grad_(False)
    return tok, vis_mode


def ensure_dirs() -> None:
    for d in (DATA_DIR, CACHE_DIR, RESULTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
