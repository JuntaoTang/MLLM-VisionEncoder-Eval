#!/usr/bin/env python3
"""Training-free VDE_pre scores for every VTB vision encoder / discrete tokenizer.

Visual Dataset Entropy (Liu et al. 2026) on pre-projector embeddings, extracted
at the same select_layer / vis_mode used in MLLM training.

  python Baseline/VDE/score_vde_pre.py --mode launch
  python Baseline/VDE/score_vde_pre.py --mode worker --gpu 0
  python Baseline/VDE/score_vde_pre.py --mode merge
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

VTB_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, VTB_ROOT)

OUT_DIR = Path(VTB_ROOT) / "Baseline" / "VDE" / "results"
IMAGE_LIST = OUT_DIR / "sample_images.json"
JOBS_PATH = OUT_DIR / "jobs.json"
STATUS_PATH = OUT_DIR / "status.json"
SCORES_CSV = OUT_DIR / "vde_pre_scores.csv"
SCORES_JSON = OUT_DIR / "vde_pre_scores.json"
LLAVA_ROOT = os.path.join(VTB_ROOT, "third_party", "LLaVA-NeXT")
PYTHON = os.environ.get("VTB_PYTHON", sys.executable)
DEFAULT_CLIP_PROCESSOR = "/cache/ckpt/download/tokenizer/continuous/clip-vit-large-patch14"
TEST_IMAGE_ROOT = "/cache/data/images/test"
N_IMAGES = 100
SEED = 42

_SSL_ENV_KEYS = (
    "VTB_VISION_WEIGHTS",
    "VTB_FORCE_IMAGE_SIZE",
    "VTB_FORCE_QUICK_GELU",
    "VTB_CLIP_IMAGE_PROCESSOR",
    "VTB_SSL_IMAGE_SIZE",
    "VTB_SSL_LAYERS",
    "VTB_DINOV3_REPO_DIR",
    "DINOV3_REPO_DIR",
    "VTB_PE_REPO_DIR",
    "VTB_PE_CONFIG",
    "VTB_EUPE_REPO_DIR",
    "VTB_EUPE_HUB",
    "VTB_DINOV3_BACKBONE",
    "VTB_PIXIO_REPO_DIR",
    "VTB_PIXIO_HUB",
    "VTB_HF_SELECT_FEATURE",
    "VTB_OPENCLIP_PRETRAINED_TAG",
)

_POST_QUANT_MODES = frozenset(
    {
        "vilau",
        "toklip_post_quant",
        "bitdance",
        "atoken",
        "uniar",
        "seed_post_quant",
        "qlip_post_quant",
        "tokenflow_post_quant",
    }
)


def log(msg: str) -> None:
    print(msg, flush=True)


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def query_gpus() -> list[dict[str, int]]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        idx, total, free, used = [int(x.strip()) for x in line.split(",")]
        rows.append({"index": idx, "total": total, "free": free, "used": used})
    return rows


def leftover_fraction(phys_idx: int, safety_mib: int = 6144) -> float:
    rows = {r["index"]: r for r in query_gpus()}
    row = rows[int(phys_idx)]
    usable = max(1024, row["free"] - safety_mib)
    return max(0.04, min(0.28, usable / float(row["total"])))


def sample_images(n: int = N_IMAGES, seed: int = SEED) -> list[str]:
    preferred = [
        "COCO_VAL",
        "GQA_TEST_BALANCED",
        "VizWiz",
        "Flickr30k",
        "ChartQA_TEST",
        "DocVQA_VAL",
        "TextVQA_VAL",
        "POPE",
        "MME",
        "ScienceQA_TEST",
    ]
    buckets: list[list[str]] = []
    for name in preferred:
        folder = os.path.join(TEST_IMAGE_ROOT, name)
        if not os.path.isdir(folder):
            continue
        files: list[str] = []
        for root, _dirs, fnames in os.walk(folder):
            for fn in fnames:
                if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                    files.append(os.path.join(root, fn))
        if files:
            buckets.append(sorted(files))
    if not buckets:
        files = []
        for root, _dirs, fnames in os.walk(TEST_IMAGE_ROOT):
            for fn in fnames:
                if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    files.append(os.path.join(root, fn))
        if not files:
            raise FileNotFoundError(f"No test images under {TEST_IMAGE_ROOT}")
        buckets = [sorted(files)]

    rng = random.Random(seed)
    per = max(1, n // len(buckets))
    picked: list[str] = []
    for bucket in buckets:
        k = min(per, len(bucket))
        picked.extend(rng.sample(bucket, k))
    leftover = [p for bucket in buckets for p in bucket if p not in set(picked)]
    rng.shuffle(leftover)
    while len(picked) < n and leftover:
        picked.append(leftover.pop())
    rng.shuffle(picked)

    from PIL import Image

    ok: list[str] = []
    candidates = picked + leftover
    for path in candidates:
        if path in ok:
            continue
        try:
            with Image.open(path) as im:
                im.convert("RGB")
            ok.append(path)
        except Exception:
            continue
        if len(ok) >= n:
            break
    if len(ok) < max(16, n // 4):
        raise RuntimeError(f"Only opened {len(ok)} test images")
    return ok[:n]


def size_score(job: dict) -> int:
    s = f"{job.get('id', '')} {job.get('image_size', '')}".lower()
    score = 0
    if "7b" in s:
        score += 1000
    if "5b" in s:
        score += 800
    if "3b" in s:
        score += 600
    if "2b" in s or "vit1b" in s:
        score += 400
    if "g14" in s or "giant" in s or "gopt" in s:
        score += 500
    if "h14" in s or "vith" in s or "huge" in s:
        score += 250
    if "518" in s or "512" in s:
        score += 80
    if "448" in s or "384" in s:
        score += 40
    if job.get("id") == "uniar_bsq":
        score += 200
    return score


def discover_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    ve_root = Path(VTB_ROOT) / "configs" / "continuous" / "vision_encoder"
    for path in sorted(ve_root.rglob("*.yaml")):
        data = load_yaml(str(path))
        ve = data.get("vision_encoder") or data
        jobs.append(
            {
                "id": ve.get("id") or path.stem,
                "kind": "continuous",
                "config_path": str(path),
                "type": ve.get("type"),
                "select_layer": ve.get("select_layer", -2),
                "select_feature": ve.get("select_feature", "patch"),
                "image_size": ve.get("image_size") or ve.get("force_image_size"),
                "display_name": ve.get("display_name"),
            }
        )
    tok_root = Path(VTB_ROOT) / "configs" / "discrete" / "tokenizer"
    for path in sorted(tok_root.glob("*.yaml")):
        data = load_yaml(str(path))
        tok = data.get("tokenizer") or data
        jobs.append(
            {
                "id": tok.get("id") or path.stem,
                "kind": "discrete",
                "config_path": str(path),
                "type": tok.get("type"),
                "select_layer": None,
                "select_feature": "post_quant",
                "image_size": tok.get("image_size"),
                "display_name": tok.get("display_name"),
            }
        )
    return jobs


def vde_pre(z) -> dict[str, float]:
    import torch

    z = z.float()
    if z.ndim != 2:
        raise ValueError(f"VDE expects [N, D], got {tuple(z.shape)}")
    n, d = z.shape
    mu = z.mean(dim=0, keepdim=True)
    zbar = z - mu
    zbar = zbar / (zbar.norm(dim=1, keepdim=True) + 1e-8)
    gram = zbar @ zbar.T
    a = gram / gram.trace().clamp_min(1e-12)
    a = 0.5 * (a + a.T)
    eig = torch.linalg.eigvalsh(a).clamp_min(0)
    eig = eig / eig.sum().clamp_min(1e-12)
    entropy = float(-(eig * (eig + 1e-12).log()).sum().item())
    hhat = entropy / math.log(max(2, min(n, d)))
    return {
        "vde_pre": hhat,
        "vde_pre_raw": entropy,
        "n": n,
        "dim": d,
        "effective_rank": float((eig > 1e-6).sum().item()),
    }


def clear_vision_env() -> None:
    for key in _SSL_ENV_KEYS:
        os.environ.pop(key, None)


def apply_vision_env(ve: dict) -> None:
    from src.utils.config import install_offline_hf_env

    install_offline_hf_env()
    clear_vision_env()
    os.environ["VTB_ROOT"] = VTB_ROOT

    tower_type = ve.get("type", "open_clip_hub")
    weights = ve.get("weights_path")
    processor = ve.get("processor_path") or DEFAULT_CLIP_PROCESSOR
    if processor and os.path.isdir(str(processor)):
        os.environ["VTB_CLIP_IMAGE_PROCESSOR"] = str(processor)

    force_qg = ve.get("force_quick_gelu")
    model_name = str(ve.get("model_name") or ve.get("vision_tower") or "")
    if force_qg is None and "siglip" in model_name.lower():
        force_qg = False
    if force_qg is True:
        os.environ["VTB_FORCE_QUICK_GELU"] = "1"
    elif force_qg is False:
        os.environ["VTB_FORCE_QUICK_GELU"] = "0"
    if "siglip" in model_name.lower() and ve.get("pretrained"):
        os.environ["VTB_OPENCLIP_PRETRAINED_TAG"] = str(ve["pretrained"])

    if weights and tower_type != "hf_clip":
        os.environ["VTB_VISION_WEIGHTS"] = str(weights)
    force_image_size = ve.get("force_image_size")
    if force_image_size is not None:
        os.environ["VTB_FORCE_IMAGE_SIZE"] = str(int(force_image_size))

    if tower_type in ("dinov3", "raev2", "ijepa", "pe", "eupe", "pixio"):
        image_size = ve.get("image_size") or force_image_size
        if image_size is not None:
            os.environ["VTB_SSL_IMAGE_SIZE"] = str(int(image_size))
        layers = ve.get("layers")
        if layers is not None:
            if isinstance(layers, (list, tuple)):
                os.environ["VTB_SSL_LAYERS"] = ".".join(str(int(x)) for x in layers)
            else:
                os.environ["VTB_SSL_LAYERS"] = str(layers)
        dinov3_repo = ve.get("dinov3_repo") or os.path.join(VTB_ROOT, "third_party", "dinov3")
        if os.path.isdir(dinov3_repo):
            os.environ["VTB_DINOV3_REPO_DIR"] = dinov3_repo
            os.environ["DINOV3_REPO_DIR"] = dinov3_repo
        pe_repo = ve.get("pe_repo") or os.path.join(VTB_ROOT, "third_party", "perception_models")
        if os.path.isdir(pe_repo):
            os.environ["VTB_PE_REPO_DIR"] = pe_repo
        pe_config = ve.get("pe_config") or ve.get("model_name")
        if pe_config:
            os.environ["VTB_PE_CONFIG"] = str(pe_config)
        eupe_repo = ve.get("eupe_repo") or os.path.join(VTB_ROOT, "third_party", "eupe")
        if os.path.isdir(eupe_repo):
            os.environ["VTB_EUPE_REPO_DIR"] = eupe_repo
        if ve.get("eupe_hub"):
            os.environ["VTB_EUPE_HUB"] = str(ve["eupe_hub"])
        if ve.get("dinov3_backbone"):
            os.environ["VTB_DINOV3_BACKBONE"] = str(ve["dinov3_backbone"])
        pixio_repo = ve.get("pixio_repo") or os.path.join(VTB_ROOT, "third_party", "pixio")
        if os.path.isdir(pixio_repo):
            os.environ["VTB_PIXIO_REPO_DIR"] = pixio_repo
        if ve.get("pixio_hub"):
            os.environ["VTB_PIXIO_HUB"] = str(ve["pixio_hub"])
    if tower_type in ("hf", "hf_vision"):
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        if ve.get("select_feature"):
            os.environ["VTB_HF_SELECT_FEATURE"] = str(ve["select_feature"])


class TowerArgs:
    def __init__(self, **kwargs):
        self.unfreeze_mm_vision_tower = False
        self.mm_tunable_parts = ""
        self.s2 = False
        for k, v in kwargs.items():
            setattr(self, k, v)


def ensure_llava_path() -> None:
    if LLAVA_ROOT not in sys.path:
        sys.path.insert(0, LLAVA_ROOT)


def load_continuous_tower(ve: dict, device, dtype):
    from src.runner.llava_train import _vision_tower_args

    apply_vision_env(ve)
    ensure_llava_path()
    from llava.model.multimodal_encoder.builder import build_vision_tower

    tower_name, pretrained, select_layer = _vision_tower_args(ve)
    image_size = ve.get("image_size") or ve.get("force_image_size")
    args = TowerArgs(
        vision_tower=tower_name,
        mm_vision_tower=tower_name,
        vision_tower_pretrained=pretrained,
        mm_vision_select_layer=int(select_layer),
        mm_vision_select_feature=str(ve.get("select_feature", "patch")),
        mm_vision_image_size=int(image_size) if image_size is not None else None,
    )
    tower = build_vision_tower(args, delay_load=True)
    if hasattr(tower, "load_model"):
        try:
            tower.load_model(device_map=None)
        except TypeError:
            tower.load_model()
    tower.eval()
    import torch

    # Training-side SSLVisionTower keeps EUPE ViT RoPE in fp32. Casting the
    # whole tower to bf16 then feeding fp32 activations fails with a dtype
    # mismatch on conv/linear bias.
    run_dtype = torch.float32 if str(ve.get("type", "")).lower() == "eupe" else dtype
    tower.to(device=device, dtype=run_dtype)
    for p in tower.parameters():
        p.requires_grad_(False)
    return tower


def preprocess_continuous(tower, images):
    import torch

    proc = getattr(tower, "image_processor", None)
    if proc is None:
        raise RuntimeError("vision tower has no image_processor")
    try:
        out = proc(images, return_tensors="pt")
        if hasattr(out, "pixel_values"):
            return out.pixel_values
        if isinstance(out, dict) and "pixel_values" in out:
            return out["pixel_values"]
    except Exception:
        pass
    try:
        out = proc.preprocess(images, return_tensors="pt")
        return out["pixel_values"]
    except Exception:
        tensors = [proc.preprocess(im, return_tensors="pt")["pixel_values"] for im in images]
        return torch.cat(tensors, dim=0)


def pils_to_tensor(pils, size: int):
    import torch
    import torchvision.transforms.functional as TF

    tensors = []
    for im in pils:
        im = im.convert("RGB")
        im = TF.resize(im, [size, size], interpolation=TF.InterpolationMode.BICUBIC)
        tensors.append(TF.to_tensor(im))
    return torch.stack(tensors, 0)


def load_discrete_tokenizer(tok_cfg: dict, device, dtype):
    from src.discrete.model.tokenizers.factory import build_visual_tokenizer
    from src.discrete.model.vision_config import resolve_vis_mode

    if tok_cfg.get("type") == "toklip":
        # Vendored TokLIP open_clip configs are not in site-packages. If a
        # previous continuous job already imported OpenCLIP, drop it so the
        # TokLIP package on sys.path is used.
        toklip_root = os.path.join(
            VTB_ROOT, "src", "discrete", "model", "tokenizers", "toklip"
        )
        for key in [k for k in list(sys.modules) if k == "open_clip" or k.startswith("open_clip.")]:
            del sys.modules[key]
        if toklip_root in sys.path:
            sys.path.remove(toklip_root)
        sys.path.insert(0, toklip_root)

    cfg = {"tokenizer": tok_cfg}
    vis_mode = resolve_vis_mode(cfg)
    tok = build_visual_tokenizer(cfg)
    tok.eval()
    tok.to(device=device)
    for p in tok.parameters():
        p.requires_grad_(False)
    return tok, vis_mode


def encode_discrete(tok, vis_mode: str, pils, device, dtype):
    import torch

    size = int(getattr(tok, "image_size", 256))
    pixels = None
    if hasattr(tok, "preprocess"):
        try:
            pixels = tok.preprocess(pils)
        except Exception:
            pixels = None
    if pixels is None:
        pixels = pils_to_tensor(pils, size)
        if hasattr(tok, "preprocess"):
            try:
                pixels = tok.preprocess(pixels)
            except Exception:
                pass
    if not torch.is_tensor(pixels):
        raise TypeError(f"tokenizer preprocess returned {type(pixels)}")
    pixels = pixels.to(device=device)
    if pixels.dtype in (torch.float32, torch.float16, torch.bfloat16):
        pixels = pixels.to(dtype=dtype)

    if vis_mode == "unitok_quant" and hasattr(tok, "encode_quant_features"):
        feats = tok.encode_quant_features(pixels)
    elif vis_mode == "unitok":
        feats = tok.encode(pixels)
        if feats.dtype in (torch.int32, torch.int64, torch.long):
            raise RuntimeError("unitok encode returned indices")
    elif vis_mode in _POST_QUANT_MODES and hasattr(tok, "encode_post_quant_features"):
        feats = tok.encode_post_quant_features(pixels)
    elif hasattr(tok, "encode_post_quant_features"):
        feats = tok.encode_post_quant_features(pixels)
    else:
        feats = tok.encode(pixels)
        if feats.dtype in (torch.int32, torch.int64, torch.long):
            raise RuntimeError(f"{vis_mode} encode returned discrete indices")
    return feats


def mean_pool_tokens(feats):
    import torch

    if isinstance(feats, (list, tuple)):
        feats = feats[0]
    if not torch.is_tensor(feats):
        raise TypeError(f"features type {type(feats)}")
    if feats.ndim == 2:
        return feats.mean(dim=0)
    if feats.ndim == 3:
        return feats.mean(dim=1)
    if feats.ndim == 4:
        return feats.flatten(2).mean(dim=2)
    raise ValueError(f"unexpected feature rank {feats.ndim} shape={tuple(feats.shape)}")


def choose_batch_size(job: dict) -> int:
    ident = (job.get("id") or "").lower()
    size = int(job.get("image_size") or 224)
    if "7b" in ident or size >= 448:
        return 1
    if size >= 336:
        return 2
    return 4


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_done_ids(jsonl: Path) -> set[str]:
    done: set[str] = set()
    if not jsonl.is_file():
        return done
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "ok" and rec.get("id"):
                done.add(rec["id"])
    return done


def score_one(job: dict, image_paths: list[str], device, dtype) -> dict:
    import torch
    from PIL import Image

    t0 = time.time()
    pils = [Image.open(p).convert("RGB") for p in image_paths]
    embeddings = []
    extra: dict[str, Any] = {}
    bs = choose_batch_size(job)

    if job["kind"] == "continuous":
        data = load_yaml(job["config_path"])
        ve = data.get("vision_encoder") or data
        extra["select_layer"] = ve.get("select_layer", -2)
        extra["select_feature"] = ve.get("select_feature", "patch")
        extra["type"] = ve.get("type")
        tower = load_continuous_tower(ve, device, dtype)
        try:
            i = 0
            while i < len(pils):
                cur = bs
                while True:
                    try:
                        batch = pils[i : i + cur]
                        pixels = preprocess_continuous(tower, batch)
                        pixels = pixels.to(device=device, dtype=dtype)
                        with torch.inference_mode():
                            feats = tower(pixels)
                        embeddings.append(mean_pool_tokens(feats).float().cpu())
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if cur == 1:
                            raise
                        cur = max(1, cur // 2)
                i += cur
        finally:
            del tower
            gc.collect()
            torch.cuda.empty_cache()
    else:
        data = load_yaml(job["config_path"])
        tok_cfg = data.get("tokenizer") or data
        tok, vis_mode = load_discrete_tokenizer(tok_cfg, device, dtype)
        extra["vis_mode"] = vis_mode
        extra["type"] = tok_cfg.get("type")
        extra["select_layer"] = None
        extra["select_feature"] = vis_mode
        try:
            i = 0
            while i < len(pils):
                cur = bs
                while True:
                    try:
                        batch = pils[i : i + cur]
                        with torch.inference_mode():
                            feats = encode_discrete(tok, vis_mode, batch, device, dtype)
                        embeddings.append(mean_pool_tokens(feats).float().cpu())
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if cur == 1:
                            raise
                        cur = max(1, cur // 2)
                i += cur
        finally:
            del tok
            gc.collect()
            torch.cuda.empty_cache()

    z = embeddings[0] if len(embeddings) == 1 else torch.cat(embeddings, dim=0)
    if z.ndim == 1:
        z = z.unsqueeze(0)
    stats = vde_pre(z)
    return {
        "id": job["id"],
        "kind": job["kind"],
        "display_name": job.get("display_name"),
        "config_path": job["config_path"],
        "n_images": int(z.shape[0]),
        "seconds": round(time.time() - t0, 2),
        "status": "ok",
        **extra,
        **stats,
    }


def run_worker(args: argparse.Namespace) -> None:
    import torch

    from src.utils.config import install_offline_hf_env

    install_offline_hf_env()
    os.environ["VTB_ROOT"] = VTB_ROOT
    phys = os.environ.get("CUDA_VISIBLE_DEVICES", str(args.gpu)).split(",")[0]
    frac = leftover_fraction(phys, safety_mib=args.safety_mib)
    torch.cuda.set_per_process_memory_fraction(frac, device=0)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    log(f"[worker gpu={phys}] leftover_fraction={frac:.3f}")

    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    images = json.loads(Path(args.images).read_text(encoding="utf-8"))
    want = set(args.job_ids) if args.job_ids else {j["id"] for j in jobs}
    jsonl = OUT_DIR / f"shard{phys}.jsonl"
    done = load_done_ids(jsonl)
    mine = [j for j in jobs if j["id"] in want]
    for job in mine:
        if job["id"] in done:
            log(f"[skip] {job['id']}")
            continue
        log(f"[start] {job['kind']} {job['id']}")
        try:
            rec = score_one(job, images, device, dtype)
            rec["gpu"] = int(phys)
            append_jsonl(jsonl, rec)
            log(
                f"[ok] {job['id']} vde_pre={rec['vde_pre']:.4f} "
                f"dim={rec['dim']} n={rec['n']} {rec['seconds']}s"
            )
        except Exception as exc:
            rec = {
                "id": job["id"],
                "kind": job["kind"],
                "config_path": job["config_path"],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
                "gpu": int(phys),
                "seconds": None,
            }
            append_jsonl(jsonl, rec)
            log(f"[error] {job['id']}: {type(exc).__name__}: {exc}")
        gc.collect()
        torch.cuda.empty_cache()
    log(f"[worker gpu={phys}] done")


def merge_scores() -> list[dict]:
    recs: dict[str, dict] = {}
    for path in sorted(OUT_DIR.glob("shard*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rid = rec.get("id")
                if not rid:
                    continue
                prev = recs.get(rid)
                if prev is None or (rec.get("status") == "ok" and prev.get("status") != "ok"):
                    recs[rid] = rec
    rows = list(recs.values())
    ok = [r for r in rows if r.get("status") == "ok"]
    ok.sort(key=lambda r: float(r.get("vde_pre", -1)), reverse=True)
    err = [r for r in rows if r.get("status") != "ok"]
    ordered = ok + err
    SCORES_JSON.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
    cols = [
        "id",
        "kind",
        "type",
        "select_layer",
        "select_feature",
        "vis_mode",
        "vde_pre",
        "vde_pre_raw",
        "dim",
        "n",
        "effective_rank",
        "seconds",
        "gpu",
        "status",
        "error",
    ]
    with open(SCORES_CSV, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in ordered:
            vals = []
            for c in cols:
                v = r.get(c, "")
                if v is None:
                    v = ""
                vals.append(str(v).replace(",", " ").replace("\n", " "))
            f.write(",".join(vals) + "\n")
    STATUS_PATH.write_text(
        json.dumps(
            {
                "total": len(ordered),
                "ok": len(ok),
                "error": len(err),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ordered


def launch(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not IMAGE_LIST.is_file() or args.resample:
        images = sample_images(args.n_images, args.seed)
        IMAGE_LIST.write_text(json.dumps(images, indent=2), encoding="utf-8")
        log(f"sampled {len(images)} images -> {IMAGE_LIST}")
    else:
        images = json.loads(IMAGE_LIST.read_text(encoding="utf-8"))
        log(f"reuse {len(images)} images from {IMAGE_LIST}")

    jobs = discover_jobs()
    JOBS_PATH.write_text(json.dumps(jobs, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"discovered {len(jobs)} encoders/tokenizers")

    gpus = query_gpus()
    log("GPU leftover: " + ", ".join(f"{g['index']}:{g['free']}MiB" for g in gpus))
    roomy = sorted([g for g in gpus if g["free"] >= 20000], key=lambda g: g["free"], reverse=True)
    if not roomy:
        roomy = sorted(gpus, key=lambda g: g["free"], reverse=True)[:2]
    all_sorted = sorted(gpus, key=lambda g: g["free"], reverse=True)
    huge = [j for j in jobs if size_score(j) >= 400]
    rest = [j for j in jobs if size_score(j) < 400]
    huge.sort(key=size_score, reverse=True)
    rest.sort(key=size_score, reverse=True)
    shards: dict[int, list[str]] = {g["index"]: [] for g in gpus}

    def _assign(job_list: list[dict], gpu_list: list[dict]) -> None:
        loads = {g["index"]: 0 for g in gpu_list}
        for job in job_list:
            gpu = min(gpu_list, key=lambda g: (loads[g["index"]], -g["free"]))
            shards[gpu["index"]].append(job["id"])
            loads[gpu["index"]] += max(1, size_score(job))

    _assign(huge, roomy)
    _assign(rest, all_sorted)

    script = os.path.abspath(__file__)
    procs = []
    for gpu, ids in shards.items():
        if not ids:
            continue
        log_path = OUT_DIR / f"worker_gpu{gpu}.log"
        cmd = [
            PYTHON,
            "-u",
            script,
            "--mode",
            "worker",
            "--gpu",
            str(gpu),
            "--jobs",
            str(JOBS_PATH),
            "--images",
            str(IMAGE_LIST),
            "--safety-mib",
            str(args.safety_mib),
            "--job-ids",
            *ids,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["VTB_ROOT"] = VTB_ROOT
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = VTB_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        log(f"launch gpu{gpu} n={len(ids)} log={log_path}")
        log_f = open(log_path, "a", encoding="utf-8")
        log_f.write(f"\n===== launch {time.strftime('%F %T')} n={len(ids)} =====\n")
        log_f.flush()
        p = subprocess.Popen(
            cmd,
            cwd=VTB_ROOT,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        procs.append({"gpu": gpu, "pid": p.pid, "n": len(ids), "log": str(log_path)})
    (OUT_DIR / "launch.json").write_text(json.dumps(procs, indent=2), encoding="utf-8")
    log(f"launched {len(procs)} workers; pids={[x['pid'] for x in procs]}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["launch", "worker", "merge", "sample"], default="launch")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--jobs", type=str, default=str(JOBS_PATH))
    p.add_argument("--images", type=str, default=str(IMAGE_LIST))
    p.add_argument("--job-ids", nargs="*", default=None)
    p.add_argument("--n-images", type=int, default=N_IMAGES)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--safety-mib", type=int, default=6144)
    p.add_argument("--resample", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "sample":
        images = sample_images(args.n_images, args.seed)
        IMAGE_LIST.write_text(json.dumps(images, indent=2), encoding="utf-8")
        log(f"wrote {len(images)} images")
        return
    if args.mode == "merge":
        rows = merge_scores()
        ok = sum(1 for r in rows if r.get("status") == "ok")
        log(f"merged {len(rows)} rows ({ok} ok)")
        return
    if args.mode == "worker":
        run_worker(args)
        return
    launch(args)


if __name__ == "__main__":
    main()
