# -*- coding: utf-8 -*-
"""
Feature Extraction (Local Weights) - v5
Fixes remaining 5 failures:
- mc2_*_384: resize model pos_embed to match checkpoint before loading
- siglip2_sm14_384: model expects 378px, not 384px
"""
import sys
import os
import json
import argparse
import time
import re
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Weights and configs of the vision encoders; override with the environment
# variables TOKENIZER_WEIGHTS_ROOT / VTB_CONFIGS_ROOT (see README, "Labels and
# paths").
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_WEIGHTS_ROOT = os.environ.get(
    "TOKENIZER_WEIGHTS_ROOT", os.path.join(_PKG, "tokenizer", "continuous"))
VTB_CONFIGS_ROOT = os.environ.get(
    "VTB_CONFIGS_ROOT",
    os.path.join(_PKG, "configs", "continuous", "vision_encoder"))

try:
    from numpy.core.multiarray import scalar as np_scalar
    torch.serialization.add_safe_globals([np_scalar])
except Exception:
    pass
try:
    torch.serialization.add_safe_globals([np.dtype])
except Exception:
    pass

# ---- multi-layer extraction (cross-layer dynamics, D descriptors) ----
MULTI_LAYER_ENABLED = False           # set by --multi_layer
MULTI_LAYER_FRACS = (0.25, 0.5, 0.75, 0.875, 1.0)   # relative depth (fraction of L)
MULTI_LAYER_ABS_OFFSETS = (-2, -4, -6, -8)          # absolute: last-N blocks
MULTI_LAYER_ABS_FIRST = (0, 1)                      # absolute: first blocks


def get_multi_layer_indices(L, official=None):
    """Union of relative-depth and official-layer-anchored block indices.

    official: absolute index of the config's select_layer (the community
    standard feature layer, e.g. CLIP -2). Relative fractions lead up to it.
    """
    if official is None:
        official = L - 1
    rel = [int(round(f * official)) for f in MULTI_LAYER_FRACS]
    abs_idx = ([L + o for o in MULTI_LAYER_ABS_OFFSETS]
               + list(MULTI_LAYER_ABS_FIRST)
               + [official - 1, official, official + 1])
    return sorted(set(rel) | {i for i in abs_idx if 0 <= i < L})


def load_yaml_config(yaml_path):
    config = {}
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("vision_encoder", data)
    except ImportError:
        with open(yaml_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line or ":" not in line:
                    continue
                if line.startswith("vision_encoder:"):
                    continue
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if value == "true":
                    value = True
                elif value == "false":
                    value = False
                elif value.lstrip("-").isdigit():
                    value = int(value)
                config[key] = value
        return config


def discover_tokenizer_configs():
    configs = {}
    if not os.path.isdir(VTB_CONFIGS_ROOT):
        return configs
    for yaml_path in sorted(Path(VTB_CONFIGS_ROOT).rglob("*.yaml")):
        try:
            cfg = load_yaml_config(str(yaml_path))
            tok_id = cfg.get("id", yaml_path.stem)
            configs[tok_id] = cfg
        except Exception:
            pass
    return configs


def get_image_size(tok_id, model_name):
    """Infer target image size from tokenizer id or model name."""
    # Special case: siglip2_sm14_384 actually uses 378px model
    if "sm14_384" in tok_id:
        return 378
    match = re.search(r'_(\d{3})$', tok_id)
    if match:
        return int(match.group(1))
    match = re.search(r'-(\d{3})$', model_name)
    if match:
        return int(match.group(1))
    return 224


def get_patch_size(model_name):
    """Get patch size from model name."""
    match = re.search(r'-(\d+)', model_name)
    if match:
        return int(match.group(1))
    return 16


def _find_blocks(model):
    """Locate the transformer block container across model families:
    open_clip VisualTransformer (transformer.resblocks), timm-wrapped
    SigLIP2 (model.blocks / trunk.blocks), HF-style (encoder.layers)."""
    visual = model.visual if hasattr(model, 'visual') else model
    for path in (("transformer", "resblocks"), ("blocks",), ("encoder", "layers"),
                 ("transformer", "encoder", "layers"),
                 ("model", "blocks"), ("trunk", "blocks"),
                 ("model", "encoder", "layers")):
        obj = visual
        ok = True
        for a in path:
            if not hasattr(obj, a):
                ok = False
                break
            obj = getattr(obj, a)
        if ok and isinstance(obj, (nn.ModuleList, nn.Sequential)):
            return obj
    return None


class VisionEncoderWrapper:
    def __init__(self, model, preprocess, feat_dim, select_layer=-2, device="cuda", layers=None):
        self.model = model
        self.preprocess = preprocess
        self.feat_dim = feat_dim
        self.select_layer = select_layer
        self.device = device
        self.layers = layers
        self._features = None
        self._features_dict = {}
        self._hooks = []
        self.total_layers = None
        self.default_layer_idx = None

    def _register_hook(self):
        blocks = _find_blocks(self.model)
        if blocks is None:
            return
        L = len(blocks)
        self.total_layers = L
        if self.layers is not None:
            targets = [i for i in self.layers if 0 <= i < L]
            for idx in targets:
                def hook_fn(module, input, output, _idx=idx):
                    self._features_dict[_idx] = output[0] if isinstance(output, tuple) else output
                self._hooks.append(blocks[idx].register_forward_hook(hook_fn))
            if self.select_layer is not None and self.select_layer != -1:
                self.default_layer_idx = L + self.select_layer if self.select_layer < 0 else self.select_layer
            return
        if self.select_layer is None or self.select_layer == -1:
            return
        target_block = blocks[self.select_layer]
        def hook_fn(module, input, output):
            self._features = output[0] if isinstance(output, tuple) else output
        self._hooks.append(target_block.register_forward_hook(hook_fn))

    @torch.no_grad()
    def encode_images(self, images, normalize=True):
        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        return self.encode_tensors(tensors, normalize)

    @torch.no_grad()
    def encode_tensors(self, tensors, normalize=True):
        tensors = tensors.to(self.device)
        if self.layers is not None:
            self._features_dict = {}
            _ = self.model.encode_image(tensors)
            out = {}
            for idx, feats in self._features_dict.items():
                f = feats[:, 0, :] if feats.dim() == 3 else feats
                if f.dim() > 2:
                    f = f.mean(dim=1)
                if normalize:
                    f = F.normalize(f.float(), dim=-1)
                out[idx] = f
            return out
        if self._hooks:
            self._features = None
            _ = self.model.encode_image(tensors)
            features = self._features
            if features.dim() == 3:
                features = features[:, 0, :]
        else:
            features = self.model.encode_image(tensors)
        if features.dim() > 2:
            features = features.mean(dim=1)
        if normalize:
            features = F.normalize(features.float(), dim=-1)
        return features

    def cleanup(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def load_hf_clip(config, device="cuda"):
    from transformers import CLIPModel, CLIPProcessor
    model_path = config.get("model_name_or_path") or config.get("vision_tower")
    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)
    model.eval()
    select_layer = config.get("select_layer", -2)
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(device)
        outputs = model.vision_model(dummy, output_hidden_states=True)
        feat_dim = outputs.hidden_states[select_layer].shape[-1]
    def preprocess(img):
        inputs = processor(images=img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)
    class W(VisionEncoderWrapper):
        @torch.no_grad()
        def encode_tensors(self, tensors, normalize=True):
            tensors = tensors.to(self.device)
            outputs = self.model.vision_model(tensors, output_hidden_states=True)
            features = outputs.hidden_states[self.select_layer][:, 0, :]
            if normalize:
                features = F.normalize(features.float(), dim=-1)
            return features
    return W(model, preprocess, feat_dim, select_layer, device)


def load_open_clip_local(config, tok_id="", device="cuda"):
    import open_clip
    from torchvision import transforms

    model_name = config.get("model_name", "ViT-B-16")
    weights_path = config.get("weights_path", "")
    force_quick_gelu = config.get("force_quick_gelu", False)
    select_layer = config.get("select_layer", -2)
    image_size = get_image_size(tok_id, model_name)

    model_kwargs = {}
    if force_quick_gelu:
        model_kwargs["force_quick_gelu"] = True

    # Create model (default size)
    model = open_clip.create_model(model_name, pretrained=None, device="cpu", **model_kwargs)

    # Load weights
    if weights_path.endswith(".npz"):
        try:
            model = open_clip.create_model(model_name, pretrained=weights_path, device="cpu", **model_kwargs)
        except Exception:
            data = np.load(weights_path, allow_pickle=True)
            sd = {k: torch.from_numpy(data[k]) for k in data.files}
            model.load_state_dict(sd, strict=False)

    elif weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(weights_path)
        model.load_state_dict(sd, strict=False)

    elif weights_path.endswith(".pt"):
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            sd = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        else:
            sd = checkpoint

        # Only visual keys
        visual_sd = {k: v for k, v in sd.items() if k.startswith("visual.")}
        if not visual_sd:
            visual_sd = sd

        # KEY FIX: Resize model's positional_embedding to match checkpoint
        pos_key = "visual.positional_embedding"
        if pos_key in visual_sd and hasattr(model, 'visual') and hasattr(model.visual, 'positional_embedding'):
            ckpt_pos = visual_sd[pos_key]
            model_pos = model.visual.positional_embedding
            if ckpt_pos.shape[0] != model_pos.shape[0]:
                # Replace model's positional_embedding with correctly sized parameter
                model.visual.positional_embedding = nn.Parameter(
                    torch.zeros(ckpt_pos.shape[0], ckpt_pos.shape[1])
                )

        # Also handle conv1 (patch embed) if image size differs
        # For 384px models, the conv1 might be the same (same patch size)
        # but positional_embedding differs

        model.load_state_dict(visual_sd, strict=False)

    model = model.to(device)
    model.eval()

    # Preprocessing with correct image size
    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )
    preprocess = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    # Detect feature dim
    with torch.no_grad():
        dummy = torch.randn(1, 3, image_size, image_size).to(device)
        feat = model.encode_image(dummy)
        feat_dim = feat.shape[-1]

    wrapper = VisionEncoderWrapper(model, preprocess, feat_dim, select_layer, device, layers=_layers_for(model, select_layer))
    wrapper._register_hook()
    return wrapper


def _layers_for(model, select_layer=-2):
    """Absolute block indices to collect for cross-layer dynamics (or None)."""
    if not MULTI_LAYER_ENABLED:
        return None
    blocks = _find_blocks(model)
    if blocks is not None:
        L = len(blocks)
        official = L + select_layer if select_layer < 0 else select_layer
        return get_multi_layer_indices(L, official)
    return None


def load_ssl_local(config, device="cuda"):
    """Load DINOv3 / RAEv2 / I-JEPA towers by reusing the upstream
    the upstream SSL tower wrappers) so extracted
    features match the pipeline that produced the GT. Patch tokens are
    mean-pooled + L2-normalized to give one vector per image (analogous to
    the CLS vectors of the open_clip family)."""
    import importlib.util
    import types
    vtb_root = os.environ.get("VTB_ROOT",
                          os.path.join(_PKG, "UniTok"))
    os.environ.setdefault("VTB_ROOT", vtb_root)
    if "llava" not in sys.modules:
        llava_mod = types.ModuleType("llava")
        utils_mod = types.ModuleType("llava.utils")
        utils_mod.rank0_print = print
        llava_mod.utils = utils_mod
        sys.modules["llava"] = llava_mod
        sys.modules["llava.utils"] = utils_mod
    ssl_path = os.path.join(vtb_root, "third_party", "LLaVA-NeXT", "llava",
                            "model", "multimodal_encoder", "ssl_encoder.py")
    spec = importlib.util.spec_from_file_location("vtb_ssl_encoder", ssl_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if config.get("layers"):
        os.environ["VTB_SSL_LAYERS"] = ",".join(str(x) for x in config["layers"])
    elif "VTB_SSL_LAYERS" in os.environ:
        del os.environ["VTB_SSL_LAYERS"]
    args = types.SimpleNamespace(
        mm_vision_select_layer=config.get("select_layer", -1),
        mm_vision_select_feature=config.get("select_feature", "patch"),
        mm_vision_image_size=config.get("image_size"),
        vision_tower_pretrained=config.get("weights_path"),
        unfreeze_mm_vision_tower=False,
    )
    tower = mod.SSLVisionTower("vtb_ssl:" + config["type"], args)
    tower = tower.to(device).eval()
    processor = tower.image_processor

    def preprocess(img):
        return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

    class W(VisionEncoderWrapper):
        @torch.no_grad()
        def encode_tensors(self, tensors, normalize=True):
            tensors = tensors.to(device)
            feats = tower(tensors)                     # [B, N, D] patch tokens
            feats = feats.float().mean(dim=1)          # mean-pool -> [B, D]
            if normalize:
                feats = F.normalize(feats, dim=-1)
            return feats

    return W(tower, preprocess, tower.hidden_size,
             config.get("select_layer", -1), device)


def load_tokenizer(tok_id, config, device="cuda"):
    enc_type = config.get("type", "open_clip_hub")
    if enc_type == "hf_clip":
        return load_hf_clip(config, device)
    if enc_type in ("hf", "pe", "pixio", "ijepa", "dinov3", "eupe", "raev2"):
        import tokenizer_loaders
        return tokenizer_loaders.load(config, tok_id, device, multi_layer=MULTI_LAYER_ENABLED)
    return load_open_clip_local(config, tok_id, device)


def get_image_paths(image_dir, num_images, seed=42):
    import random
    image_dir = Path(image_dir)
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_images = sorted([str(p) for p in image_dir.iterdir() if p.suffix.lower() in extensions])
    if not all_images:
        all_images = sorted([str(p) for p in image_dir.rglob("*") if p.suffix.lower() in extensions])
    if not all_images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    random.seed(seed)
    selected = random.sample(all_images, min(num_images, len(all_images)))
    selected.sort()
    return selected


def extract_all(image_paths, output_dir, tokenizer_ids=None, device="cuda", batch_size=64, force=False):
    configs = discover_tokenizer_configs()
    if tokenizer_ids:
        configs = {k: v for k, v in configs.items() if k in tokenizer_ids}

    total = len(configs)
    print(f"\n  Found {total} tokenizer configs, {len(image_paths)} images")
    print(f"  Output: {output_dir}\n")
    print(f"  {'#':<6} {'Tokenizer':<28} {'Status':<10} {'Dim':<6} {'Time':<8}")
    print(f"  {'-'*62}")

    results = []
    ok, skip, fail = 0, 0, 0

    tok_items = sorted(configs.items())
    tok_pbar = tqdm(tok_items, desc="  Tokenizers", unit="tok",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for idx, (tok_id, config) in enumerate(tok_pbar, 1):
        tok_pbar.set_postfix_str(tok_id)
        tok_output_dir = os.path.join(output_dir, tok_id)
        feat_path = os.path.join(tok_output_dir, "visual_features.pt")

        if not force and os.path.exists(feat_path):
            try:
                existing = torch.load(feat_path, map_location="cpu", weights_only=False)
                if existing.shape[0] >= len(image_paths):
                    tok_pbar.write(f"  {idx:<6} {tok_id:<28} {'CACHED':<10} {existing.shape[1]:<6}")
                    results.append({"tokenizer": tok_id, "status": "cached", "dim": existing.shape[1]})
                    skip += 1
                    continue
            except Exception:
                pass

        t0 = time.time()
        try:
            encoder = load_tokenizer(tok_id, config, device)
            os.makedirs(tok_output_dir, exist_ok=True)
            all_features = []
            layer_buffers = {}

            n_batches = (len(image_paths) + batch_size - 1) // batch_size
            for i in tqdm(range(0, len(image_paths), batch_size),
                         desc=f"    {tok_id}", unit="batch", leave=False, total=n_batches):
                batch_imgs = []
                for p in image_paths[i:i+batch_size]:
                    try:
                        batch_imgs.append(Image.open(p).convert("RGB"))
                    except Exception:
                        pass
                if batch_imgs:
                    out = encoder.encode_images(batch_imgs)
                    if isinstance(out, dict):
                        for lidx, f in out.items():
                            layer_buffers.setdefault(lidx, []).append(f.cpu())
                        dl = encoder.default_layer_idx
                        if dl is not None and dl in layer_buffers:
                            all_features.append(layer_buffers[dl][-1])
                    else:
                        all_features.append(out.cpu())

            if not all_features:
                raise RuntimeError("No features")

            all_features = torch.cat(all_features, dim=0)
            torch.save(all_features, feat_path)

            if layer_buffers:
                ml = {lidx: torch.cat(fs, dim=0) for lidx, fs in layer_buffers.items()}
                torch.save(ml, os.path.join(tok_output_dir, "visual_features_layers.pt"))
                with open(os.path.join(tok_output_dir, "layer_indices.json"), "w") as f:
                    json.dump({"tokenizer": tok_id, "total_layers": encoder.total_layers,
                               "layers": sorted(ml.keys()),
                               "relative_fracs": list(MULTI_LAYER_FRACS),
                               "absolute_offsets": list(MULTI_LAYER_ABS_OFFSETS),
                               "default_layer": encoder.default_layer_idx}, f, indent=2)

            with open(os.path.join(tok_output_dir, "meta.json"), "w") as f:
                json.dump({"tokenizer": tok_id, "dim": all_features.shape[1],
                           "n": all_features.shape[0], "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
            with open(os.path.join(tok_output_dir, "image_paths.txt"), "w") as f:
                f.write("\n".join(image_paths))

            elapsed = time.time() - t0
            tok_pbar.write(f"  {idx:<6} {tok_id:<28} {'OK':<10} {all_features.shape[1]:<6} {elapsed:.0f}s")
            results.append({"tokenizer": tok_id, "status": "ok", "dim": all_features.shape[1]})
            ok += 1

            encoder.cleanup()
            del encoder
            torch.cuda.empty_cache()

        except Exception as e:
            elapsed = time.time() - t0
            tok_pbar.write(f"  {idx:<6} {tok_id:<28} {'FAILED':<10} {'-':<6} {elapsed:.0f}s  {str(e)[:60]}")
            results.append({"tokenizer": tok_id, "status": "failed", "error": str(e)})
            fail += 1
            torch.cuda.empty_cache()

    return results, ok, skip, fail


def extract_text_features(output_dir, device="cuda"):
    import open_clip
    print(f"\n  [Text Features] Extracting task prompt embeddings...")

    clip_pt = os.path.join(TOKENIZER_WEIGHTS_ROOT, "ViT-L-14-openai.pt")
    model = None

    if os.path.exists(clip_pt):
        try:
            model = torch.jit.load(clip_pt, map_location=device)
            model.eval()
            print(f"    Loaded CLIP via TorchScript")
        except Exception:
            try:
                model = open_clip.create_model("ViT-L-14", pretrained=None, device=device)
                sd = torch.load(clip_pt, map_location="cpu", weights_only=False)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                model.load_state_dict(sd, strict=False)
                model.eval()
            except Exception as e:
                print(f"    [WARN] Cannot load CLIP: {e}")

    if model is None:
        print(f"    [WARN] Using random text features as fallback")
        text_features = {t: F.normalize(torch.randn(768), dim=0) for t in ["mcq", "vqa", "caption", "binary"]}
        os.makedirs(output_dir, exist_ok=True)
        torch.save(text_features, os.path.join(output_dir, "text_features.pt"))
        return

    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    prompts = {
        "mcq": "Answer the multiple choice question about this image.",
        "vqa": "Answer the question about this image.",
        "caption": "Describe this image in detail.",
        "binary": "Is this statement true about the image?",
    }
    text_features = {}
    for task, prompt in prompts.items():
        tokens = tokenizer([prompt]).to(device)
        with torch.no_grad():
            feat = model.encode_text(tokens) if hasattr(model, 'encode_text') else model(tokens)
            feat = F.normalize(feat.float(), dim=-1)
        text_features[task] = feat.cpu().squeeze(0)

    os.makedirs(output_dir, exist_ok=True)
    torch.save(text_features, os.path.join(output_dir, "text_features.pt"))
    print(f"    Done! 4 task types, dim={list(text_features.values())[0].shape[0]}")
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tokenizers", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--skip_text", action="store_true")
    parser.add_argument("--force", action="store_true", help="Force re-extraction, ignore cached features")
    parser.add_argument("--multi_layer", action="store_true",
                        help="extract features at relative+absolute depth layers "
                             "(cross-layer dynamics D descriptors)")
    args = parser.parse_args()

    global MULTI_LAYER_ENABLED
    MULTI_LAYER_ENABLED = args.multi_layer

    if args.list:
        configs = discover_tokenizer_configs()
        print(f"\n  Available tokenizers ({len(configs)}):")
        for tok_id in sorted(configs.keys()):
            print(f"    {tok_id}")
        return

    script_dir = Path(__file__).parent
    output_dir = args.output_dir or str(script_dir.parent / "features")

    print("=" * 64)
    print("  VTBench Feature Extraction (local weights, no network)")
    print("=" * 64)
    print(f"  Images: {args.image_dir}")
    print(f"  Device: {args.device}")

    if not os.path.isdir(args.image_dir):
        print(f"\n  [ERROR] Image dir not found: {args.image_dir}")
        sys.exit(1)

    image_paths = get_image_paths(args.image_dir, args.num_images, args.seed)

    results, ok, skip, fail = extract_all(
        image_paths, output_dir, args.tokenizers, args.device, args.batch_size,
        force=args.force,
    )

    if not args.skip_text:
        extract_text_features(output_dir, args.device)

    print(f"\n{'='*64}")
    print(f"  DONE: {ok} extracted, {skip} cached, {fail} failed, {len(results)} total")
    if fail > 0:
        print(f"  Failed:")
        for r in results:
            if r["status"] == "failed":
                print(f"    - {r['tokenizer']}")
    print("  Next: build the descriptor with `bash run.sh cka`")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()