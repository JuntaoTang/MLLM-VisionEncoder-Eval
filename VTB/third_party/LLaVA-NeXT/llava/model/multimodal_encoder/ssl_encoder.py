"""VTB continuous SSL vision towers: DINOv3, RAEv2, I-JEPA, PE, EUPE."""

from __future__ import annotations

import glob
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
from transformers import CLIPImageProcessor

from llava.utils import rank0_print

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PE_MEAN = [0.5, 0.5, 0.5]
PE_STD = [0.5, 0.5, 0.5]

VTB_ROOT = os.environ.get("VTB_ROOT", "/home/ma-user/work_space/VTB")
DEFAULT_DINOV3_REPO = os.path.join(VTB_ROOT, "third_party", "dinov3")
DEFAULT_RAEV2_SRC = os.path.join(VTB_ROOT, "third_party", "RAEv2", "src")
DEFAULT_PE_REPO = os.path.join(VTB_ROOT, "third_party", "perception_models")
DEFAULT_EUPE_REPO = os.path.join(VTB_ROOT, "third_party", "eupe")
DEFAULT_PIXIO_REPO = os.path.join(VTB_ROOT, "third_party", "pixio")

PIXIO_HUB_MAP = {
    "pixio_vitb16": "pixio_vitb16",
    "pixio_vitl16": "pixio_vitl16",
    "pixio_vith16": "pixio_vith16",
    "pixio_vit1b16": "pixio_vit1b16",
    "pixio_vit5b16": "pixio_vit5b16",
    "vitb16": "pixio_vitb16",
    "vitl16": "pixio_vitl16",
    "vith16": "pixio_vith16",
    "vit1b16": "pixio_vit1b16",
    "vit5b16": "pixio_vit5b16",
}

# RAEv2 flagship encoder (stage1 general dinov3l-k7).
RAEV2_DEFAULT_LAYERS = [11, 13, 15, 17, 19, 21, 23]

EUPE_HUB_MAP = {
    "EUPE-ViT-T": "eupe_vitt16",
    "EUPE-ViT-S": "eupe_vits16",
    "EUPE-ViT-B": "eupe_vitb16",
    "EUPE-ConvNeXt-T": "eupe_convnext_tiny",
    "EUPE-ConvNeXt-S": "eupe_convnext_small",
    "EUPE-ConvNeXt-B": "eupe_convnext_base",
    "eupe_vitt16": "eupe_vitt16",
    "eupe_vits16": "eupe_vits16",
    "eupe_vitb16": "eupe_vitb16",
    "eupe_convnext_tiny": "eupe_convnext_tiny",
    "eupe_convnext_small": "eupe_convnext_small",
    "eupe_convnext_base": "eupe_convnext_base",
}


def _ensure_path(path: str) -> None:
    if path and os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


def _resolve_weights(explicit: Optional[str] = None) -> Optional[str]:
    for cand in (explicit, os.environ.get("VTB_VISION_WEIGHTS")):
        if not cand:
            continue
        if os.path.isfile(cand):
            return cand
        if os.path.isdir(cand):
            for pat in ("*.pt", "*.pth", "model.safetensors", "pytorch_model.bin"):
                hits = sorted(glob.glob(os.path.join(cand, pat)))
                if hits:
                    return hits[0]
    return None


def _make_image_processor(
    image_size: int,
    *,
    mean: Optional[list[float]] = None,
    std: Optional[list[float]] = None,
) -> CLIPImageProcessor:
    # Match RAEv2/DINOv3 make_dinov3_transform: direct Resize((H, W)), no center-crop.
    # Always set crop_size to image_size: LLaVA finetune builds text-only dummy
    # tensors from processor.crop_size (default 224). If size=256 but crop_size
    # stays 224, mixed batches fail with torch.cat size mismatch.
    hw = {"height": image_size, "width": image_size}
    return CLIPImageProcessor(
        do_resize=True,
        size=hw,
        do_center_crop=False,
        crop_size=hw,
        do_rescale=True,
        rescale_factor=1 / 255.0,
        do_normalize=True,
        image_mean=list(mean or IMAGENET_MEAN),
        image_std=list(std or IMAGENET_STD),
    )


def _parse_layers(spec: Optional[str]) -> list[int]:
    if not spec:
        return list(RAEV2_DEFAULT_LAYERS)
    # Accept "11.13.15..." or "11,13,15..."
    parts = spec.replace(",", ".").split(".")
    return [int(p) for p in parts if p.strip()]


class SSLVisionTower(nn.Module):
    """Frozen SSL encoder tower producing [B, N, D] patch tokens for LLaVA."""

    def __init__(self, vision_tower: str, args, delay_load: bool = False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        # vtb_ssl:dinov3 | vtb_ssl:raev2 | vtb_ssl:ijepa
        self.ssl_type = vision_tower.split(":", 1)[-1].lower().strip()
        self.select_layer = getattr(args, "mm_vision_select_layer", -1)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        default_size = {
            "ijepa": 224,
            "pe": 448,
            "eupe": 256,
            "pixio": 256,
        }.get(self.ssl_type, 256)
        self.image_size = int(
            os.environ.get("VTB_SSL_IMAGE_SIZE")
            or getattr(args, "mm_vision_image_size", None)
            or default_size
        )
        self.layer_indices = _parse_layers(os.environ.get("VTB_SSL_LAYERS"))
        self.weights_path = _resolve_weights(getattr(args, "vision_tower_pretrained", None))
        self.pe_config_name = os.environ.get("VTB_PE_CONFIG") or getattr(args, "pe_config", None)
        self.eupe_hub_name = os.environ.get("VTB_EUPE_HUB") or getattr(args, "eupe_hub", None)
        self.pixio_hub_name = os.environ.get("VTB_PIXIO_HUB") or getattr(args, "pixio_hub", None)
        self._hidden_size: Optional[int] = None
        self._patch_size: Optional[int] = None
        self.vision_tower: Optional[nn.Module] = None
        self.image_processor: Optional[CLIPImageProcessor] = None

        if not delay_load:
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            self.load_model()
        else:
            # Delay-load path: still expose processor + shapes for data pipeline.
            mean, std = (PE_MEAN, PE_STD) if self.ssl_type == "pe" else (IMAGENET_MEAN, IMAGENET_STD)
            self.image_processor = _make_image_processor(self.image_size, mean=mean, std=std)
            self._hidden_size = self._delay_load_hidden_size()
            self._patch_size = 14 if self.ssl_type in ("ijepa", "pe") else 16

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print(f"{self.vision_tower_name} is already loaded, `load_model` called again, skipping.")
            return

        self.weights_path = _resolve_weights(self.weights_path)
        if not self.weights_path:
            raise FileNotFoundError(
                f"SSL vision weights missing for {self.vision_tower_name}. "
                "Set vision_tower_pretrained / VTB_VISION_WEIGHTS to a local .pth file."
            )

        mean, std = (PE_MEAN, PE_STD) if self.ssl_type == "pe" else (IMAGENET_MEAN, IMAGENET_STD)
        self.image_processor = _make_image_processor(self.image_size, mean=mean, std=std)

        if self.ssl_type in ("dinov3", "raev2"):
            self.vision_tower = self._load_dinov3(self.weights_path)
            self._hidden_size = int(self.vision_tower.embed_dim)
            # ConvNeXt: 4× stage downsamples → stride 32; ViT patch=16.
            backbone = self._dinov3_backbone_name(self.weights_path)
            self._patch_size = 32 if "convnext" in backbone else 16
        elif self.ssl_type == "ijepa":
            self.vision_tower = self._load_ijepa(self.weights_path)
            self._hidden_size = int(self.vision_tower.embed_dim)
            self._patch_size = 14
        elif self.ssl_type == "pe":
            self.vision_tower = self._load_pe(self.weights_path)
            self._hidden_size = int(self.vision_tower.width)
            self._patch_size = int(self.vision_tower.patch_size)
            self.image_size = int(getattr(self.vision_tower, "image_size", self.image_size) or self.image_size)
            self.image_processor = _make_image_processor(self.image_size, mean=PE_MEAN, std=PE_STD)
        elif self.ssl_type == "eupe":
            self.vision_tower = self._load_eupe(self.weights_path)
            self._hidden_size = int(self.vision_tower.embed_dim)
            self._patch_size = int(getattr(self.vision_tower, "patch_size", 16) or 16)
        elif self.ssl_type == "pixio":
            self.vision_tower = self._load_pixio(self.weights_path)
            # PixioViT does not expose embed_dim; infer from cls_token.
            embed = getattr(self.vision_tower, "embed_dim", None)
            if embed is None:
                embed = int(self.vision_tower.cls_token.shape[-1])
                self.vision_tower.embed_dim = embed
            self._hidden_size = int(embed)
            self._patch_size = int(self.vision_tower.patch_embed.patch_size[0])
        else:
            raise ValueError(f"Unknown SSL vision tower type: {self.ssl_type}")

        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()
        self.is_loaded = True
        self.keep_eupe_rope_fp32()
        rank0_print(
            f"Loaded SSL tower {self.ssl_type} from {self.weights_path} "
            f"(image_size={self.image_size}, hidden={self._hidden_size}, "
            f"patches={self.num_patches})"
        )

    def keep_eupe_rope_fp32(self) -> None:
        """Keep EUPE ViT weights in fp32. bf16 LayerNorm affines collapse this tiny ViT."""
        vt = self.vision_tower
        if vt is None or not hasattr(vt, "rope_embed"):
            return
        self.vision_tower.float()

    def load_eupe_weights_from_llava_ckpt(self, model_path: str, device=None) -> None:
        """Load the frozen encoder that training actually saved (delay_load skips it)."""
        if self.vision_tower is None or self.ssl_type != "eupe":
            return
        st_path = os.path.join(model_path, "model.safetensors")
        if not os.path.isfile(st_path):
            return
        from safetensors.torch import load_file

        blob = load_file(st_path, device="cpu")
        prefix = "model.vision_tower.vision_tower."
        vt_sd = {k[len(prefix) :]: v.float() for k, v in blob.items() if k.startswith(prefix)}
        if not vt_sd:
            return
        incompatible = self.vision_tower.load_state_dict(vt_sd, strict=False)
        self.vision_tower.float()
        if device is not None:
            self.vision_tower.to(device=device)
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()
        missing = getattr(incompatible, "missing_keys", [])
        if missing:
            rank0_print(f"EUPE ckpt overlay missing {len(missing)} keys (first 4): {missing[:4]}")

    def _delay_load_hidden_size(self) -> int:
        """Placeholder dim before load_model(); must match the real tower."""
        if self.ssl_type == "eupe":
            hub = (self.eupe_hub_name or "").strip() or EUPE_HUB_MAP.get(
                os.path.splitext(os.path.basename(self.weights_path or ""))[0], ""
            )
            return {
                "eupe_vitt16": 192,
                "eupe_vits16": 384,
                "eupe_vitb16": 768,
                "eupe_convnext_tiny": 768,
                "eupe_convnext_small": 768,
                "eupe_convnext_base": 1024,
            }.get(hub, 768)
        return {
            "ijepa": 1280,
            "pe": 1024,
            "pixio": 1280,
        }.get(self.ssl_type, 1024)

    def _dinov3_backbone_name(self, weights: str) -> str:
        """Infer dinov3 hub entry from weights filename / env override."""
        override = (os.environ.get("VTB_DINOV3_BACKBONE") or "").strip()
        if override:
            return override if override.startswith("dinov3_") else f"dinov3_{override}"
        stem = os.path.basename(weights).lower()
        # Order matters: longer / more specific tokens first.
        for key in (
            "vit7b16",
            "vith16plus",
            "vitl16plus",
            "vits16plus",
            "convnext_large",
            "convnext_base",
            "convnext_small",
            "convnext_tiny",
            "vitl16",
            "vitb16",
            "vits16",
        ):
            if key in stem:
                return f"dinov3_{key}"
        return "dinov3_vitl16"

    def _load_dinov3(self, weights: str) -> nn.Module:
        repo = os.environ.get("VTB_DINOV3_REPO_DIR") or os.environ.get("DINOV3_REPO_DIR") or DEFAULT_DINOV3_REPO
        if not os.path.isdir(os.path.join(repo, "dinov3")):
            raise FileNotFoundError(f"DINOv3 package not found under {repo}")
        _ensure_path(repo)
        # Import backbone helpers directly — avoid hubconf.py pulling segmentation deps.
        import dinov3.hub.backbones as bb

        backbone = self._dinov3_backbone_name(weights)
        factory = getattr(bb, backbone, None)
        if factory is None:
            raise ValueError(f"Unknown DINOv3 backbone '{backbone}' for weights={weights}")
        # Prefer local path load: dinov3 hub treats str weights as URL/path.
        model = factory(pretrained=True, weights=weights)
        # RAEv2 DINOv3Encoder strips final LN affine by default; plain DINOv3 keeps it.
        if (
            self.ssl_type == "raev2"
            and hasattr(model, "norm")
            and isinstance(model.norm, nn.LayerNorm)
        ):
            model.norm = nn.LayerNorm(model.embed_dim, elementwise_affine=False)
        return model

    def _load_ijepa(self, weights: str) -> nn.Module:
        _ensure_path(DEFAULT_RAEV2_SRC)
        from encoders.models.jepa import vit_huge

        model = vit_huge(img_size=[self.image_size, self.image_size], patch_size=14)
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "encoder" in ckpt:
            state = ckpt["encoder"]
        elif isinstance(ckpt, dict) and "target_encoder" in ckpt:
            state = ckpt["target_encoder"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        # Drop DistributedDataParallel "module." and optional "backbone." prefixes.
        cleaned = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("backbone."):
                nk = nk[len("backbone.") :]
            cleaned[nk] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            rank0_print(f"I-JEPA missing keys (first 8): {missing[:8]}")
        if unexpected:
            rank0_print(f"I-JEPA unexpected keys (first 8): {unexpected[:8]}")
        return model

    def _load_pe(self, weights: str) -> nn.Module:
        repo = os.environ.get("VTB_PE_REPO_DIR") or DEFAULT_PE_REPO
        if not os.path.isdir(repo):
            raise FileNotFoundError(f"Perception Encoder repo not found: {repo}")
        _ensure_path(repo)
        import core.vision_encoder.pe as pe  # noqa: WPS433

        name = self.pe_config_name
        if not name:
            # Infer from weight filename / parent dir, e.g. PE-Lang-L14-448.pt
            stem = os.path.splitext(os.path.basename(weights))[0]
            parent = os.path.basename(os.path.dirname(weights))
            for cand in (stem, parent):
                if cand in pe.VisionTransformer.available_configs():
                    name = cand
                    break
        if not name:
            raise ValueError(
                "PE config name missing. Set VTB_PE_CONFIG / pe_config "
                "(e.g. PE-Lang-L14-448)."
            )
        model = pe.VisionTransformer.from_config(
            name, pretrained=True, checkpoint_path=weights
        )
        return model

    def _load_eupe(self, weights: str) -> nn.Module:
        repo = os.environ.get("VTB_EUPE_REPO_DIR") or DEFAULT_EUPE_REPO
        if not os.path.isdir(repo):
            raise FileNotFoundError(f"EUPE repo not found: {repo}")
        hub = self.eupe_hub_name
        if not hub:
            stem = os.path.splitext(os.path.basename(weights))[0]
            parent = os.path.basename(os.path.dirname(weights))
            hub = EUPE_HUB_MAP.get(stem) or EUPE_HUB_MAP.get(parent)
        if not hub:
            raise ValueError(
                "EUPE hub entry missing. Set VTB_EUPE_HUB / eupe_hub "
                "(e.g. eupe_vitb16)."
            )
        model = torch.hub.load(
            repo,
            hub,
            source="local",
            pretrained=os.path.isfile(weights),
            weights=weights,
        )
        return model

    def _pixio_hub_name(self, weights: str) -> str:
        override = (self.pixio_hub_name or "").strip()
        if override:
            return PIXIO_HUB_MAP.get(override, override)
        stem = os.path.basename(weights).lower()
        parent = os.path.basename(os.path.dirname(os.path.abspath(weights))).lower()
        blob = f"{parent}/{stem}"
        # Longer tokens first so vit1b16 / vith16 are not swallowed by vitb16.
        for key in ("vit5b16", "vit1b16", "vith16", "vitl16", "vitb16"):
            if key in blob:
                return f"pixio_{key}"
        return "pixio_vith16"

    def _load_pixio(self, weights: str) -> nn.Module:
        repo = os.environ.get("VTB_PIXIO_REPO_DIR") or DEFAULT_PIXIO_REPO
        pixio_pkg = os.path.join(repo, "pixio")
        if not os.path.isdir(pixio_pkg):
            raise FileNotFoundError(f"Pixio package not found under {repo}")
        _ensure_path(pixio_pkg)
        import pixio as pixio_mod  # noqa: WPS433 — local third_party package

        hub = self._pixio_hub_name(weights)
        factory = getattr(pixio_mod, hub, None)
        if factory is None:
            raise ValueError(f"Unknown Pixio hub entry '{hub}' for weights={weights}")
        # Accept native .pth or HF safetensors (converted below).
        weight_path = weights
        if weights.endswith(".safetensors") or (
            os.path.isdir(weights) and os.path.isfile(os.path.join(weights, "model.safetensors"))
        ):
            weight_path = self._pixio_hf_to_native_pth(weights, hub)
        return factory(pretrained=weight_path)

    def _pixio_hf_to_native_pth(self, weights: str, hub: str) -> str:
        """Convert HuggingFace Pixio safetensors → native PixioViT state_dict .pth."""
        if os.path.isdir(weights):
            st_path = os.path.join(weights, "model.safetensors")
            out_path = os.path.join(weights, f"{hub}.pth")
        else:
            st_path = weights
            out_path = os.path.splitext(weights)[0] + ".native.pth"
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 1_000_000:
            return out_path
        from safetensors.torch import load_file

        hf_sd = load_file(st_path)
        native: dict = {
            "cls_token": hf_sd["embeddings.cls_token"],
            "pos_embed": hf_sd["embeddings.position_embeddings"],
            "patch_embed.proj.weight": hf_sd["embeddings.patch_embeddings.projection.weight"],
            "patch_embed.proj.bias": hf_sd["embeddings.patch_embeddings.projection.bias"],
        }
        for cand in ("layernorm", "norm", "encoder.layernorm"):
            wkey = f"{cand}.weight"
            if wkey in hf_sd:
                native["norm.weight"] = hf_sd[wkey]
                native["norm.bias"] = hf_sd[f"{cand}.bias"]
                break
        i = 0
        while f"encoder.layer.{i}.attention.attention.query.weight" in hf_sd:
            p = f"encoder.layer.{i}"
            qw = hf_sd[f"{p}.attention.attention.query.weight"]
            qb = hf_sd[f"{p}.attention.attention.query.bias"]
            kw = hf_sd[f"{p}.attention.attention.key.weight"]
            kb = hf_sd[f"{p}.attention.attention.key.bias"]
            vw = hf_sd[f"{p}.attention.attention.value.weight"]
            vb = hf_sd[f"{p}.attention.attention.value.bias"]
            native[f"blocks.{i}.attn.qkv.weight"] = torch.cat([qw, kw, vw], dim=0)
            native[f"blocks.{i}.attn.qkv.bias"] = torch.cat([qb, kb, vb], dim=0)
            native[f"blocks.{i}.attn.proj.weight"] = hf_sd[f"{p}.attention.output.dense.weight"]
            native[f"blocks.{i}.attn.proj.bias"] = hf_sd[f"{p}.attention.output.dense.bias"]
            native[f"blocks.{i}.norm1.weight"] = hf_sd[f"{p}.norm1.weight"]
            native[f"blocks.{i}.norm1.bias"] = hf_sd[f"{p}.norm1.bias"]
            native[f"blocks.{i}.norm2.weight"] = hf_sd[f"{p}.norm2.weight"]
            native[f"blocks.{i}.norm2.bias"] = hf_sd[f"{p}.norm2.bias"]
            native[f"blocks.{i}.mlp.fc1.weight"] = hf_sd[f"{p}.mlp.fc1.weight"]
            native[f"blocks.{i}.mlp.fc1.bias"] = hf_sd[f"{p}.mlp.fc1.bias"]
            native[f"blocks.{i}.mlp.fc2.weight"] = hf_sd[f"{p}.mlp.fc2.weight"]
            native[f"blocks.{i}.mlp.fc2.bias"] = hf_sd[f"{p}.mlp.fc2.bias"]
            i += 1
        torch.save(native, out_path)
        rank0_print(f"Converted HF Pixio → native {out_path} ({i} layers)")
        return out_path

    @torch.no_grad()
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return patch tokens [B, N, D]."""
        x = images.to(device=self.device, dtype=self.dtype)
        if self.ssl_type == "dinov3":
            out = self.vision_tower.forward_features(x)
            if not isinstance(out, dict) or "x_norm_patchtokens" not in out:
                raise RuntimeError(f"Unexpected DINOv3 forward_features output: {type(out)}")
            return out["x_norm_patchtokens"]

        if self.ssl_type == "raev2":
            outputs = self.vision_tower.get_intermediate_layers(
                x,
                n=self.layer_indices,
                reshape=False,
                return_class_token=False,
                norm=True,
            )
            patch_tokens = torch.stack(outputs, dim=0).mean(dim=0)
            final_mean = outputs[-1].mean(dim=1, keepdim=True)
            return patch_tokens + final_mean

        if self.ssl_type == "pe":
            # PE Lang/Spatial: strip CLS when present; return dense tokens.
            return self.vision_tower.forward_features(
                x, norm=True, strip_cls_token=True
            )

        if self.ssl_type == "eupe":
            # ViT RoPE must stay fp32 (hub pos_embed_rope_dtype=fp32). ConvNeXt has no rope.
            if hasattr(self.vision_tower, "rope_embed"):
                x = images.to(device=self.device, dtype=torch.float32)
            out = self.vision_tower.forward_features(x)
            if isinstance(out, list):
                out = out[0]
            if not isinstance(out, dict) or "x_norm_patchtokens" not in out:
                raise RuntimeError(f"Unexpected EUPE forward_features output: {type(out)}")
            return out["x_norm_patchtokens"].to(dtype=images.dtype)

        if self.ssl_type == "pixio":
            # Last block normalized patch tokens (matches dense-prediction usage).
            feats = self.vision_tower(x, block_ids=[len(self.vision_tower.blocks) - 1])
            return feats[-1]["patch_tokens_norm"]

        # I-JEPA: ViT returns patch tokens only (no CLS in RAEv2 jepa impl).
        return self.vision_tower(x)

    def forward(self, images):
        if type(images) is list:
            feats = []
            for image in images:
                feat = self._encode(image.unsqueeze(0)).to(image.dtype)
                feats.append(feat)
            return feats
        return self._encode(images).to(images.dtype)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        if self.vision_tower is None:
            return torch.float32
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        if self.vision_tower is None:
            return torch.device("cpu")
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        return None

    @property
    def hidden_size(self):
        return int(self._hidden_size or 1024)

    @property
    def num_patches(self):
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.image_size // self.patch_size

    @property
    def patch_size(self):
        return int(self._patch_size or (14 if self.ssl_type == "ijepa" else 16))

    # LLaVA reads `.image_size` from the vision tower.
    # Stored as plain attribute in __init__; keep property alias for clarity.
