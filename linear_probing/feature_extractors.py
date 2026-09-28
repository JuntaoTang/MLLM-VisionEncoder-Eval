"""Frozen feature extractors for the tokenizer linear-probing baseline.

Each extractor returns exactly one preselected visual representation shaped
``[batch, feature_dim]``.  Pooling performed here is part of that representation;
the evaluator does not perform DINOv2's multi-block or CLS/patch readout search.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Callable

import torch
from torch import nn

from dinov2.data.transforms import make_classification_eval_transform, make_classification_train_transform


PM1_MEAN = (0.5, 0.5, 0.5)
PM1_STD = (0.5, 0.5, 0.5)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IDENTITY_MEAN = (0.0, 0.0, 0.0)
IDENTITY_STD = (1.0, 1.0, 1.0)

MC1_SPECS = {
    "mc1_b32_224_400m": (
        "vit_base_patch32_clip_224.metaclip_400m",
        "mc1_b32_224_400m_checkpoint",
        "final normalized CLS before the MetaCLIP 768-to-512 projection",
    ),
    "mc1_b16_224_400m": (
        "vit_base_patch16_clip_224.metaclip_400m",
        "mc1_b16_224_400m_checkpoint",
        "final normalized CLS before the MetaCLIP 768-to-512 projection",
    ),
    "mc1_l14_224_400m": (
        "vit_large_patch14_clip_224.metaclip_400m",
        "mc1_l14_224_400m_checkpoint",
        "final normalized CLS before the MetaCLIP 1024-to-768 projection",
    ),
    "mc1_b32_224_2.5b": (
        "vit_base_patch32_clip_224.metaclip_2pt5b",
        "mc1_b32_224_2_5b_checkpoint",
        "final normalized CLS before the MetaCLIP 768-to-512 projection",
    ),
    "mc1_b16_224_2.5b": (
        "vit_base_patch16_clip_224.metaclip_2pt5b",
        "mc1_b16_224_2_5b_checkpoint",
        "final normalized CLS before the MetaCLIP 768-to-512 projection",
    ),
    "mc1_l14_224_2.5b": (
        "vit_large_patch14_clip_224.metaclip_2pt5b",
        "mc1_l14_224_2_5b_checkpoint",
        "final normalized CLS before the MetaCLIP 1024-to-768 projection",
    ),
    "mc1_h14_224_2.5b": (
        "vit_huge_patch14_clip_224.metaclip_2pt5b",
        "mc1_h14_224_2_5b_checkpoint",
        "final normalized CLS before the MetaCLIP 1280-to-1024 projection",
    ),
    "mc1_g14_224_2.5b": (
        "vit_gigantic_patch14_clip_224.metaclip_2pt5b",
        "mc1_g14_224_2_5b_checkpoint",
        "final normalized CLS before the MetaCLIP 1664-to-1280 projection",
    ),
    "mc1_h14_224_v1.2": (
        "vit_huge_patch14_clip_224.metaclip_altogether",
        "mc1_h14_224_v1_2_checkpoint",
        "final normalized CLS before the MetaCLIP 1280-to-1024 projection",
    ),
}

MC2_TIMM_SPECS = {
    "mc2_h14_378": (
        "vit_huge_patch14_clip_378.metaclip2_worldwide",
        "mc2_h14_378_checkpoint",
        "final normalized CLS before the MetaCLIP 2 1280-to-1024 projection",
    ),
    "mc2_g14_224": (
        "vit_gigantic_patch14_clip_224.metaclip2_worldwide",
        "mc2_g14_224_checkpoint",
        "final normalized CLS before the MetaCLIP 2 1664-to-1280 projection",
    ),
    "mc2_g14_378": (
        "vit_gigantic_patch14_clip_378.metaclip2_worldwide",
        "mc2_g14_378_checkpoint",
        "final normalized CLS before the MetaCLIP 2 1664-to-1280 projection",
    ),
}

# checkpoint arg, image size, patch size, width, depth, projection output dim
MC2_DISTILLED_SPECS = {
    "mc2_s16_224": ("mc2_s16_224_checkpoint", 224, 16, 384, 12, 384),
    "mc2_s16_384": ("mc2_s16_384_checkpoint", 384, 16, 384, 12, 384),
    "mc2_s16_224_mt5": ("mc2_s16_224_mt5_checkpoint", 224, 16, 384, 12, 384),
    "mc2_m16_224": ("mc2_m16_224_checkpoint", 224, 16, 512, 12, 512),
    "mc2_m16_384": ("mc2_m16_384_checkpoint", 384, 16, 512, 12, 512),
    "mc2_m16_224_mt5": ("mc2_m16_224_mt5_checkpoint", 224, 16, 512, 12, 512),
    "mc2_b32_224": ("mc2_b32_224_checkpoint", 224, 32, 768, 12, 512),
    "mc2_b32_384": ("mc2_b32_384_checkpoint", 384, 32, 768, 12, 512),
    "mc2_b32_224_mt5": ("mc2_b32_224_mt5_checkpoint", 224, 32, 768, 12, 512),
    "mc2_b16_224": ("mc2_b16_224_checkpoint", 224, 16, 768, 12, 512),
    "mc2_b16_384": ("mc2_b16_384_checkpoint", 384, 16, 768, 12, 512),
    "mc2_l14_224": ("mc2_l14_224_checkpoint", 224, 14, 1024, 24, 768),
}

SIGLIP2_SPECS = {
    "siglip2_b32_256": (
        "vit_base_patch32_siglip_256",
        "siglip2_b32_256_model_path",
        256,
        768,
    ),
    "siglip2_b16_224": (
        "vit_base_patch16_siglip_224",
        "siglip2_b16_224_model_path",
        224,
        768,
    ),
    "siglip2_b16_256": (
        "vit_base_patch16_siglip_256",
        "siglip2_b16_256_model_path",
        256,
        768,
    ),
    "siglip2_b16_384": (
        "vit_base_patch16_siglip_384",
        "siglip2_b16_384_model_path",
        384,
        768,
    ),
    "siglip2_b16_512": (
        "vit_base_patch16_siglip_512",
        "siglip2_b16_512_model_path",
        512,
        768,
    ),
    "siglip2_l16_256": (
        "vit_large_patch16_siglip_256",
        "siglip2_l16_256_model_path",
        256,
        1024,
    ),
    "siglip2_l16_384": (
        "vit_large_patch16_siglip_384",
        "siglip2_l16_384_model_path",
        384,
        1024,
    ),
    "siglip2_l16_512": (
        "vit_large_patch16_siglip_512",
        "siglip2_l16_512_model_path",
        512,
        1024,
    ),
    "siglip2_sm14_224": (
        "vit_so400m_patch14_siglip_224",
        "siglip2_sm14_224_model_path",
        224,
        1152,
    ),
    # The official checkpoint is named "..._384", but its native grid is
    # 27x27 with patch size 14, so the converted timm model uses 378x378.
    "siglip2_sm14_384": (
        "vit_so400m_patch14_siglip_378",
        "siglip2_sm14_384_model_path",
        378,
        1152,
    ),
    "siglip2_sm16_256": (
        "vit_so400m_patch16_siglip_256",
        "siglip2_sm16_256_model_path",
        256,
        1152,
    ),
    "siglip2_sm16_384": (
        "vit_so400m_patch16_siglip_384",
        "siglip2_sm16_384_model_path",
        384,
        1152,
    ),
    "siglip2_sm16_512": (
        "vit_so400m_patch16_siglip_512",
        "siglip2_sm16_512_model_path",
        512,
        1152,
    ),
    "siglip2_g16_256": (
        "vit_giantopt_patch16_siglip_256",
        "siglip2_g16_256_model_path",
        256,
        1536,
    ),
    "siglip2_g16_384": (
        "vit_giantopt_patch16_siglip_384",
        "siglip2_g16_384_model_path",
        384,
        1536,
    ),
}

RAEV2_SPECS = {
    "dinov3": {
        "stats": "stage1/imagenet/dinov3l-k1/stats.pt",
        "representation": (
            "spatial mean of the normalized RAEv2 DINOv3-L/16 K=1 tokenizer latent"
        ),
    },
    "raev2": {
        "stats": "stage1/imagenet/dinov3l-k23/stats.pt",
        "representation": (
            "spatial mean of the normalized RAEv2 DINOv3-L/16 K=23 "
            "multi-layer tokenizer latent"
        ),
    },
    "raev2_dinov3l_k7": {
        "stats": "stage1/imagenet/dinov3l-k7/stats.pt",
        "representation": (
            "spatial mean of the normalized RAEv2 DINOv3-L/16 K=7 "
            "multi-layer tokenizer latent"
        ),
    },
    "ijepa": {
        "stats": "stage1/imagenet/jepa-h-k1/stats.pt",
        "representation": (
            "spatial mean of the normalized RAEv2 I-JEPA-H/14 K=1 tokenizer latent"
        ),
    },
}

# directory, timm architecture, input size, readout kind, backbone width,
# projection width (PE-Core only). The local checkpoints use the official
# Perception Encoder key layout; timm's EVA checkpoint filter converts it.
PE_SPECS = {
    "pe_lang_g14_448": (
        "PE-Lang-G14-448", "vit_pe_lang_gigantic_patch14_448", 448, "patch_mean", 1536, None,
    ),
    "pe_lang_l14_448": (
        "PE-Lang-L14-448", "vit_pe_lang_large_patch14_448", 448, "patch_mean", 1024, None,
    ),
    "pe_lang_g14_448_tiling": (
        "PE-Lang-G14-448-Tiling", "vit_pe_lang_gigantic_patch14_448", 448, "patch_mean", 1536, None,
    ),
    "pe_lang_l14_448_tiling": (
        "PE-Lang-L14-448-Tiling", "vit_pe_lang_large_patch14_448", 448, "patch_mean", 1024, None,
    ),
    "pe_core_g14_448": (
        "PE-Core-G14-448", "vit_pe_core_gigantic_patch14_448", 448, "attention_pool", 1536, 1280,
    ),
    "pe_core_l14_336": (
        "PE-Core-L14-336", "vit_pe_core_large_patch14_336", 336, "attention_pool", 1024, 1024,
    ),
    "pe_core_b16_224": (
        "PE-Core-B16-224", "vit_pe_core_base_patch16_224", 224, "attention_pool", 768, 1024,
    ),
    "pe_core_s16_384": (
        "PE-Core-S16-384", "vit_pe_core_small_patch16_384", 384, "attention_pool", 384, 512,
    ),
    "pe_core_t16_384": (
        "PE-Core-T16-384", "vit_pe_core_tiny_patch16_384", 384, "attention_pool", 192, 512,
    ),
    "pe_spatial_g14_448": (
        "PE-Spatial-G14-448", "vit_pe_spatial_gigantic_patch14_448", 448, "patch_mean", 1536, None,
    ),
    "pe_spatial_l14_448": (
        "PE-Spatial-L14-448", "vit_pe_spatial_large_patch14_448", 448, "patch_mean", 1024, None,
    ),
    "pe_spatial_b16_512": (
        "PE-Spatial-B16-512", "vit_pe_spatial_base_patch16_512", 512, "patch_mean", 768, None,
    ),
    "pe_spatial_s16_512": (
        "PE-Spatial-S16-512", "vit_pe_spatial_small_patch16_512", 512, "patch_mean", 384, None,
    ),
    "pe_spatial_t16_512": (
        "PE-Spatial-T16-512", "vit_pe_spatial_tiny_patch16_512", 512, "patch_mean", 192, None,
    ),
}

# directory, expected transformers model_type. All ViT entries use the same
# fixed concat(CLS, mean(patch)) readout. Register tokens are excluded.
DINO_VIT_SPECS = {
    "dinov3_vitl16_lvd1689m": ("dinov3-vitl16-pretrain-lvd1689m", "dinov3_vit"),
    "dinov3_vith16plus_lvd1689m": ("dinov3-vith16plus-pretrain-lvd1689m", "dinov3_vit"),
    "dinov3_vitb16_lvd1689m": ("dinov3-vitb16-pretrain-lvd1689m", "dinov3_vit"),
    "dinov3_vits16_lvd1689m": ("dinov3-vits16-pretrain-lvd1689m", "dinov3_vit"),
    "dinov3_vits16plus_lvd1689m": ("dinov3-vits16plus-pretrain-lvd1689m", "dinov3_vit"),
    "dinov3_vit7b16_lvd1689m": ("dinov3-vit7b16-pretrain-lvd1689m", "dinov3_vit"),
    "dinov2_giant": ("dinov2-giant", "dinov2"),
    "dinov2_large": ("dinov2-large", "dinov2"),
    "dinov2_base": ("dinov2-base", "dinov2"),
    "dinov2_small": ("dinov2-small", "dinov2"),
    "dinov1_vitb16": ("dino-vitb16", "vit"),
    "dinov1_vits16": ("dino-vits16", "vit"),
    "dinov1_vitb8": ("dino-vitb8", "vit"),
    "dinov1_vits8": ("dino-vits8", "vit"),
}

DINOV3_CONVNEXT_SPECS = {
    "dinov3_convnext_large_lvd1689m": (
        "dinov3-convnext-large-pretrain-lvd1689m", "dinov3_convnext",
    ),
    "dinov3_convnext_base_lvd1689m": (
        "dinov3-convnext-base-pretrain-lvd1689m", "dinov3_convnext",
    ),
    "dinov3_convnext_small_lvd1689m": (
        "dinov3-convnext-small-pretrain-lvd1689m", "dinov3_convnext",
    ),
    "dinov3_convnext_tiny_lvd1689m": (
        "dinov3-convnext-tiny-pretrain-lvd1689m", "dinov3_convnext",
    ),
}

WEBSSL_DINO_SPECS = {
    "webssl_dino300m_full2b_224": ("webssl-dino300m-full2b-224", "dinov2"),
    "webssl_dino1b_full2b_224": ("webssl-dino1b-full2b-224", "dinov2"),
    "webssl_dino2b_full2b_224": ("webssl-dino2b-full2b-224", "dinov2"),
    "webssl_dino3b_full2b_224": ("webssl-dino3b-full2b-224", "dinov2"),
    "webssl_dino5b_full2b_224": ("webssl-dino5b-full2b-224", "dinov2"),
    "webssl_dino7b_full8b_224": ("webssl-dino7b-full8b-224", "dinov2"),
    "webssl_dino7b_full8b_378": ("webssl-dino7b-full8b-378", "dinov2"),
    "webssl_dino7b_full8b_518": ("webssl-dino7b-full8b-518", "dinov2"),
    "webssl_dino2b_light2b_224": ("webssl-dino2b-light2b-224", "dinov2"),
    "webssl_dino2b_heavy2b_224": ("webssl-dino2b-heavy2b-224", "dinov2"),
    "webssl_dino3b_light2b_224": ("webssl-dino3b-light2b-224", "dinov2"),
    "webssl_dino3b_heavy2b_224": ("webssl-dino3b-heavy2b-224", "dinov2"),
    "webssl_dino300m_light2b_224": ("webssl-dino300m-light2b-224", "dinov2"),
}

WEBSSL_MAE_SPECS = {
    "webssl_mae300m_full2b_224": ("webssl-mae300m-full2b-224", "vit"),
    "webssl_mae700m_full2b_224": ("webssl-mae700m-full2b-224", "vit"),
    "webssl_mae1b_full2b_224": ("webssl-mae1b-full2b-224", "vit"),
    "webssl_mae2b_full2b_224": ("webssl-mae2b-full2b-224", "vit"),
    "webssl_mae3b_full2b_224": ("webssl-mae3b-full2b-224", "vit"),
}

# directory, hidden width, encoder depth, attention heads. Pixio uses eight
# learned CLS tokens for every released size; the source-native global readout
# averages the final block's pre-LayerNorm CLS tokens.
PIXIO_SPECS = {
    "pixio_vitb16": ("pixio-vitb16", 768, 12, 12),
    "pixio_vitl16": ("pixio-vitl16", 1024, 24, 16),
    "pixio_vith16": ("pixio-vith16", 1280, 32, 16),
    "pixio_vit1b16": ("pixio-vit1b16", 1536, 48, 24),
    "pixio_vit5b16": ("pixio-vit5b16", 3072, 48, 32),
}

# directory, architecture family, architecture size, released feature width.
# EUPE checkpoints contain the backbone plus training-only distillation
# projectors; the loader below admits only the latter as non-backbone keys and
# strictly loads every backbone tensor.
EUPE_SPECS = {
    "eupe_vit_t": ("EUPE-ViT-T", "vit", "tiny", 192),
    "eupe_vit_s": ("EUPE-ViT-S", "vit", "small", 384),
    "eupe_vit_b": ("EUPE-ViT-B", "vit", "base", 768),
    "eupe_convnext_t": ("EUPE-ConvNeXt-T", "convnext", "tiny", 768),
    "eupe_convnext_s": ("EUPE-ConvNeXt-S", "convnext", "small", 768),
    "eupe_convnext_b": ("EUPE-ConvNeXt-B", "convnext", "base", 1024),
}

DINOV3_L_CHECKPOINT = (
    "encoders/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)
IJEPA_H_CHECKPOINT = "encoders/ijepa/ijepa_vith.pth"
RAEV2_K23_LAYERS = tuple(range(1, 24))
RAEV2_K7_LAYERS = (11, 13, 15, 17, 19, 21, 23)


@dataclass
class FeatureBundle:
    encoder: nn.Module
    train_transform: Callable
    eval_transform: Callable
    autocast_context: Callable
    representation: str
    transform_description: str
    backbone_precision: str
    checkpoint_paths: list[str]


class MetaCLIPEncoder(nn.Module):
    """Final normalized MetaCLIP CLS, before the vision projection head."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model.get_intermediate_layers(
            images,
            n=1,
            return_prefix_tokens=True,
            norm=True,
        )
        _patch_tokens, prefix_tokens = outputs[-1]
        if prefix_tokens.ndim != 3 or prefix_tokens.shape[1] < 1:
            raise RuntimeError(f"Unexpected MetaCLIP prefix-token shape: {tuple(prefix_tokens.shape)}")
        return prefix_tokens[:, 0].float()


class MetaCLIP2DistilledEncoder(nn.Module):
    """Final normalized CLS from an OpenAI-style MetaCLIP 2 visual tower."""

    def __init__(self, visual: nn.Module):
        super().__init__()
        if visual.proj is not None:
            raise ValueError("MetaCLIP 2 visual projection must be disabled for linear probing")
        self.visual = visual

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.visual(images)
        if features.ndim != 2:
            raise RuntimeError(f"Unexpected MetaCLIP 2 CLS shape: {tuple(features.shape)}")
        return features.float()


class OpenAIClipEncoder(nn.Module):
    """Final post-LayerNorm OpenAI CLIP CLS, before visual_projection."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=images, return_dict=True)
        if outputs.pooler_output is None or outputs.pooler_output.ndim != 2:
            shape = None if outputs.pooler_output is None else tuple(outputs.pooler_output.shape)
            raise RuntimeError(f"Unexpected OpenAI CLIP pooled-output shape: {shape}")
        return outputs.pooler_output.float()


class SigLIP2MAPEncoder(nn.Module):
    """Native SigLIP 2 MAP output returned directly by ``model(images)``."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images)
        if features.ndim != 2:
            raise RuntimeError(f"Unexpected SigLIP 2 MAP shape: {tuple(features.shape)}")
        return features.float()


class PerceptionEncoderReadout(nn.Module):
    """Fixed PE readout: Core attention pool or Lang/Spatial patch mean."""

    def __init__(self, model: nn.Module, readout_kind: str):
        super().__init__()
        if readout_kind not in {"attention_pool", "patch_mean"}:
            raise ValueError(f"Unsupported Perception Encoder readout: {readout_kind}")
        self.model = model
        self.readout_kind = readout_kind

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(images)
        if tokens.ndim != 3:
            raise RuntimeError(f"Unexpected Perception Encoder token shape: {tuple(tokens.shape)}")
        if self.readout_kind == "attention_pool":
            features = self.model.forward_head(tokens, pre_logits=True)
        else:
            prefix_count = int(self.model.num_prefix_tokens)
            patch_tokens = tokens[:, prefix_count:]
            if patch_tokens.shape[1] == 0:
                raise RuntimeError("Perception Encoder returned no patch tokens")
            features = patch_tokens.mean(dim=1)
        if features.ndim != 2:
            raise RuntimeError(
                f"Unexpected Perception Encoder readout shape: {tuple(features.shape)}"
            )
        return features.float()


class HFClsPatchEncoder(nn.Module):
    """Concatenate final normalized CLS with the mean patch token."""

    def __init__(self, model: nn.Module, register_token_count: int):
        super().__init__()
        self.model = model
        self.prefix_token_count = 1 + int(register_token_count)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=images, return_dict=True)
        tokens = outputs.last_hidden_state
        if tokens.ndim != 3 or tokens.shape[1] <= self.prefix_token_count:
            raise RuntimeError(
                "Unexpected DINO token shape: "
                f"tokens={tuple(tokens.shape)}, prefix_count={self.prefix_token_count}"
            )
        cls_token = tokens[:, 0]
        patch_mean = tokens[:, self.prefix_token_count:].mean(dim=1)
        return torch.cat((cls_token, patch_mean), dim=-1).float()


class HFConvNeXtGlobalEncoder(nn.Module):
    """DINOv3 ConvNeXt final-stage GAP followed by the released final norm."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=images, return_dict=True)
        features = outputs.pooler_output
        if features is None or features.ndim != 2:
            shape = None if features is None else tuple(features.shape)
            raise RuntimeError(f"Unexpected DINOv3 ConvNeXt pooled shape: {shape}")
        return features.float()


class HFClsEncoder(nn.Module):
    """Final normalized CLS token from a Hugging Face ViT encoder."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=images, return_dict=True)
        tokens = outputs.last_hidden_state
        if tokens.ndim != 3 or tokens.shape[1] < 1:
            raise RuntimeError(f"Unexpected Web-MAE token shape: {tuple(tokens.shape)}")
        return tokens[:, 0].float()


class PixioPatchEmbeddings(nn.Module):
    """Patch projection with names matching the released HF safetensors."""

    def __init__(self, hidden_size: int, patch_size: int):
        super().__init__()
        self.patch_size = int(patch_size)
        self.projection = nn.Conv2d(
            3,
            hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images).flatten(2).transpose(1, 2)


class PixioEmbeddings(nn.Module):
    """Eight CLS tokens followed by the spatial patch-token sequence."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_size: int,
        n_cls_tokens: int,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.n_cls_tokens = int(n_cls_tokens)
        self.cls_token = nn.Parameter(torch.empty(1, n_cls_tokens, hidden_size))
        self.patch_embeddings = PixioPatchEmbeddings(hidden_size, patch_size)
        patch_count = (image_size // patch_size) ** 2
        self.position_embeddings = nn.Parameter(
            torch.empty(1, n_cls_tokens + patch_count, hidden_size)
        )

    def _position_encoding(
        self,
        token_count: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        native_patch_count = self.position_embeddings.shape[1] - self.n_cls_tokens
        if (
            token_count == native_patch_count
            and height == self.image_size
            and width == self.image_size
        ):
            return self.position_embeddings
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Pixio input {(height, width)} must be divisible by patch size "
                f"{self.patch_size}"
            )
        native_grid = math.isqrt(native_patch_count)
        if native_grid * native_grid != native_patch_count:
            raise RuntimeError(
                f"Pixio positional embedding has a non-square patch grid: "
                f"{native_patch_count}"
            )
        cls_position = self.position_embeddings[:, : self.n_cls_tokens]
        patch_position = self.position_embeddings[:, self.n_cls_tokens :]
        patch_position = patch_position.reshape(
            1, native_grid, native_grid, -1
        ).permute(0, 3, 1, 2)
        target_dtype = patch_position.dtype
        patch_position = nn.functional.interpolate(
            patch_position.float(),
            size=(height // self.patch_size, width // self.patch_size),
            mode="bicubic",
            align_corners=False,
        ).to(target_dtype)
        patch_position = patch_position.permute(0, 2, 3, 1).reshape(
            1, token_count, -1
        )
        return torch.cat((cls_position, patch_position), dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, _channels, height, width = images.shape
        patch_tokens = self.patch_embeddings(images)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, patch_tokens), dim=1)
        return tokens + self._position_encoding(
            patch_tokens.shape[1], height, width
        )


class PixioSelfAttention(nn.Module):
    """Pre-norm multi-head attention with checkpoint-compatible names."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(
                f"Pixio hidden size {hidden_size} is not divisible by {num_heads} heads"
            )
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def _split_heads(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, hidden_size = tokens.shape
        return tokens.reshape(
            batch_size, token_count, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.query(tokens))
        key = self._split_heads(self.key(tokens))
        value = self._split_heads(self.value(tokens))
        attended = nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        return attended.transpose(1, 2).reshape(tokens.shape)


class PixioAttentionOutput(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.dense(tokens)


class PixioAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.attention = PixioSelfAttention(hidden_size, num_heads)
        self.output = PixioAttentionOutput(hidden_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output(self.attention(tokens))


class PixioMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * mlp_ratio)
        self.fc2 = nn.Linear(hidden_size * mlp_ratio, hidden_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.fc2(nn.functional.gelu(self.fc1(tokens)))


class PixioLayer(nn.Module):
    """One official Pixio pre-normalization Transformer block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.attention = PixioAttention(hidden_size, num_heads)
        self.mlp = PixioMLP(hidden_size, mlp_ratio)
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.norm1(tokens))
        return tokens + self.mlp(self.norm2(tokens))


class PixioLayers(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: int,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                PixioLayer(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    layer_norm_eps,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.layer:
            tokens = block(tokens)
        return tokens


class PixioEncoder(nn.Module):
    """Released Pixio encoder with a selectable CLS-token readout."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: int,
        n_cls_tokens: int,
        layer_norm_eps: float,
        readout: str = "post-ln",
    ):
        super().__init__()
        if readout not in {"pre-ln", "post-ln"}:
            raise ValueError(f"Unsupported Pixio readout: {readout}")
        self.n_cls_tokens = int(n_cls_tokens)
        self.readout = readout
        self.embeddings = PixioEmbeddings(
            image_size,
            patch_size,
            hidden_size,
            n_cls_tokens,
        )
        self.encoder = PixioLayers(
            hidden_size,
            depth,
            num_heads,
            mlp_ratio,
            layer_norm_eps,
        )
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(self.embeddings(images))
        if self.readout == "post-ln":
            tokens = self.layernorm(tokens)
        return tokens[:, : self.n_cls_tokens].mean(dim=1).float()


class EUPEViTEncoder(nn.Module):
    """Concatenate EUPE's normalized CLS with its mean normalized patch token."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model.forward_features(images)
        cls_token = outputs.get("x_norm_clstoken")
        patch_tokens = outputs.get("x_norm_patchtokens")
        if (
            not torch.is_tensor(cls_token)
            or not torch.is_tensor(patch_tokens)
            or cls_token.ndim != 2
            or patch_tokens.ndim != 3
            or patch_tokens.shape[1] == 0
        ):
            cls_shape = None if not torch.is_tensor(cls_token) else tuple(cls_token.shape)
            patch_shape = (
                None if not torch.is_tensor(patch_tokens) else tuple(patch_tokens.shape)
            )
            raise RuntimeError(
                f"Unexpected EUPE ViT features: CLS={cls_shape}, patches={patch_shape}"
            )
        return torch.cat((cls_token, patch_tokens.mean(dim=1)), dim=-1).float()


class EUPEConvNeXtEncoder(nn.Module):
    """EUPE ConvNeXt final-stage GAP followed by the released final norm."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(images).get("x_norm_clstoken")
        if not torch.is_tensor(features) or features.ndim != 2:
            shape = None if not torch.is_tensor(features) else tuple(features.shape)
            raise RuntimeError(f"Unexpected EUPE ConvNeXt pooled shape: {shape}")
        return features.float()


class RAEv2LatentEncoder(nn.Module):
    """Pool the exact normalized spatial latent consumed by an RAEv2 decoder."""

    def __init__(
        self,
        model: nn.Module,
        variant: str,
        latent_mean: torch.Tensor,
        latent_var: torch.Tensor,
        eps: float = 1e-5,
    ):
        super().__init__()
        if variant not in RAEV2_SPECS:
            raise ValueError(f"Unsupported RAEv2 tokenizer variant: {variant}")
        self.model = model
        self.variant = variant
        self.eps = float(eps)
        self.register_buffer(
            "pixel_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer("latent_mean", latent_mean.unsqueeze(0).float())
        self.register_buffer("latent_var", latent_var.unsqueeze(0).float())

    def _encode_tokens(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.pixel_mean) / self.pixel_std
        if self.variant == "ijepa":
            # Match RAEv2's JEPAEncoder: native RAE input is 256, while the
            # I-JEPA backbone itself receives 224x224.
            images = nn.functional.interpolate(images, 224, mode="bicubic")
            return self.model(images)

        # DINOv3's RAEv2 transform is a square resize to the native 256 input.
        if images.shape[-2:] != (256, 256):
            images = nn.functional.interpolate(
                images,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        if self.variant == "dinov3":
            return self.model.forward_features(images)["x_norm_patchtokens"]

        layers = (
            RAEV2_K7_LAYERS
            if self.variant == "raev2_dinov3l_k7"
            else RAEV2_K23_LAYERS
        )
        outputs = self.model.get_intermediate_layers(
            images,
            n=layers,
            reshape=False,
            return_class_token=False,
            norm=True,
        )
        patch_tokens = torch.stack(outputs, dim=0).mean(dim=0)
        return patch_tokens + outputs[-1].mean(dim=1, keepdim=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self._encode_tokens(images)
        if tokens.ndim != 3:
            raise RuntimeError(
                f"Unexpected {self.variant} tokenizer-token shape: {tuple(tokens.shape)}"
            )
        batch_size, token_count, feature_dim = tokens.shape
        grid_size = math.isqrt(token_count)
        if grid_size * grid_size != token_count:
            raise RuntimeError(
                f"{self.variant} returned a non-square token grid: {token_count}"
            )
        latent = tokens.transpose(1, 2).reshape(
            batch_size,
            feature_dim,
            grid_size,
            grid_size,
        )
        expected_shape = tuple(self.latent_mean.shape[1:])
        if tuple(latent.shape[1:]) != expected_shape:
            raise RuntimeError(
                f"{self.variant} latent shape mismatch: expected {expected_shape}, "
                f"got {tuple(latent.shape[1:])}"
            )
        latent = (latent - self.latent_mean) / torch.sqrt(self.latent_var + self.eps)
        return latent.mean(dim=(2, 3)).float()


class TokLIPEncoder(nn.Module):
    """Mean of TokLIP's final normalized semantic tokens."""

    def __init__(self, trunk: nn.Module, encode_tokens: Callable):
        super().__init__()
        self.trunk = trunk
        self.encode_tokens = encode_tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.encode_tokens(self.trunk, images)
        return tokens.mean(dim=1).float()


class UniTokEncoder(nn.Module):
    """Mean-pooled quantized UniTok tokens after fc_norm, before projection."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.model.encoder(images).float()
        tokens = self.model.quant_proj(tokens)
        indices = self.model.quantizer.f_to_idx(tokens)
        tokens = self.model.quantizer.idx_to_f(indices)
        tokens = self.model.post_quant_proj(tokens)
        return self.model.fc_norm(tokens.mean(dim=1)).float()


class UniARBSQEncoder(nn.Module):
    """Mean of UniAR's final-block BSQ tokens after the official patch merger."""

    def __init__(self, model: nn.Module, image_size: int):
        super().__init__()
        self.model = model
        self.image_size = int(image_size)
        self.patch_size = int(model.config.patch_size)
        self.temporal_patch_size = int(model.config.temporal_patch_size)
        self.merge_size = int(model.config.spatial_merge_size)
        self.output_dim = int(model.config.out_hidden_size)

        factor = self.patch_size * self.merge_size
        if self.image_size % factor != 0:
            raise ValueError(
                f"UniAR image size {self.image_size} must be divisible by "
                f"patch_size * merge_size = {factor}"
            )

    def _patchify(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Match Qwen2VLImageProcessor's merge-aware image patch ordering."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise RuntimeError(
                f"Expected UniAR input shaped [B,3,H,W], got {tuple(images.shape)}"
            )
        batch_size, channels, height, width = images.shape
        if (height, width) != (self.image_size, self.image_size):
            raise RuntimeError(
                f"Expected {self.image_size}x{self.image_size} UniAR crops, "
                f"got {height}x{width}"
            )

        patch = self.patch_size
        merge = self.merge_size
        grid_h, grid_w = height // patch, width // patch
        patches = images.reshape(
            batch_size,
            channels,
            grid_h // merge,
            merge,
            patch,
            grid_w // merge,
            merge,
            patch,
        )
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
        patches = (
            patches.unsqueeze(6)
            .expand(-1, -1, -1, -1, -1, -1, self.temporal_patch_size, -1, -1)
            .reshape(
                batch_size * grid_h * grid_w,
                channels * self.temporal_patch_size * patch * patch,
            )
        )
        grid_thw = torch.tensor(
            [[1, grid_h, grid_w]] * batch_size,
            dtype=torch.long,
            device=images.device,
        )
        return patches, grid_thw

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches, grid_thw = self._patchify(images)
        model_dtype = next(self.model.parameters()).dtype
        main_tokens, _deepstack_tokens = self.model(
            patches.to(dtype=model_dtype),
            grid_thw=grid_thw,
        )

        batch_size = images.shape[0]
        tokens_per_image = (
            (self.image_size // self.patch_size) ** 2 // (self.merge_size ** 2)
        )
        expected_shape = (batch_size * tokens_per_image, self.output_dim)
        if tuple(main_tokens.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected UniAR final-token shape: expected {expected_shape}, "
                f"got {tuple(main_tokens.shape)}"
            )
        main_tokens = main_tokens.reshape(
            batch_size,
            tokens_per_image,
            self.output_dim,
        )
        return main_tokens.float().mean(dim=1)


class VQGANEncoder(nn.Module):
    """Mean-pooled quantized Taming VQGAN codebook embeddings."""

    def __init__(
        self,
        encoder: nn.Module,
        quant_conv: nn.Module,
        quantize: nn.Module,
        feature_dim: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv
        # Keep the original VQModel attribute name so its state_dict keys load
        # directly without rewriting checkpoint keys.
        self.quantize = quantize
        self.feature_dim = int(feature_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        latent = self.quant_conv(self.encoder(images))
        quantized, _embedding_loss, _info = self.quantize(latent)
        if quantized.ndim != 4 or quantized.shape[1] != self.feature_dim:
            raise RuntimeError(
                "Unexpected VQGAN quantized-latent shape: "
                f"expected [B,{self.feature_dim},H,W], got {tuple(quantized.shape)}"
            )
        return quantized.mean(dim=(2, 3)).float()


def _encode_vilau_penultimate(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    vision_model = model.siglip_model.vision_model
    hidden_states = vision_model.embeddings(images)
    target_index = len(vision_model.encoder.layers) - 2

    for index, encoder_layer in enumerate(vision_model.encoder.layers):
        layer_outputs = encoder_layer(
            hidden_states,
            None,
            output_attentions=None,
        )
        hidden_states = layer_outputs[0]
        if index == target_index:
            return hidden_states.mean(dim=1).float()

    raise RuntimeError("Failed to extract VILA-U penultimate SigLIP tokens")


class VilaUEncoder(nn.Module):
    """Mean of the penultimate SigLIP tokens used by the VILA-U tokenizer."""

    def __init__(self, model: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.dtype = dtype

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return _encode_vilau_penultimate(self.model, images.to(self.dtype))


def _pm1_transforms(image_size: int):
    train_transform = make_classification_train_transform(
        crop_size=image_size,
        hflip_prob=0.5,
        mean=PM1_MEAN,
        std=PM1_STD,
    )
    eval_transform = make_classification_eval_transform(
        resize_size=image_size,
        crop_size=image_size,
        mean=PM1_MEAN,
        std=PM1_STD,
    )
    return train_transform, eval_transform


def _resolve_checkpoint(path: str) -> str:
    checkpoint = Path(path).expanduser().resolve()
    if checkpoint.is_file():
        return str(checkpoint)
    if checkpoint.is_dir():
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "open_clip_pytorch_model.bin",
            "checkpoint.pth",
        ):
            candidate = checkpoint / name
            if candidate.is_file():
                return str(candidate)
    raise FileNotFoundError(f"No supported checkpoint found at {checkpoint}")


def _load_metaclip(
    model_name: str,
    checkpoint_path: str,
    representation: str,
    device: torch.device,
) -> FeatureBundle:
    import timm
    from timm.data import create_transform, resolve_model_data_config
    from timm.models import load_checkpoint

    checkpoint = _resolve_checkpoint(checkpoint_path)
    model = timm.create_model(model_name, pretrained=False)
    load_checkpoint(model, checkpoint, strict=True)
    data_config = resolve_model_data_config(model)
    image_size = int(data_config["input_size"][-1])
    mean = tuple(float(value) for value in data_config["mean"])
    std = tuple(float(value) for value in data_config["std"])

    # Deliberately use the DINO classification augmentation here.  timm's
    # training transform adds ColorJitter, which would make this model's probe
    # use a different augmentation protocol from the other tokenizers.
    train_transform = make_classification_train_transform(
        crop_size=image_size,
        hflip_prob=0.5,
        mean=mean,
        std=std,
    )
    eval_transform = create_transform(**data_config, is_training=False)
    encoder = MetaCLIPEncoder(model).to(device).eval().requires_grad_(False)
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=nullcontext,
        representation=representation,
        transform_description=(
            f"train=RandomResizedCrop({image_size})+HorizontalFlip(0.5), "
            f"eval=timm deterministic native transform, mean={mean}, std={std}, no ColorJitter"
        ),
        backbone_precision="float32",
        checkpoint_paths=[checkpoint],
    )


def _load_metaclip2_distilled(
    checkpoint_path: str,
    image_size: int,
    patch_size: int,
    width: int,
    depth: int,
    projection_dim: int,
    device: torch.device,
) -> FeatureBundle:
    clip_root = Path(__file__).resolve().parent / "vendor" / "CLIP"
    if str(clip_root) not in sys.path:
        sys.path.insert(0, str(clip_root))
    from clip.model import VisionTransformer

    checkpoint = _resolve_checkpoint(checkpoint_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise RuntimeError(f"Expected a MetaCLIP 2 state_dict checkpoint at {checkpoint}")
    state_dict = payload["state_dict"]
    visual_state = {
        key.removeprefix("visual."): value
        for key, value in state_dict.items()
        if key.startswith("visual.")
    }
    required_keys = {"conv1.weight", "positional_embedding", "proj"}
    missing_required = sorted(required_keys - visual_state.keys())
    if missing_required:
        raise RuntimeError(f"Missing MetaCLIP 2 visual keys: {missing_required}")

    actual_width = int(visual_state["conv1.weight"].shape[0])
    actual_patch_size = int(visual_state["conv1.weight"].shape[-1])
    patch_tokens = int(visual_state["positional_embedding"].shape[0]) - 1
    grid_size = math.isqrt(patch_tokens)
    if grid_size * grid_size != patch_tokens:
        raise RuntimeError(f"Non-square MetaCLIP 2 positional grid: {patch_tokens} tokens")
    actual_image_size = grid_size * actual_patch_size
    actual_depth = len(
        {
            key.split(".")[2]
            for key in visual_state
            if key.startswith("transformer.resblocks.") and key.endswith(".attn.in_proj_weight")
        }
    )
    actual_projection_shape = tuple(visual_state["proj"].shape)
    expected = (image_size, patch_size, width, depth, (width, projection_dim))
    actual = (
        actual_image_size,
        actual_patch_size,
        actual_width,
        actual_depth,
        actual_projection_shape,
    )
    if actual != expected:
        raise RuntimeError(f"MetaCLIP 2 visual architecture mismatch: expected={expected}, actual={actual}")

    visual = VisionTransformer(
        input_resolution=image_size,
        patch_size=patch_size,
        width=width,
        layers=depth,
        heads=width // 64,
        output_dim=projection_dim,
    )
    # The distilled MetaCLIP 2 model configs use standard GELU rather than
    # the QuickGELU variants used by MetaCLIP 1.
    for block in visual.transformer.resblocks:
        block.mlp.gelu = nn.GELU()
    visual.load_state_dict(visual_state, strict=True)
    visual.proj = None
    del payload, state_dict, visual_state

    encoder = MetaCLIP2DistilledEncoder(visual).to(device).eval().requires_grad_(False)
    train_transform = make_classification_train_transform(
        crop_size=image_size,
        hflip_prob=0.5,
        mean=CLIP_MEAN,
        std=CLIP_STD,
    )
    eval_transform = make_classification_eval_transform(
        resize_size=image_size,
        crop_size=image_size,
        mean=CLIP_MEAN,
        std=CLIP_STD,
    )
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=nullcontext,
        representation=(
            f"final normalized CLS before the MetaCLIP 2 {width}-to-{projection_dim} projection"
        ),
        transform_description=(
            f"train=RandomResizedCrop({image_size})+HorizontalFlip(0.5), "
            f"eval=Resize({image_size})+CenterCrop({image_size}), "
            f"mean={CLIP_MEAN}, std={CLIP_STD}"
        ),
        backbone_precision="float32",
        checkpoint_paths=[checkpoint],
    )


def _load_siglip2_map(
    model_path: str,
    expected_architecture: str,
    expected_image_size: int,
    expected_feature_dim: int,
    device: torch.device,
) -> FeatureBundle:
    import timm
    from timm.data import create_transform, resolve_model_data_config
    from timm.models import load_checkpoint

    model_dir = Path(model_path).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Missing SigLIP 2 model directory: {model_dir}")
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing SigLIP 2 config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    expected_config = {
        "architecture": expected_architecture,
        "num_classes": 0,
        "num_features": expected_feature_dim,
        "global_pool": "map",
    }
    actual_config = {key: config.get(key) for key in expected_config}
    if actual_config != expected_config:
        raise RuntimeError(
            f"SigLIP 2 model config mismatch: expected={expected_config}, actual={actual_config}"
        )
    pretrained_cfg = config.get("pretrained_cfg")
    if not isinstance(pretrained_cfg, dict):
        raise RuntimeError(f"Missing pretrained_cfg object in {config_path}")
    input_size = pretrained_cfg.get("input_size")
    if input_size != [3, expected_image_size, expected_image_size]:
        raise RuntimeError(
            f"SigLIP 2 input-size mismatch: expected "
            f"{[3, expected_image_size, expected_image_size]}, got {input_size}"
        )

    checkpoint = _resolve_checkpoint(str(model_dir))
    model = timm.create_model(
        expected_architecture,
        pretrained=False,
        pretrained_cfg=pretrained_cfg,
        num_classes=0,
        global_pool="map",
    )
    load_checkpoint(model, checkpoint, strict=True)
    if getattr(model, "attn_pool", None) is None:
        raise RuntimeError("SigLIP 2 model does not expose the expected MAP attention pool")

    data_config = resolve_model_data_config(model)
    image_size = int(data_config["input_size"][-1])
    if image_size != expected_image_size:
        raise RuntimeError(
            f"Resolved SigLIP 2 input size {image_size} does not match {expected_image_size}"
        )
    mean = tuple(float(value) for value in data_config["mean"])
    std = tuple(float(value) for value in data_config["std"])
    train_transform = make_classification_train_transform(
        crop_size=image_size,
        hflip_prob=0.5,
        mean=mean,
        std=std,
    )
    eval_transform = create_transform(**data_config, is_training=False)
    encoder = SigLIP2MAPEncoder(model).to(device).eval().requires_grad_(False)
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=nullcontext,
        representation=(
            "native SigLIP 2 model(images) [B,D] output after MAP attention pooling"
        ),
        transform_description=(
            f"train=RandomResizedCrop({image_size})+HorizontalFlip(0.5), "
            f"eval=timm deterministic native transform, mean={mean}, std={std}, no ColorJitter"
        ),
        backbone_precision="float32",
        checkpoint_paths=[checkpoint, str(config_path)],
    )


def _load_dinov3_l(
    dinov3_path: Path,
    checkpoint: Path,
    device: torch.device,
) -> nn.Module:
    if not (dinov3_path / "hubconf.py").is_file():
        raise FileNotFoundError(f"Missing local DINOv3 repository: {dinov3_path}")
    model = torch.hub.load(
        str(dinov3_path),
        "dinov3_vitl16",
        source="local",
        trust_repo=True,
        skip_validation=True,
        weights=str(checkpoint),
    )
    if int(model.embed_dim) != 1024 or len(model.blocks) != 24:
        raise RuntimeError(
            "Unexpected DINOv3-L architecture: "
            f"embed_dim={model.embed_dim}, depth={len(model.blocks)}"
        )
    # This is part of RAEv2's released representation: its DINOv3 wrapper
    # deliberately removes the final norm's affine parameters.
    model.norm = nn.LayerNorm(1024, elementwise_affine=False)
    return model.to(device).eval().requires_grad_(False)


def _load_ijepa_h(
    raev2_path: Path,
    checkpoint: Path,
    device: torch.device,
) -> nn.Module:
    raev2_src = raev2_path / "src"
    if not (raev2_src / "encoders" / "models" / "jepa.py").is_file():
        raise FileNotFoundError(f"Missing RAEv2 I-JEPA source: {raev2_src}")
    if str(raev2_src) not in sys.path:
        sys.path.insert(0, str(raev2_src))
    from encoders.models.jepa import vit_huge

    model = vit_huge(img_size=[224, 224], patch_size=14)
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    encoder_state = payload.get("encoder") if isinstance(payload, dict) else None
    if not isinstance(encoder_state, dict):
        raise RuntimeError(f"Missing I-JEPA encoder state_dict in {checkpoint}")
    if not encoder_state or any(not key.startswith("module.") for key in encoder_state):
        raise RuntimeError(f"Unexpected I-JEPA encoder key layout in {checkpoint}")
    encoder_state = {
        key.removeprefix("module."): value
        for key, value in encoder_state.items()
    }
    model.load_state_dict(encoder_state, strict=True)
    del payload, encoder_state
    if int(model.embed_dim) != 1280 or len(model.blocks) != 32:
        raise RuntimeError(
            "Unexpected I-JEPA-H architecture: "
            f"embed_dim={model.embed_dim}, depth={len(model.blocks)}"
        )
    return model.to(device).eval().requires_grad_(False)


def _load_raev2_variant(args, device: torch.device, variant: str) -> FeatureBundle:
    spec = RAEV2_SPECS[variant]
    model_root = Path(args.raev2_model_root).expanduser().resolve()
    raev2_path = Path(args.raev2_path).expanduser().resolve()
    dinov3_path = Path(args.dinov3_path).expanduser().resolve()
    stats_path = model_root / spec["stats"]
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing {variant} normalization statistics: {stats_path}")

    if variant in {"dinov3", "raev2", "raev2_dinov3l_k7"}:
        checkpoint = model_root / DINOV3_L_CHECKPOINT
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing DINOv3-L checkpoint: {checkpoint}")
        model = _load_dinov3_l(dinov3_path, checkpoint, device)
        expected_feature_dim = 1024
    else:
        checkpoint = model_root / IJEPA_H_CHECKPOINT
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing I-JEPA-H checkpoint: {checkpoint}")
        model = _load_ijepa_h(raev2_path, checkpoint, device)
        expected_feature_dim = 1280

    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    latent_mean = stats.get("mean") if isinstance(stats, dict) else None
    latent_var = stats.get("var") if isinstance(stats, dict) else None
    expected_stats_shape = (expected_feature_dim, 16, 16)
    if (
        not torch.is_tensor(latent_mean)
        or not torch.is_tensor(latent_var)
        or tuple(latent_mean.shape) != expected_stats_shape
        or tuple(latent_var.shape) != expected_stats_shape
    ):
        mean_shape = None if not torch.is_tensor(latent_mean) else tuple(latent_mean.shape)
        var_shape = None if not torch.is_tensor(latent_var) else tuple(latent_var.shape)
        raise RuntimeError(
            f"{variant} normalization-stat shape mismatch: expected {expected_stats_shape}, "
            f"got mean={mean_shape}, var={var_shape}"
        )
    if not torch.isfinite(latent_mean).all() or not torch.isfinite(latent_var).all():
        raise RuntimeError(f"{variant} normalization statistics contain non-finite values")
    if not torch.all(latent_var >= 0):
        raise RuntimeError(f"{variant} normalization variance contains negative values")

    encoder = RAEv2LatentEncoder(
        model,
        variant,
        latent_mean,
        latent_var,
    ).to(device).eval().requires_grad_(False)
    train_transform = make_classification_train_transform(
        crop_size=256,
        hflip_prob=0.5,
        mean=IDENTITY_MEAN,
        std=IDENTITY_STD,
    )
    eval_transform = make_classification_eval_transform(
        resize_size=256,
        crop_size=256,
        mean=IDENTITY_MEAN,
        std=IDENTITY_STD,
    )
    source_paths = [str(raev2_path)]
    if variant in {"dinov3", "raev2", "raev2_dinov3l_k7"}:
        source_paths.append(str(dinov3_path))
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=nullcontext,
        representation=spec["representation"],
        transform_description=(
            "train=RandomResizedCrop(256)+HorizontalFlip(0.5), "
            "eval=Resize(256)+CenterCrop(256), transform output in [0,1]; "
            f"encoder-internal normalization mean={IMAGENET_MEAN}, std={IMAGENET_STD}"
        ),
        backbone_precision="float32",
        checkpoint_paths=[str(checkpoint), str(stats_path), *source_paths],
    )


def _processor_square_size(value, name: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        shortest_edge = value.get("shortest_edge")
        height = value.get("height")
        width = value.get("width")
    else:
        shortest_edge = getattr(value, "shortest_edge", None)
        height = getattr(value, "height", None)
        width = getattr(value, "width", None)
    if shortest_edge is not None:
        return int(shortest_edge)
    if height is not None and width is not None and int(height) == int(width):
        return int(height)
    raise ValueError(f"Expected a square image-processor {name}, got {value!r}")


def _load_perception_encoder(
    model_root: str,
    spec,
    device: torch.device,
) -> FeatureBundle:
    import timm
    from timm.models import load_checkpoint
    from timm.models.eva import checkpoint_filter_fn

    directory, architecture, image_size, readout_kind, feature_dim, projection_dim = spec
    model_dir = Path(model_root).expanduser().resolve() / directory
    checkpoint = model_dir / f"{directory}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing Perception Encoder checkpoint: {checkpoint}")

    model = timm.create_model(
        architecture,
        pretrained=False,
        num_classes=projection_dim or 0,
        dtype=torch.bfloat16,
    )
    if int(model.num_features) != feature_dim:
        raise RuntimeError(
            f"Perception Encoder width mismatch: expected {feature_dim}, got {model.num_features}"
        )
    if readout_kind == "attention_pool":
        if getattr(model, "attn_pool", None) is None:
            raise RuntimeError("PE-Core model does not expose its attention pool")
        if not isinstance(model.head, nn.Linear):
            raise RuntimeError("PE-Core model does not expose its projection head")
        head_shape = (int(model.head.in_features), int(model.head.out_features))
        if head_shape != (feature_dim, projection_dim):
            raise RuntimeError(
                "PE-Core projection mismatch: "
                f"expected {(feature_dim, projection_dim)}, got {head_shape}"
            )

    load_checkpoint(
        model,
        str(checkpoint),
        strict=True,
        filter_fn=checkpoint_filter_fn,
    )
    encoder = (
        PerceptionEncoderReadout(model, readout_kind)
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    train_transform, eval_transform = _pm1_transforms(image_size)
    if readout_kind == "attention_pool":
        representation = (
            "PE-Core learned attention-pool output before the released CLIP projection"
        )
    else:
        representation = (
            "mean of last-layer Perception Encoder patch tokens, excluding any CLS token"
        )
    tiling_note = (
        "; tiling-aligned checkpoint evaluated as one native 448x448 crop"
        if directory.endswith("-Tiling")
        else ""
    )
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=partial(torch.autocast, device_type="cuda", dtype=torch.bfloat16),
        representation=representation,
        transform_description=(
            f"train=RandomResizedCrop({image_size})+HorizontalFlip(0.5), "
            f"eval=Resize({image_size})+CenterCrop({image_size}), "
            f"mean={PM1_MEAN}, std={PM1_STD}{tiling_note}"
        ),
        backbone_precision="bfloat16 weights and autocast",
        checkpoint_paths=[str(checkpoint.resolve())],
    )


def _load_hf_visual_encoder(
    model_root: str,
    spec,
    readout_kind: str,
    device: torch.device,
) -> FeatureBundle:
    # transformers 5.10 references this newly added PyTorch dtype while
    # importing optional FP8 support. These checkpoints do not use FP8; the
    # alias only keeps transformers importable with the workspace's torch 2.6.
    if not hasattr(torch, "float8_e8m0fnu"):
        torch.float8_e8m0fnu = torch.float8_e4m3fn
    from transformers import AutoConfig, AutoImageProcessor, AutoModel

    directory, expected_model_type = spec
    model_dir = Path(model_root).expanduser().resolve() / directory
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Missing Hugging Face visual model directory: {model_dir}")
    config_path = model_dir / "config.json"
    processor_path = model_dir / "preprocessor_config.json"
    if not config_path.is_file() or not processor_path.is_file():
        raise FileNotFoundError(
            f"Expected config.json and preprocessor_config.json in {model_dir}"
        )
    checkpoint = _resolve_checkpoint(str(model_dir))

    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    if config.model_type != expected_model_type:
        raise RuntimeError(
            f"HF model_type mismatch: expected {expected_model_type}, got {config.model_type}"
        )
    model_kwargs = (
        {"add_pooling_layer": False}
        if expected_model_type == "vit" and readout_kind == "cls_patch"
        else {}
    )
    torch_version = tuple(
        int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2]
    )
    # Transformers rejects every PyTorch .bin checkpoint on torch<2.6 after
    # CVE-2025-32434, even when the caller requests weights_only=True.  These
    # two legacy DINOv1 files are trusted local model-zoo state_dicts. Load
    # them explicitly with the safe tensor-only unpickler and strict key
    # validation; all other HF checkpoints retain from_pretrained semantics.
    if Path(checkpoint).suffix == ".bin" and torch_version < (2, 6):
        model = AutoModel.from_config(config, **model_kwargs)
        state_dict = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(state_dict, dict) or not state_dict:
            raise RuntimeError(f"Expected a non-empty tensor state_dict in {checkpoint}")
        invalid_values = [
            key for key, value in state_dict.items() if not torch.is_tensor(value)
        ]
        if invalid_values:
            raise RuntimeError(
                f"Legacy HF checkpoint contains non-tensor values: {invalid_values[:10]}"
            )
        incompatible = model.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Failed to strictly load {directory}: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        del state_dict
        model = model.to(dtype=torch.bfloat16)
        loading_info = {}
    else:
        model, loading_info = AutoModel.from_pretrained(
            str(model_dir),
            config=config,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            output_loading_info=True,
            **model_kwargs,
        )
    loading_errors = {
        key: loading_info.get(key, [])
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(key)
    }
    if loading_errors:
        raise RuntimeError(f"Failed to strictly load {directory}: {loading_errors}")
    processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)

    crop_value = getattr(processor, "crop_size", None)
    image_size = _processor_square_size(
        crop_value if crop_value is not None else processor.size,
        "crop_size" if crop_value is not None else "size",
    )
    resize_size = _processor_square_size(processor.size, "size")
    mean = tuple(float(value) for value in processor.image_mean)
    std = tuple(float(value) for value in processor.image_std)
    train_transform = make_classification_train_transform(
        crop_size=image_size,
        hflip_prob=0.5,
        mean=mean,
        std=std,
    )
    eval_transform = make_classification_eval_transform(
        resize_size=resize_size,
        crop_size=image_size,
        mean=mean,
        std=std,
    )

    if readout_kind == "cls_patch":
        register_count = int(getattr(config, "num_register_tokens", 0) or 0)
        encoder = HFClsPatchEncoder(model, register_count)
        representation = (
            "concat(final normalized CLS, mean(final normalized patch tokens)); "
            f"excludes {register_count} register token(s)"
        )
    elif readout_kind == "convnext_gap":
        encoder = HFConvNeXtGlobalEncoder(model)
        representation = "DINOv3 ConvNeXt final-stage GAP after the released final LayerNorm"
    elif readout_kind == "cls":
        encoder = HFClsEncoder(model)
        representation = "final normalized Web-MAE encoder CLS token"
    else:
        raise ValueError(f"Unsupported Hugging Face visual readout: {readout_kind}")

    encoder = encoder.to(device).eval().requires_grad_(False)
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=partial(torch.autocast, device_type="cuda", dtype=torch.bfloat16),
        representation=representation,
        transform_description=(
            f"train=RandomResizedCrop({image_size})+HorizontalFlip(0.5), "
            f"eval=Resize({resize_size})+CenterCrop({image_size}), mean={mean}, std={std}"
        ),
        backbone_precision="bfloat16 weights and autocast",
        checkpoint_paths=[checkpoint, str(config_path), str(processor_path)],
    )


def _load_pixio(
    model_root: str,
    spec,
    device: torch.device,
    readout: str = "post-ln",
) -> FeatureBundle:
    from safetensors.torch import load_file

    directory, expected_hidden_size, expected_depth, expected_heads = spec
    model_dir = Path(model_root).expanduser().resolve() / directory
    config_path = model_dir / "config.json"
    processor_path = model_dir / "preprocessor_config.json"
    checkpoint_path = model_dir / "model.safetensors"
    for required_path in (config_path, processor_path, checkpoint_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing Pixio model file: {required_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    expected_config = {
        "model_type": "pixio",
        "image_size": 256,
        "patch_size": 16,
        "hidden_size": expected_hidden_size,
        "num_hidden_layers": expected_depth,
        "num_attention_heads": expected_heads,
        "mlp_ratio": 4,
        "n_cls_tokens": 8,
        "layer_norm_eps": 1e-6,
    }
    actual_config = {key: config.get(key) for key in expected_config}
    if actual_config != expected_config:
        raise RuntimeError(
            f"Pixio config mismatch for {directory}: "
            f"expected={expected_config}, actual={actual_config}"
        )

    with processor_path.open("r", encoding="utf-8") as handle:
        processor = json.load(handle)
    mean = tuple(float(value) for value in processor.get("image_mean", ()))
    std = tuple(float(value) for value in processor.get("image_std", ()))
    if mean != IMAGENET_MEAN or std != IMAGENET_STD:
        raise RuntimeError(
            f"Unexpected Pixio normalization for {directory}: mean={mean}, std={std}"
        )

    # Construct on the meta device, assign the mmap-backed safetensors, then
    # materialize directly as BF16 on the GPU. This avoids an additional full
    # FP32 allocation, which is important for Pixio-1B and Pixio-5B.
    with torch.device("meta"):
        model = PixioEncoder(
            image_size=expected_config["image_size"],
            patch_size=expected_config["patch_size"],
            hidden_size=expected_hidden_size,
            depth=expected_depth,
            num_heads=expected_heads,
            mlp_ratio=expected_config["mlp_ratio"],
            n_cls_tokens=expected_config["n_cls_tokens"],
            layer_norm_eps=expected_config["layer_norm_eps"],
            readout=readout,
        )
    # Official safetensors releases (including 0.7) memory-map files
    # internally and expose only ``filename`` and ``device`` here.  Some
    # development builds accepted a non-standard ``backend="mmap"`` keyword,
    # which makes the loader fail before reading any weights on official builds.
    state_dict = load_file(str(checkpoint_path), device="cpu")
    incompatible = model.load_state_dict(state_dict, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Failed to strictly load {directory}: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    del state_dict
    model = model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)

    image_size = expected_config["image_size"]
    train_transform = make_classification_train_transform(
        crop_size=image_size,
        hflip_prob=0.5,
        mean=mean,
        std=std,
    )
    eval_transform = make_classification_eval_transform(
        resize_size=image_size,
        crop_size=image_size,
        mean=mean,
        std=std,
    )
    return FeatureBundle(
        encoder=model,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=partial(
            torch.autocast,
            device_type="cuda",
            dtype=torch.bfloat16,
        ),
        representation=(
            "mean of the eight final-LayerNorm Pixio CLS tokens"
            if readout == "post-ln"
            else "mean of the eight pre-LayerNorm Pixio CLS tokens"
        ),
        transform_description=(
            f"train=RandomResizedCrop({image_size})+HorizontalFlip(0.5), "
            f"eval=Resize({image_size})+CenterCrop({image_size}), "
            f"mean={mean}, std={std}"
        ),
        backbone_precision="bfloat16 weights and autocast",
        checkpoint_paths=[
            str(checkpoint_path),
            str(config_path),
            str(processor_path),
        ],
    )


def _load_eupe(
    model_root: str,
    dinov3_path: Path,
    spec,
    device: torch.device,
) -> FeatureBundle:
    from torchvision import transforms

    directory, family, size, feature_dim = spec
    checkpoint = Path(model_root).expanduser().resolve() / directory / f"{directory}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing EUPE checkpoint: {checkpoint}")

    dinov3_path = Path(dinov3_path).expanduser().resolve()
    if not (dinov3_path / "dinov3" / "models" / "vision_transformer.py").is_file():
        raise FileNotFoundError(f"Missing compatible local DINOv3 source: {dinov3_path}")
    if str(dinov3_path) not in sys.path:
        sys.path.insert(0, str(dinov3_path))

    if family == "vit":
        from dinov3.models.vision_transformer import DinoVisionTransformer

        vit_sizes = {
            "tiny": (192, 3),
            "small": (384, 6),
            "base": (768, 12),
        }
        embed_dim, num_heads = vit_sizes[size]
        model = DinoVisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=embed_dim,
            depth=12,
            num_heads=num_heads,
            ffn_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-5,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        )
        encoder_type = EUPEViTEncoder
        representation = (
            "concat(final normalized EUPE CLS, mean(final normalized patch tokens)); "
            "excludes 4 storage tokens"
        )
    elif family == "convnext":
        from dinov3.models.convnext import ConvNeXt

        convnext_sizes = {
            "tiny": ([3, 3, 9, 3], [96, 192, 384, 768]),
            "small": ([3, 3, 27, 3], [96, 192, 384, 768]),
            "base": ([3, 3, 27, 3], [128, 256, 512, 1024]),
        }
        depths, dims = convnext_sizes[size]
        model = ConvNeXt(
            in_chans=3,
            depths=depths,
            dims=dims,
            drop_path_rate=0.0,
            layer_scale_init_value=1.0e-6,
        )
        encoder_type = EUPEConvNeXtEncoder
        representation = (
            "EUPE ConvNeXt final-stage global average pooling after the released "
            "final LayerNorm"
        )
    else:
        raise ValueError(f"Unsupported EUPE architecture family: {family}")

    if int(model.embed_dim) != feature_dim:
        raise RuntimeError(
            f"EUPE feature-width mismatch: expected {feature_dim}, got {model.embed_dim}"
        )

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"Expected a non-empty EUPE state_dict at {checkpoint}")
    backbone_keys = set(model.state_dict())
    backbone_state = {key: value for key, value in payload.items() if key in backbone_keys}
    missing_keys = sorted(backbone_keys - set(backbone_state))
    non_backbone_keys = sorted(set(payload) - backbone_keys)
    invalid_extra_keys = [
        key for key in non_backbone_keys if not key.startswith("projectors.")
    ]
    if missing_keys or invalid_extra_keys or not non_backbone_keys:
        raise RuntimeError(
            f"Unexpected EUPE checkpoint layout for {directory}: "
            f"missing_backbone={missing_keys}, invalid_extra={invalid_extra_keys}, "
            f"training_projector_keys={len(non_backbone_keys)}"
        )
    model = model.to(dtype=torch.bfloat16)
    model.load_state_dict(backbone_state, strict=True)
    del payload, backbone_state

    encoder = encoder_type(model).to(device).eval().requires_grad_(False)
    image_size = 256
    train_transform = make_classification_train_transform(
        crop_size=image_size,
        hflip_prob=0.5,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=partial(torch.autocast, device_type="cuda", dtype=torch.bfloat16),
        representation=representation,
        transform_description=(
            f"train=RandomResizedCrop({image_size})+HorizontalFlip(0.5), "
            f"eval=direct square Resize(({image_size},{image_size})) per EUPE release, "
            f"mean={IMAGENET_MEAN}, std={IMAGENET_STD}"
        ),
        backbone_precision="bfloat16 weights and autocast",
        checkpoint_paths=[str(checkpoint)],
    )


def _load_clip_openai_l14(args, device: torch.device) -> FeatureBundle:
    from transformers import CLIPImageProcessor, CLIPVisionModel

    model_path = Path(args.clip_openai_model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Missing OpenAI CLIP model directory: {model_path}")
    checkpoint = _resolve_checkpoint(str(model_path))
    model, loading_info = CLIPVisionModel.from_pretrained(
        str(model_path),
        local_files_only=True,
        output_loading_info=True,
    )
    unexpected_vision_keys = [
        key for key in loading_info["unexpected_keys"] if key.startswith("vision_model.")
    ]
    if (
        loading_info["missing_keys"]
        or loading_info["mismatched_keys"]
        or loading_info["error_msgs"]
        or unexpected_vision_keys
    ):
        raise RuntimeError(f"Failed to strictly load the OpenAI CLIP vision tower: {loading_info}")
    processor = CLIPImageProcessor.from_pretrained(str(model_path), local_files_only=True)

    image_size = int(model.config.image_size)
    crop_size = _processor_square_size(processor.crop_size, "crop_size")
    resize_size = _processor_square_size(processor.size, "size")
    if crop_size != image_size:
        raise ValueError(
            f"CLIP processor crop size {crop_size} does not match model image size {image_size}"
        )
    mean = tuple(float(value) for value in processor.image_mean)
    std = tuple(float(value) for value in processor.image_std)
    train_transform = make_classification_train_transform(
        crop_size=image_size,
        hflip_prob=0.5,
        mean=mean,
        std=std,
    )
    eval_transform = make_classification_eval_transform(
        resize_size=resize_size,
        crop_size=image_size,
        mean=mean,
        std=std,
    )
    encoder = OpenAIClipEncoder(model).to(device).eval().requires_grad_(False)
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=nullcontext,
        representation="final post-LayerNorm CLS before the OpenAI CLIP 1024-to-768 visual projection",
        transform_description=(
            f"train=RandomResizedCrop({image_size})+HorizontalFlip(0.5), "
            f"eval=Resize({resize_size})+CenterCrop({image_size}), mean={mean}, std={std}"
        ),
        backbone_precision="float32",
        checkpoint_paths=[checkpoint],
    )


def _load_toklip(args, device: torch.device, variant: str) -> FeatureBundle:
    if str(args.image_scripts) not in sys.path:
        sys.path.insert(0, str(args.image_scripts))
    from toklip_rec_common import encode_toklip_semantic_tokens, load_toklip_semantic_model

    is_small = variant == "s"
    image_size = 256 if is_small else 384
    model_name = "toklip_s" if is_small else "toklip_l"
    checkpoint = args.toklip_s_checkpoint if is_small else args.toklip_l_checkpoint
    checkpoint = str(Path(checkpoint).expanduser().resolve())
    vq_checkpoint = str(Path(args.toklip_vq_checkpoint).expanduser().resolve())
    trunk = load_toklip_semantic_model(
        SimpleNamespace(
            toklip_path=str(Path(args.toklip_path).expanduser().resolve()),
            toklip_ckpt_path=checkpoint,
            vq_ckpt_path=vq_checkpoint,
            model_name=model_name,
            toklip_model_config=None,
        ),
        device=str(device),
    )
    encoder = TokLIPEncoder(trunk, encode_toklip_semantic_tokens)
    encoder = encoder.to(device).eval().requires_grad_(False)
    train_transform, eval_transform = _pm1_transforms(image_size)
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=nullcontext,
        representation="mean of final normalized TokLIP semantic tokens; forward_head is not used",
        transform_description=(
            f"train=RandomResizedCrop({image_size})+HorizontalFlip(0.5), "
            f"eval=Resize({image_size})+CenterCrop({image_size}), mean={PM1_MEAN}, std={PM1_STD}"
        ),
        backbone_precision="float32",
        checkpoint_paths=[checkpoint, vq_checkpoint],
    )


def _load_unitok(args, device: torch.device) -> FeatureBundle:
    if str(args.image_scripts) not in sys.path:
        sys.path.insert(0, str(args.image_scripts))
    from unitok_vae_rec import load_unitok

    checkpoint = str(Path(args.unitok_checkpoint).expanduser().resolve())
    model, _preprocess = load_unitok(
        SimpleNamespace(
            unitok_path=str(Path(args.unitok_path).expanduser().resolve()),
            ckpt_path=checkpoint,
        )
    )
    encoder = UniTokEncoder(model).to(device).eval().requires_grad_(False)
    train_transform, eval_transform = _pm1_transforms(256)
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=nullcontext,
        representation=(
            "quantize/dequantize -> post_quant_proj -> token mean -> fc_norm, before UniTok projection"
        ),
        transform_description=(
            f"train=RandomResizedCrop(256)+HorizontalFlip(0.5), "
            f"eval=Resize(256)+CenterCrop(256), mean={PM1_MEAN}, std={PM1_STD}"
        ),
        backbone_precision="float32",
        checkpoint_paths=[checkpoint],
    )


def _load_uniar_bsq(args, device: torch.device) -> FeatureBundle:
    source_root = Path(args.uniar_path).expanduser().resolve()
    source_file = source_root / "uniar" / "vision_encoder" / "modeling_vision_encoder.py"
    if not source_file.is_file():
        raise FileNotFoundError(
            f"Missing official UniAR source checkout: {source_root}. "
            "Clone https://github.com/ShareLab-SII/UniAR.git there or pass --uniar-path."
        )

    checkpoint_dir = Path(args.uniar_checkpoint).expanduser().resolve()
    if (checkpoint_dir / "bsq_encoder").is_dir():
        checkpoint_dir /= "bsq_encoder"
    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"Expected config.json and model.safetensors in {checkpoint_dir}"
        )

    # transformers 5.10 references this PyTorch 2.7 dtype while importing
    # optional FP8 support. UniAR does not use FP8, so retain the compatibility
    # alias already used by the other local Hugging Face feature extractors.
    if not hasattr(torch, "float8_e8m0fnu"):
        torch.float8_e8m0fnu = torch.float8_e4m3fn
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        from uniar.vision_encoder.configuration_vision_encoder import UniARVisionConfig
        from uniar.vision_encoder.modeling_vision_encoder import UniARVisionModel
    except ImportError as error:
        raise RuntimeError(
            "Failed to import the official UniAR vision encoder. Install its "
            "Qwen3-VL-compatible dependencies (the release pins transformers==4.57.0)."
        ) from error

    config = UniARVisionConfig.from_pretrained(
        str(checkpoint_dir),
        local_files_only=True,
    )
    expected_config = {
        "depth": 27,
        "hidden_size": 1152,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "out_hidden_size": 4096,
        "bsq_dim": 64,
        "deepstack_visual_indexes": [8, 16, 24],
        "use_bsq": True,
    }
    for name, expected in expected_config.items():
        actual = getattr(config, name)
        if actual != expected:
            raise RuntimeError(
                f"Unexpected UniAR config {name}: expected {expected!r}, got {actual!r}"
            )

    model, loading_info = UniARVisionModel.from_pretrained(
        str(checkpoint_dir),
        config=config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    missing = loading_info.get("missing_keys", [])
    unexpected = loading_info.get("unexpected_keys", [])
    mismatched = loading_info.get("mismatched_keys", [])
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "UniAR checkpoint did not load strictly: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )

    image_size = int(args.uniar_image_size)
    encoder = UniARBSQEncoder(model, image_size)
    encoder = encoder.to(device).eval().requires_grad_(False)
    train_transform, eval_transform = _pm1_transforms(image_size)
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=partial(torch.autocast, device_type="cuda", dtype=torch.bfloat16),
        representation=(
            "mean of UniAR block 27 tokens after 64-bit BSQ quantize/dequantize, "
            "BSQ output projection, and the official 2x2 patch merger"
        ),
        transform_description=(
            f"train=RandomResizedCrop({image_size})+HorizontalFlip(0.5), "
            f"eval=Resize({image_size})+CenterCrop({image_size}), "
            f"mean={PM1_MEAN}, std={PM1_STD}; merge-aware Qwen patch ordering"
        ),
        backbone_precision="bfloat16 weights and autocast",
        checkpoint_paths=[str(config_path), str(weights_path)],
    )


def _load_vilau(args, device: torch.device) -> FeatureBundle:
    if str(args.image_scripts) not in sys.path:
        sys.path.insert(0, str(args.image_scripts))
    from vilau_rec import load_vilau_tokenizer

    model_path = str(Path(args.vilau_model_path).expanduser().resolve())
    siglip_config_path = str(Path(args.vilau_siglip_config).expanduser().resolve())
    tokenizer = load_vilau_tokenizer(
        SimpleNamespace(
            vilau_path=str(Path(args.vilau_path).expanduser().resolve()),
            model_path=model_path,
            siglip_config_path=siglip_config_path,
            dtype="bfloat16",
        ),
        device=str(device),
    )
    encoder = VilaUEncoder(tokenizer.model, tokenizer.dtype)
    encoder = encoder.to(device).eval().requires_grad_(False)
    train_transform, eval_transform = _pm1_transforms(tokenizer.image_size)
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=partial(torch.autocast, device_type="cuda", dtype=torch.bfloat16),
        representation="mean of the penultimate VILA-U SigLIP encoder-block tokens",
        transform_description=(
            f"train=RandomResizedCrop({tokenizer.image_size})+HorizontalFlip(0.5), "
            f"eval=Resize({tokenizer.image_size})+CenterCrop({tokenizer.image_size}), "
            f"mean={PM1_MEAN}, std={PM1_STD}"
        ),
        backbone_precision="bfloat16",
        checkpoint_paths=[model_path, siglip_config_path],
    )


def _load_vqgan(args, device: torch.device) -> FeatureBundle:
    taming_path = Path(args.vqgan_path).expanduser().resolve()
    config_path = Path(args.vqgan_config).expanduser().resolve()
    checkpoint_path = Path(args.vqgan_checkpoint).expanduser().resolve()
    if not taming_path.is_dir():
        raise FileNotFoundError(f"Missing taming-transformers directory: {taming_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing VQGAN config: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing VQGAN checkpoint: {checkpoint_path}")
    if str(taming_path) not in sys.path:
        sys.path.insert(0, str(taming_path))

    from omegaconf import OmegaConf
    from taming.modules.diffusionmodules.model import Encoder
    from taming.modules.vqvae.quantize import VectorQuantizer2

    config = OmegaConf.load(config_path)
    if config.model.target != "taming.models.vqgan.VQModel":
        raise ValueError(f"Unsupported VQGAN model target: {config.model.target}")
    params = config.model.params
    ddconfig = OmegaConf.to_container(params.ddconfig, resolve=True)
    image_size = int(ddconfig["resolution"])
    feature_dim = int(params.embed_dim)
    n_embed = int(params.n_embed)

    # Build exactly the three VQModel submodules used by encode().  Importing
    # VQModel itself would require PyTorch Lightning and would also construct
    # the unused decoder, discriminator, and LPIPS/VGG tower.
    encoder = VQGANEncoder(
        Encoder(**ddconfig),
        nn.Conv2d(int(ddconfig["z_channels"]), feature_dim, kernel_size=1),
        VectorQuantizer2(
            n_embed,
            feature_dim,
            beta=0.25,
            remap=params.get("remap"),
            sane_index_shape=bool(params.get("sane_index_shape", False)),
        ),
        feature_dim,
    )

    # This legacy Lightning checkpoint contains one serialized ModelCheckpoint
    # callback alongside its tensors.  A local inert stand-in lets PyTorch's
    # restricted weights-only loader read it without installing Lightning or
    # executing arbitrary pickle globals.
    lightning_checkpoint_stub = type("ModelCheckpoint", (), {})
    lightning_checkpoint_stub.__module__ = (
        "pytorch_lightning.callbacks.model_checkpoint"
    )
    previous_safe_globals = list(torch.serialization.get_safe_globals())
    torch.serialization.add_safe_globals([lightning_checkpoint_stub])
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    finally:
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(previous_safe_globals)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise RuntimeError(f"VQGAN checkpoint has no state_dict: {checkpoint_path}")
    encoder_state = {
        key: value
        for key, value in payload["state_dict"].items()
        if key.startswith(("encoder.", "quant_conv.", "quantize."))
    }
    encoder.load_state_dict(encoder_state, strict=True)
    del payload, encoder_state
    encoder = encoder.to(device).eval().requires_grad_(False)
    train_transform, eval_transform = _pm1_transforms(image_size)
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=nullcontext,
        representation=(
            "spatial mean of quantized Taming VQGAN codebook embeddings, "
            "before post_quant_conv"
        ),
        transform_description=(
            f"train=RandomResizedCrop({image_size})+HorizontalFlip(0.5), "
            f"eval=Resize({image_size})+CenterCrop({image_size}), "
            f"mean={PM1_MEAN}, std={PM1_STD}"
        ),
        backbone_precision="float32",
        checkpoint_paths=[str(config_path), str(checkpoint_path)],
    )


def load_feature_bundle(model_name: str, args, device: torch.device) -> FeatureBundle:
    if model_name == "metaclip":
        return _load_metaclip(
            args.metaclip_model,
            args.metaclip_checkpoint,
            "final normalized CLS before the MetaCLIP 768-to-512 projection",
            device,
        )
    if model_name == "clip_openai__l14":
        return _load_clip_openai_l14(args, device)
    if model_name == "clip_meta__l14":
        return _load_metaclip(
            args.clip_meta_model,
            args.clip_meta_checkpoint,
            "final normalized CLS before the MetaCLIP 1024-to-768 projection",
            device,
        )
    if model_name in MC1_SPECS:
        timm_model_name, checkpoint_arg, representation = MC1_SPECS[model_name]
        return _load_metaclip(
            timm_model_name,
            getattr(args, checkpoint_arg),
            representation,
            device,
        )
    if model_name in MC2_TIMM_SPECS:
        timm_model_name, checkpoint_arg, representation = MC2_TIMM_SPECS[model_name]
        return _load_metaclip(
            timm_model_name,
            getattr(args, checkpoint_arg),
            representation,
            device,
        )
    if model_name in MC2_DISTILLED_SPECS:
        checkpoint_arg, image_size, patch_size, width, depth, projection_dim = (
            MC2_DISTILLED_SPECS[model_name]
        )
        return _load_metaclip2_distilled(
            getattr(args, checkpoint_arg),
            image_size,
            patch_size,
            width,
            depth,
            projection_dim,
            device,
        )
    if model_name in SIGLIP2_SPECS:
        architecture, model_path_arg, image_size, feature_dim = SIGLIP2_SPECS[model_name]
        return _load_siglip2_map(
            getattr(args, model_path_arg),
            architecture,
            image_size,
            feature_dim,
            device,
        )
    if model_name in PE_SPECS:
        return _load_perception_encoder(
            args.continuous_model_root,
            PE_SPECS[model_name],
            device,
        )
    if model_name in DINO_VIT_SPECS:
        return _load_hf_visual_encoder(
            args.continuous_model_root,
            DINO_VIT_SPECS[model_name],
            "cls_patch",
            device,
        )
    if model_name in DINOV3_CONVNEXT_SPECS:
        return _load_hf_visual_encoder(
            args.continuous_model_root,
            DINOV3_CONVNEXT_SPECS[model_name],
            "convnext_gap",
            device,
        )
    if model_name in WEBSSL_DINO_SPECS:
        return _load_hf_visual_encoder(
            args.continuous_model_root,
            WEBSSL_DINO_SPECS[model_name],
            "cls_patch",
            device,
        )
    if model_name in WEBSSL_MAE_SPECS:
        return _load_hf_visual_encoder(
            args.continuous_model_root,
            WEBSSL_MAE_SPECS[model_name],
            "cls",
            device,
        )
    if model_name in PIXIO_SPECS:
        return _load_pixio(
            args.continuous_model_root,
            PIXIO_SPECS[model_name],
            device,
            readout=getattr(args, "pixio_readout", "post-ln"),
        )
    if model_name in EUPE_SPECS:
        return _load_eupe(
            args.continuous_model_root,
            args.dinov3_path,
            EUPE_SPECS[model_name],
            device,
        )
    if model_name in RAEV2_SPECS:
        return _load_raev2_variant(args, device, model_name)
    if model_name == "toklip_s":
        return _load_toklip(args, device, "s")
    if model_name == "toklip_l":
        return _load_toklip(args, device, "l")
    if model_name == "unitok":
        return _load_unitok(args, device)
    if model_name == "uniar_bsq":
        return _load_uniar_bsq(args, device)
    if model_name == "vilau":
        return _load_vilau(args, device)
    if model_name == "vqgan":
        return _load_vqgan(args, device)
    raise ValueError(f"Unsupported tokenizer model: {model_name}")
