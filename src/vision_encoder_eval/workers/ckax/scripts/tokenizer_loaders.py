# -*- coding: utf-8 -*-
"""
Additional tokenizer loaders for VTBench feature extraction.

Handled types: hf / dinov3 / eupe / raev2 / ijepa / pixio / pe.
Feature convention: patch tokens from the final layer mean-pooled + L2-normed
(select_feature == "cls" uses the class token). Storage/register tokens and
Pixio's extra cls tokens are dropped.
"""

import math
import os
import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

try:
    import numpy as np
    from numpy.core.multiarray import scalar as np_scalar
    torch.serialization.add_safe_globals([np_scalar])
except Exception:
    pass
try:
    torch.serialization.add_safe_globals([np.dtype])
except Exception:
    pass


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _make_preprocess(image_size, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _load_state_dict(path):
    """Load a checkpoint, unwrap container keys and strip a ``module.`` prefix."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise RuntimeError(f"checkpoint is not a dict: {type(obj).__name__}")
    sd = obj
    for key in ("state_dict", "model", "teacher", "target_encoder", "encoder"):
        if key in obj and isinstance(obj[key], dict):
            sd = obj[key]
            break
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}


def _ml_indices(L, official=None):
    """Union of relative-depth and official-layer-anchored block indices."""
    try:
        from vision_encoder_eval.workers.ckax.scripts.extract_features import get_multi_layer_indices as _g
        return _g(L, official)
    except Exception:
        if official is None:
            official = L - 1
        rel = [int(round(f * official)) for f in (0.25, 0.5, 0.75, 0.875, 1.0)]
        abs_idx = ([L + o for o in (-2, -4, -6, -8)] + [0, 1]
                   + [official - 1, official, official + 1])
        return sorted(set(rel) | {i for i in abs_idx if 0 <= i < L})


def _enable_multilayer(enc, model, default_layer=-1, multi_layer=False):
    """Attach the cross-layer layer set to a _FeatureEncoder (no-op if off)."""
    if not multi_layer:
        return enc
    L = None
    stages = getattr(model, 'stages', None)
    blocks = getattr(model, 'blocks', None)
    if isinstance(stages, nn.ModuleList):
        L = len(stages)                       # ConvNeXt: stage-level depth points
    elif isinstance(blocks, nn.ModuleList):
        L = len(blocks)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'resblocks'):
        L = len(model.transformer.resblocks)
    elif hasattr(model, 'config'):
        L = (getattr(model.config, 'num_hidden_layers', None)
             or getattr(getattr(model.config, 'vision_config', None), 'num_hidden_layers', None))
    if not L:
        return enc
    official = (L + default_layer) if default_layer < 0 else default_layer
    layers = _ml_indices(L, official)
    enc.layers = layers
    enc.total_layers = L
    enc.default_layer_idx = (L + default_layer) if default_layer < 0 else default_layer
    return enc


class _FeatureEncoder:
    def __init__(self, model, preprocess, feat_dim, device="cuda", layers=None,
                 total_layers=None, default_layer_idx=None):
        self.model = model
        self.preprocess = preprocess
        self.feat_dim = feat_dim
        self.device = device
        self.layers = layers
        self.total_layers = total_layers
        self.default_layer_idx = default_layer_idx

    @torch.no_grad()
    def encode_images(self, images, normalize=True):
        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        return self.encode_tensors(tensors, normalize)

    def encode_tensors(self, tensors, normalize=True):
        raise NotImplementedError

    def cleanup(self):
        pass


def _pool_and_norm(feat, normalize=True):
    feat = feat.float()
    if feat.dim() == 3:
        feat = feat.mean(dim=1)
    if normalize:
        feat = F.normalize(feat, dim=-1)
    return feat


# --- DINOv3 / EUPE ViT backbone (RoPE + storage tokens + layer scale) --- #
def _rope_rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _rope_apply(x, sin, cos):
    return x * cos + _rope_rotate_half(x) * sin


class _LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class _MaskedBiasLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.ones_like(self.bias))

    def forward(self, x):
        if self.bias is not None:
            b = self.bias * self.bias_mask.to(self.bias.dtype)
        else:
            b = None
        return F.linear(x, self.weight, b)


class _RopeEmbedding(nn.Module):
    """Axial 2D RoPE matching DINOv3; ``periods`` buffer shape [head_dim//4]."""

    def __init__(self, periods_len, compute_dtype=torch.float32):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.register_buffer("periods", torch.zeros(periods_len))

    def forward(self, H, W):
        device = self.periods.device
        dd = {"device": device, "dtype": self.compute_dtype}
        periods = self.periods.to(self.compute_dtype)
        coords_h = torch.arange(0.5, H, **dd) / H
        coords_w = torch.arange(0.5, W, **dd) / W
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0
        angles = 2 * math.pi * coords[:, :, None] / periods[None, None, :]
        angles = angles.flatten(1, 2)
        angles = angles.tile(2)
        return (torch.sin(angles), torch.cos(angles))


class _PatchEmbed(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x):
        return self.proj(x)


class _MetaAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = _MaskedBiasLinear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if rope is not None:
            sin, cos = rope
            q_dtype, k_dtype = q.dtype, k.dtype
            rd = sin.dtype
            prefix = N - sin.shape[-2]
            q, k = q.to(rd), k.to(rd)
            if prefix > 0:
                q = torch.cat([q[:, :, :prefix, :], _rope_apply(q[:, :, prefix:, :], sin, cos)], dim=-2)
                k = torch.cat([k[:, :, :prefix, :], _rope_apply(k[:, :, prefix:, :], sin, cos)], dim=-2)
            else:
                q, k = _rope_apply(q, sin, cos), _rope_apply(k, sin, cos)
            q, k = q.to(q_dtype), k.to(k_dtype)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _MetaMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _MetaBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _MetaAttention(dim, num_heads)
        self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _MetaMlp(dim, int(dim * mlp_ratio))
        self.ls2 = _LayerScale(dim)

    def forward(self, x, rope=None):
        x = x + self.ls1(self.attn(self.norm1(x), rope))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class _MetaViT(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, patch_size, n_storage_tokens,
                 mlp_ratio=4.0, rope_dtype=torch.float32):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_storage_tokens = n_storage_tokens
        self.num_heads = num_heads
        self.patch_embed = _PatchEmbed(patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, n_storage_tokens, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.rope_embed = _RopeEmbedding((embed_dim // num_heads) // 4, compute_dtype=rope_dtype)
        self.blocks = nn.ModuleList([_MetaBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def _prep(self, x):
        x = self.patch_embed(x)
        B, C, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        st = self.storage_tokens.expand(B, -1, -1)
        x = torch.cat([cls, st, x], dim=1)
        return x, self.rope_embed(Hp, Wp)

    def forward_intermediate(self, x, layers):
        x, rope = self._prep(x)
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, rope)
            if i in layers:
                outs.append(x[:, 1 + self.n_storage_tokens:, :])
        return outs

    def forward_patch_tokens(self, x):
        x, rope = self._prep(x)
        for blk in self.blocks:
            x = blk(x, rope)
        x = self.norm(x)
        return x[:, 1 + self.n_storage_tokens:, :]


# --- MAE-style plain ViT (pixio / ijepa), no RoPE / no layer scale --- #
class _SimpleAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _SimpleBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _SimpleAttention(dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _MetaMlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _SimpleViT(nn.Module):
    """MAE-style ViT: learned pos_embed + optional cls tokens, no RoPE / ls."""

    def __init__(self, embed_dim, depth, num_heads, patch_size, n_cls_tokens, num_patches, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_cls_tokens = n_cls_tokens
        self.num_patches = num_patches
        self.patch_embed = _PatchEmbed(patch_size, embed_dim)
        if n_cls_tokens > 0:
            self.cls_token = nn.Parameter(torch.zeros(1, n_cls_tokens, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + n_cls_tokens, embed_dim))
        self.blocks = nn.ModuleList([_SimpleBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def _interp_pos_embed(self, Hp, Wp):
        if Hp * Wp == self.num_patches:
            return self.pos_embed
        n = self.n_cls_tokens
        patch_pe = self.pos_embed[:, n:]
        pt = int(patch_pe.shape[1] ** 0.5)
        patch_pe = patch_pe.reshape(1, pt, pt, -1).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(patch_pe, size=(Hp, Wp), mode="bicubic", align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, Hp * Wp, -1)
        return torch.cat([self.pos_embed[:, :n], patch_pe], dim=1)

    def forward_patch_tokens(self, x):
        x = self.patch_embed(x)
        B, C, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)
        pos = self._interp_pos_embed(Hp, Wp)
        if self.n_cls_tokens > 0:
            cls = (self.cls_token + pos[:, : self.n_cls_tokens, :]).expand(B, -1, -1)
            x = x + pos[:, self.n_cls_tokens:, :]
            x = torch.cat([cls, x], dim=1)
        else:
            x = x + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.n_cls_tokens > 0:
            return x[:, self.n_cls_tokens:, :]
        return x

    def forward_intermediate(self, x, layers):
        x = self.patch_embed(x)
        B, C, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)
        pos = self._interp_pos_embed(Hp, Wp)
        if self.n_cls_tokens > 0:
            cls = (self.cls_token + pos[:, : self.n_cls_tokens, :]).expand(B, -1, -1)
            x = x + pos[:, self.n_cls_tokens:, :]
            x = torch.cat([cls, x], dim=1)
        else:
            x = x + pos
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in layers:
                if self.n_cls_tokens > 0:
                    outs.append(x[:, self.n_cls_tokens:, :])
                else:
                    outs.append(x)
        return outs


# --- ConvNeXt (eupe_convnext_* / dinov3_convnext_*) --- #
class _ConvLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class _ConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = _ConvLayerNorm(dim, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        inp = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.pwconv2(x)
        x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        return inp + x


class _ConvNeXt(nn.Module):
    def __init__(self, dims, depths, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(OrderedDict([
            ("0", nn.Conv2d(3, dims[0], kernel_size=4, stride=4)),
            ("1", _ConvLayerNorm(dims[0], data_format="channels_first")),
        ])))
        for i in range(1, len(dims)):
            self.downsample_layers.append(nn.Sequential(OrderedDict([
                ("0", _ConvLayerNorm(dims[i - 1], data_format="channels_first")),
                ("1", nn.Conv2d(dims[i - 1], dims[i], kernel_size=2, stride=2)),
            ])))
        self.stages = nn.ModuleList([
            nn.Sequential(*[_ConvBlock(dims[i]) for _ in range(depths[i])]) for i in range(len(dims))
        ])
        self.norm = _ConvLayerNorm(dims[-1], data_format="channels_last")

    def forward_patch_tokens(self, x):
        B, _, H, W = x.shape
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = F.interpolate(x, size=(H // self.patch_size, W // self.patch_size),
                          mode="bilinear", align_corners=False, antialias=True)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)

    def forward_intermediate(self, x, layers):
        """Stage-level features: layers are stage indices (0..n_stages-1)."""
        B, _, H, W = x.shape
        outs = []
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in layers:
                f = F.interpolate(x, size=(H // self.patch_size, W // self.patch_size),
                                  mode="bilinear", align_corners=False, antialias=True)
                f = f.flatten(2).transpose(1, 2)
                outs.append(f)
        return outs


# --- Perception Encoder (open_clip-style ViT + attentional pool) --- #
class _PEResBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, has_ls=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ls_1 = _LayerScale(dim) if has_ls else nn.Identity()
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(dim, int(dim * mlp_ratio))),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(int(dim * mlp_ratio), dim)),
        ]))
        self.ls_2 = _LayerScale(dim) if has_ls else nn.Identity()

    def forward(self, x):
        x = x + self.ls_1(self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0])
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class _PEAttnPool(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.layernorm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(dim, int(dim * mlp_ratio))),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(int(dim * mlp_ratio), dim)),
        ]))

    def forward(self, x):
        B = x.shape[0]
        q = self.probe.expand(B, -1, -1)
        return q + self.mlp(self.layernorm(q + self.attn(q, x, x, need_weights=False)[0]))


class _PEViT(nn.Module):
    def __init__(self, width, depth, num_heads, patch_size, num_patches, has_cls=True,
                 has_ls=False, has_post=True, has_attn_pool=False, proj_out=None,
                 mlp_ratio=4.0, attn_pool_mlp_ratio=4.0):
        super().__init__()
        self.width = width
        self.has_cls = has_cls
        self.has_post = has_post
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        if has_cls:
            self.class_embedding = nn.Parameter(torch.randn(width) * (width ** -0.5))
        n_pos = num_patches + (1 if has_cls else 0)
        self.positional_embedding = nn.Parameter(torch.randn(n_pos, width) * (width ** -0.5))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList(
            [_PEResBlock(width, num_heads, mlp_ratio=mlp_ratio, has_ls=has_ls) for _ in range(depth)]
        )
        if has_post:
            self.ln_post = nn.LayerNorm(width)
        if has_attn_pool:
            self.attn_pool = _PEAttnPool(width, mlp_ratio=attn_pool_mlp_ratio)
        if proj_out is not None:
            self.proj = nn.Parameter(torch.randn(width, proj_out) * (width ** -0.5))

    def forward_patch_tokens(self, x):
        x = self.conv1(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, width]
        if self.has_cls:
            cls = self.class_embedding.to(x.dtype).unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        for blk in self.transformer.resblocks:
            x = blk(x)
        if self.has_post:
            x = self.ln_post(x)
        if self.has_cls:
            return x[:, 1:, :]
        return x

    def forward_intermediate(self, x, layers):
        x = self.conv1(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, width]
        if self.has_cls:
            cls = self.class_embedding.to(x.dtype).unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        outs = []
        for i, blk in enumerate(self.transformer.resblocks):
            x = blk(x)
            if i in layers:
                if self.has_cls:
                    outs.append(x[:, 1:, :])
                else:
                    outs.append(x)
        return outs


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _report_load(res, name):
    missing = res.missing_keys
    unexpected = res.unexpected_keys
    if missing or unexpected:
        print(f"    [load {name}] missing={len(missing)} unexpected={len(unexpected)}"
              + (f" e.g. {unexpected[:3]}" if unexpected else ""))


def load_hf(config, device="cuda", multi_layer=False):
    from transformers import AutoModel, AutoImageProcessor
    path = config.get("model_name_or_path") or config.get("vision_tower")
    model = AutoModel.from_pretrained(path).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(path)
    select_layer = config.get("select_layer", -1)
    select_feature = config.get("select_feature", "patch")
    n_reg = getattr(model.config, "num_register_tokens", 0)

    def preprocess(img):
        return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

    class W(_FeatureEncoder):
        @torch.no_grad()
        def encode_tensors(self, tensors, normalize=True):
            tensors = tensors.to(self.device)
            out = self.model(tensors, output_hidden_states=True)
            if self.layers is not None:
                res = {}
                for idx in self.layers:
                    hs = out.hidden_states[idx]
                    if select_feature == "cls":
                        f = hs[:, 0, :]
                    else:
                        f = hs[:, 1 + n_reg:, :].mean(dim=1)
                    res[idx] = _pool_and_norm(f, normalize)
                return res
            hs = out.hidden_states[select_layer]
            if select_feature == "cls":
                feat = hs[:, 0, :]
            else:
                feat = hs[:, 1 + n_reg:, :].mean(dim=1)
            return _pool_and_norm(feat, normalize)

    return _enable_multilayer(W(model, preprocess, model.config.hidden_size, device),
                              model, default_layer=select_layer, multi_layer=multi_layer)


def _infer_meta_vit(sd):
    embed = sd["cls_token"].shape[-1]
    n_storage = sd["storage_tokens"].shape[1]
    patch = sd["patch_embed.proj.weight"].shape[2]
    depth = max(int(re.match(r"blocks\.(\d+)\.", k).group(1)) for k in sd if k.startswith("blocks.")) + 1
    periods = sd["rope_embed.periods"]
    num_heads = embed // (periods.shape[0] * 4)
    mlp_ratio = sd["blocks.0.mlp.fc1.weight"].shape[0] / embed
    return embed, depth, num_heads, patch, n_storage, mlp_ratio, periods.dtype


def load_meta_vit(config, tok_id="", device="cuda", multi_layer=False):
    path = config.get("weights_path")
    sd = _load_state_dict(path)
    sd = {k: v for k, v in sd.items() if not k.startswith("projectors.")}
    embed, depth, heads, patch, n_storage, mlp_ratio, rope_dtype = _infer_meta_vit(sd)
    model = _MetaViT(embed, depth, heads, patch, n_storage, mlp_ratio, rope_dtype)
    _report_load(model.load_state_dict(sd, strict=False), tok_id)
    image_size = config.get("image_size", 224)
    preprocess = _make_preprocess(image_size)

    if config.get("type") == "raev2":
        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(path)), "raev2", "dinov3l_k7_stats.pt")
        stats = torch.load(stats_path, map_location="cpu", weights_only=False) if os.path.exists(stats_path) else None
        layers = list(config.get("layers") or [])

        class W(_FeatureEncoder):
            @torch.no_grad()
            def encode_tensors(self, tensors, normalize=True):
                tensors = tensors.to(self.device)
                outs = self.model.forward_intermediate(tensors, layers)
                feat = sum(outs).float()
                if stats is not None:
                    C, N = feat.shape[-1], feat.shape[1]
                    m = stats["mean"].reshape(C, N).t().reshape(1, N, C).to(self.device)
                    v = stats["var"].reshape(C, N).t().reshape(1, N, C).to(self.device)
                    feat = (feat - m) / torch.sqrt(v + 1e-6)
                return _pool_and_norm(feat, normalize)
        return W(model.to(device).eval(), preprocess, embed, device)

    class W(_FeatureEncoder):
        @torch.no_grad()
        def encode_tensors(self, tensors, normalize=True):
            tensors = tensors.to(self.device)
            if self.layers is not None:
                outs = self.model.forward_intermediate(tensors, self.layers)
                return {idx: _pool_and_norm(f, normalize)
                        for idx, f in zip(self.layers, outs)}
            feat = self.model.forward_patch_tokens(tensors)
            return _pool_and_norm(feat, normalize)
    return _enable_multilayer(W(model.to(device).eval(), preprocess, embed, device),
                              model, default_layer=config.get("select_layer", -1),
                              multi_layer=multi_layer)


_pixio_heads = {
    "pixio_vitb16": 12, "pixio_vitl16": 16, "pixio_vith16": 16,
    "pixio_vit1b16": 24, "pixio_vit5b16": 32,
}


def load_simple_vit(config, tok_id="", device="cuda", multi_layer=False):
    path = config.get("weights_path")
    sd = _load_state_dict(path)
    if "cls_token" in sd:
        embed = sd["cls_token"].shape[-1]
        n_cls = sd["cls_token"].shape[1]
        num_patches = sd["pos_embed"].shape[1] - n_cls
        patch = sd["patch_embed.proj.weight"].shape[2]
        depth = max(int(re.match(r"blocks\.(\d+)\.", k).group(1)) for k in sd if k.startswith("blocks.")) + 1
        mlp_ratio = sd["blocks.0.mlp.fc1.weight"].shape[0] / embed
        heads = _pixio_heads.get(config.get("pixio_hub", ""), embed // 64)
    else:
        embed = sd["pos_embed"].shape[-1]
        n_cls = 0
        num_patches = sd["pos_embed"].shape[1]
        patch = sd["patch_embed.proj.weight"].shape[2]
        depth = max(int(re.match(r"blocks\.(\d+)\.", k).group(1)) for k in sd if k.startswith("blocks.")) + 1
        mlp_ratio = sd["blocks.0.mlp.fc1.weight"].shape[0] / embed
        heads = {"ijepa_vith14": 16}.get(tok_id, embed // 64)
    model = _SimpleViT(embed, depth, heads, patch, n_cls, num_patches, mlp_ratio)
    _report_load(model.load_state_dict(sd, strict=False), tok_id)
    image_size = config.get("image_size", 224)
    preprocess = _make_preprocess(image_size)

    class W(_FeatureEncoder):
        @torch.no_grad()
        def encode_tensors(self, tensors, normalize=True):
            tensors = tensors.to(self.device)
            if self.layers is not None:
                outs = self.model.forward_intermediate(tensors, self.layers)
                return {idx: _pool_and_norm(f, normalize)
                        for idx, f in zip(self.layers, outs)}
            feat = self.model.forward_patch_tokens(tensors)
            return _pool_and_norm(feat, normalize)
    return _enable_multilayer(W(model.to(device).eval(), preprocess, embed, device),
                              model, default_layer=config.get("select_layer", -1),
                              multi_layer=multi_layer)


def load_convnext(config, tok_id="", device="cuda", multi_layer=False):
    path = config.get("weights_path")
    sd = _load_state_dict(path)
    sd = {k: v for k, v in sd.items()
          if not k.startswith("projectors.") and not k.startswith("norms.")}
    dims = [sd[f"downsample_layers.{i}.1.weight"].shape[0] for i in range(4)]
    depths = []
    for i in range(4):
        depths.append(max(int(re.match(r"stages\.(\d+)\.(\d+)\.", k).group(2))
                          for k in sd if k.startswith(f"stages.{i}.")) + 1)
    patch_size = config.get("patch_size", 16)
    model = _ConvNeXt(dims, depths, patch_size)
    _report_load(model.load_state_dict(sd, strict=False), tok_id)
    image_size = config.get("image_size", 224)
    preprocess = _make_preprocess(image_size)

    class W(_FeatureEncoder):
        @torch.no_grad()
        def encode_tensors(self, tensors, normalize=True):
            tensors = tensors.to(self.device)
            if self.layers is not None:
                outs = self.model.forward_intermediate(tensors, self.layers)
                return {idx: _pool_and_norm(f, normalize)
                        for idx, f in zip(self.layers, outs)}
            feat = self.model.forward_patch_tokens(tensors)
            return _pool_and_norm(feat, normalize)
    return _enable_multilayer(W(model.to(device).eval(), preprocess, dims[-1], device),
                              model, default_layer=config.get("select_layer", -1),
                              multi_layer=multi_layer)


def load_pe(config, tok_id="", device="cuda", multi_layer=False):
    path = config.get("weights_path")
    sd = _load_state_dict(path)
    prefix = "visual." if any(k.startswith("visual.") for k in sd) else ""
    trunk = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)} if prefix else sd
    width = trunk["conv1.weight"].shape[0]
    patch = trunk["conv1.weight"].shape[2]
    has_cls = "class_embedding" in trunk
    depth = max(int(re.match(r"transformer\.resblocks\.(\d+)\.", k).group(1))
                for k in trunk if k.startswith("transformer.resblocks.")) + 1
    has_ls = "transformer.resblocks.0.ls_1.gamma" in trunk
    has_post = "ln_post.weight" in trunk
    has_attn_pool = "attn_pool.probe" in trunk
    proj_out = trunk["proj"].shape[1] if "proj" in trunk else None
    n_pos = trunk["positional_embedding"].shape[0]
    num_patches = n_pos - (1 if has_cls else 0)
    heads = width // 64
    mlp_ratio = trunk["transformer.resblocks.0.mlp.c_fc.weight"].shape[0] / width
    attn_pool_mlp_ratio = 4.0
    if "attn_pool.mlp.c_fc.weight" in trunk:
        attn_pool_mlp_ratio = trunk["attn_pool.mlp.c_fc.weight"].shape[0] / width
    model = _PEViT(width, depth, heads, patch, num_patches,
                   has_cls=has_cls, has_ls=has_ls, has_post=has_post,
                   has_attn_pool=has_attn_pool, proj_out=proj_out,
                   mlp_ratio=mlp_ratio, attn_pool_mlp_ratio=attn_pool_mlp_ratio)
    _report_load(model.load_state_dict(trunk, strict=False), tok_id)
    image_size = config.get("image_size", 224)
    preprocess = _make_preprocess(image_size, mean=CLIP_MEAN, std=CLIP_STD)

    class W(_FeatureEncoder):
        @torch.no_grad()
        def encode_tensors(self, tensors, normalize=True):
            tensors = tensors.to(self.device)
            if self.layers is not None:
                outs = self.model.forward_intermediate(tensors, self.layers)
                return {idx: _pool_and_norm(f, normalize)
                        for idx, f in zip(self.layers, outs)}
            feat = self.model.forward_patch_tokens(tensors)
            return _pool_and_norm(feat, normalize)
    return _enable_multilayer(W(model.to(device).eval(), preprocess, width, device),
                              model, default_layer=config.get("select_layer", -1),
                              multi_layer=multi_layer)


def load(config, tok_id, device="cuda", multi_layer=False):
    enc_type = config.get("type", "")
    if enc_type == "hf":
        return load_hf(config, device, multi_layer=multi_layer)
    if enc_type == "pe":
        return load_pe(config, tok_id, device, multi_layer=multi_layer)
    if enc_type in ("pixio", "ijepa"):
        return load_simple_vit(config, tok_id, device, multi_layer=multi_layer)
    if enc_type in ("dinov3", "eupe", "raev2"):
        if _is_convnext(config, tok_id):
            return load_convnext(config, tok_id, device, multi_layer=multi_layer)
        return load_meta_vit(config, tok_id, device, multi_layer=multi_layer)
    raise ValueError(f"unknown type: {enc_type}")


def _is_convnext(config, tok_id):
    for k in ("model_name", "eupe_hub", "dinov3_backbone", "id"):
        if "convnext" in str(config.get(k, "")).lower():
            return True
    return "convnext" in tok_id.lower()
