"""Minimal flash_attn API shim backed by PyTorch SDPA (encode-only fallback)."""

from __future__ import annotations

import sys
import types

import torch
import torch.nn.functional as F


def _sdpa(q, k, v, causal: bool = False):
    # q/k/v: [B, H, L, D] or [1, H, L, D]
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    del dropout_p, softmax_scale, kwargs
    # qkv: [B, L, 3, H, D]
    q, k, v = qkv.unbind(dim=2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = _sdpa(q, k, v, causal=causal)
    return out.transpose(1, 2)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    del dropout_p, softmax_scale, kwargs
    # qkv: [T, 3, H, D]; process each sequence independently.
    outs = []
    for i in range(len(cu_seqlens) - 1):
        s = int(cu_seqlens[i].item())
        e = int(cu_seqlens[i + 1].item())
        if e <= s:
            continue
        chunk = qkv[s:e]  # [L, 3, H, D]
        q, k, v = chunk.unbind(dim=1)
        q = q.unsqueeze(0).transpose(1, 2)  # [1, H, L, D]
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)
        out = _sdpa(q, k, v, causal=causal).transpose(1, 2).squeeze(0)  # [L, H, D]
        outs.append(out)
    if not outs:
        return qkv.new_zeros(qkv.shape[0], qkv.shape[2], qkv.shape[3])
    return torch.cat(outs, dim=0)


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    del max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, kwargs
    outs = []
    for i in range(len(cu_seqlens_q) - 1):
        qs, qe = int(cu_seqlens_q[i].item()), int(cu_seqlens_q[i + 1].item())
        ks, ke = int(cu_seqlens_k[i].item()), int(cu_seqlens_k[i + 1].item())
        qi = q[qs:qe].unsqueeze(0).transpose(1, 2)
        k, v = kv[ks:ke].unbind(dim=1)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)
        out = _sdpa(qi, k, v, causal=causal).transpose(1, 2).squeeze(0)
        outs.append(out)
    return torch.cat(outs, dim=0) if outs else q.new_zeros(q.shape[0], q.shape[1], q.shape[2])


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    del max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, kwargs
    outs = []
    for i in range(len(cu_seqlens_q) - 1):
        qs, qe = int(cu_seqlens_q[i].item()), int(cu_seqlens_q[i + 1].item())
        ks, ke = int(cu_seqlens_k[i].item()), int(cu_seqlens_k[i + 1].item())
        qi = q[qs:qe].unsqueeze(0).transpose(1, 2)
        ki = k[ks:ke].unsqueeze(0).transpose(1, 2)
        vi = v[ks:ke].unsqueeze(0).transpose(1, 2)
        out = _sdpa(qi, ki, vi, causal=causal).transpose(1, 2).squeeze(0)
        outs.append(out)
    return torch.cat(outs, dim=0) if outs else q.new_zeros(q.shape[0], q.shape[1], q.shape[2])


def install_flash_attn_fallback() -> None:
    """Register a fake ``flash_attn`` module if the real package is unavailable."""
    existing = sys.modules.get("flash_attn")
    if existing is not None and getattr(existing, "__version__", "").endswith("+vtb_sdpa_fallback"):
        return
    try:
        import flash_attn  # noqa: F401

        if getattr(flash_attn, "__version__", "").endswith("+vtb_sdpa_fallback"):
            pass
        else:
            return
    except Exception:
        pass

    import importlib.machinery

    mod = types.ModuleType("flash_attn")
    mod.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func
    mod.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
    mod.flash_attn_varlen_kvpacked_func = flash_attn_varlen_kvpacked_func
    mod.flash_attn_varlen_func = flash_attn_varlen_func
    mod.__version__ = "0.0.0+vtb_sdpa_fallback"
    mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    mod.__file__ = __file__
    sys.modules["flash_attn"] = mod
