"""AToken-So/D discrete tokenizer (FSQ post-quant features → projector)."""

from __future__ import annotations

import os
import sys
import types
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from src.discrete.model.tokenizers.base import BaseDiscreteTokenizer

_DEFAULT_ATOKEN_REPO = "/cache/repos/ml-atoken"


def _ensure_atoken_repo(repo_root: str | None = None) -> str:
    root = os.path.abspath(repo_root or os.environ.get("ATOKEN_REPO", _DEFAULT_ATOKEN_REPO))
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"AToken repo not found at {root}. Clone apple/ml-atoken or set ATOKEN_REPO."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _install_decoder_gs_stub() -> None:
    """Provide a stub so encode-only use does not require the GS decoder stack."""
    mod_name = "atoken_inference.model.decoder_gs"
    existing = sys.modules.get(mod_name)
    if existing is not None and not getattr(existing, "_vtb_stub", False):
        return

    stub = types.ModuleType(mod_name)
    stub._vtb_stub = True  # type: ignore[attr-defined]

    class SLatGaussianDecoderTrain:  # pragma: no cover - encode path never constructs this
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "AToken GS decoder was requested but is stubbed. "
                "Install the full ml-atoken 3D decoder stack to use inference_3d."
            )

    stub.SLatGaussianDecoderTrain = SLatGaussianDecoderTrain  # type: ignore[attr-defined]
    sys.modules[mod_name] = stub


def _import_atoken_wrapper():
    from src.discrete.model.tokenizers.atoken.flash_attn_fallback import (
        install_flash_attn_fallback,
    )

    install_flash_attn_fallback()
    _install_decoder_gs_stub()
    from atoken_inference.atoken_wrapper import ATokenWrapper

    return ATokenWrapper


class ATokenTokenizer(BaseDiscreteTokenizer):
    """Wrap Apple AToken; MLLM path uses FSQ-quantized sparse features padded to fixed T."""

    def __init__(
        self,
        wrapper: Any,
        *,
        codebook_size: int,
        embed_dim: int,
        post_quant_embed_dim: int,
        image_size: int,
        num_tokens: int,
        fp16: bool = True,
    ):
        super().__init__()
        self._wrapper = wrapper
        self._wrapper.eval()
        self._wrapper.requires_grad_(False)
        self._vocab_size = codebook_size
        self._embed_dim = embed_dim
        self._post_quant_embed_dim = post_quant_embed_dim
        self._image_size = image_size
        self._num_tokens = num_tokens
        self._fp16 = fp16
        self.add_module("_atoken", wrapper)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def post_quant_embed_dim(self) -> int:
        return self._post_quant_embed_dim

    @property
    def num_image_tokens(self) -> int:
        return self._num_tokens

    @property
    def image_size(self) -> int:
        return self._image_size

    def preprocess(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.shape[-1] != self._image_size or pixel_values.shape[-2] != self._image_size:
            pixel_values = F.interpolate(
                pixel_values,
                size=(self._image_size, self._image_size),
                mode="bicubic",
                align_corners=False,
            )
        if pixel_values.max() <= 1.0:
            pixel_values = pixel_values * 2.0 - 1.0
        return pixel_values

    def _quantize_sparse(self, z) -> torch.Tensor:
        model = self._wrapper.model
        feat_dim = int(model.quantizer_feature_dim)
        z_feat = z.feats[:, :feat_dim]
        chunk_dim = feat_dim // model.quantizer_chunk_size
        n = z_feat.shape[0]
        if model.quantizer_chunk_size > 1:
            z_feat = z_feat.reshape(n * model.quantizer_chunk_size, chunk_dim)
        quantized, _indices, _loss = model.quantizer(z_feat)
        if model.quantizer_chunk_size > 1:
            quantized = quantized.reshape(n, feat_dim)
        return quantized

    def _encode_quantized_batch(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixels = self.preprocess(pixel_values)
        param = next(self._wrapper.parameters())
        pixels = pixels.to(device=param.device, dtype=param.dtype)

        images = [pixels[i] for i in range(pixels.shape[0])]
        with torch.no_grad():
            x = self._wrapper.image_video_to_sparse_tensor(images)
            z, _image_feat, _x_no_proj = self._wrapper.encode(x, normalize=True)
            if self._wrapper.arch_cfg.get("use_quantizer", False):
                q = self._quantize_sparse(z)
            else:
                q = z.feats[:, : self._post_quant_embed_dim]

            batch_ids = z.coords[:, 0].long()
            batch_size = pixels.shape[0]
            out = torch.zeros(
                batch_size,
                self._num_tokens,
                self._post_quant_embed_dim,
                device=pixels.device,
                dtype=pixel_values.dtype,
            )
            for b in range(batch_size):
                fb = q[batch_ids == b]
                if fb.shape[-1] != self._post_quant_embed_dim:
                    fb = fb[..., : self._post_quant_embed_dim]
                t = min(int(fb.shape[0]), self._num_tokens)
                if t > 0:
                    out[b, :t] = fb[:t].to(dtype=out.dtype)
        return out

    def encode(self, pixel_values: torch.Tensor) -> torch.LongTensor:
        feats = self.encode_post_quant_features(pixel_values)
        return (feats > 0).to(torch.long).sum(dim=-1)

    def encode_post_quant_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return FSQ features [B, num_tokens, post_quant_embed_dim], pad/trunc to fixed T."""
        return self._encode_quantized_batch(pixel_values)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        config_path: str | None = None,
        repo_root: str | None = None,
        codebook_size: int = 4096,
        embed_dim: int = 48,
        post_quant_embed_dim: int = 48,
        image_size: int = 256,
        num_tokens: int = 256,
        fp16: bool = True,
    ) -> "ATokenTokenizer":
        root = _ensure_atoken_repo(repo_root)
        checkpoint_path = os.path.abspath(checkpoint_path)
        if config_path is None:
            sibling = os.path.join(os.path.dirname(checkpoint_path), "atoken-sod.yaml")
            repo_cfg = os.path.join(root, "configs", "atoken-sod.yaml")
            config_path = sibling if os.path.isfile(sibling) else repo_cfg
        config_path = os.path.abspath(config_path)

        ATokenWrapper = _import_atoken_wrapper()

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        arch = ((cfg.get("model") or {}).get("model_cfg") or {}).get("arch_cfg") or {}
        resolved_dim = int(post_quant_embed_dim or arch.get("quantizer_feature_dim", 48))
        resolved_vocab = int(codebook_size or arch.get("quantizer_codebook_size", 4096))

        wrapper = ATokenWrapper(config_path, checkpoint_path)
        dtype = torch.bfloat16 if fp16 else torch.float32
        wrapper = wrapper.to(dtype=dtype)
        wrapper.eval()

        print(
            f"Loaded AToken tokenizer: codebook={resolved_vocab}, "
            f"post_quant_dim={resolved_dim}, image={image_size}, tokens={num_tokens}",
            flush=True,
        )
        return cls(
            wrapper,
            codebook_size=resolved_vocab,
            embed_dim=int(embed_dim or resolved_dim),
            post_quant_embed_dim=resolved_dim,
            image_size=image_size,
            num_tokens=num_tokens,
            fp16=fp16,
        )
