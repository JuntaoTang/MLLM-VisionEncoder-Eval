"""BitDance / UniWeTok LFQ discrete tokenizer (post-quant ±1 features → projector)."""

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file as load_safetensors

from src.discrete.model.tokenizers.base import BaseDiscreteTokenizer
from src.discrete.model.tokenizers.bitdance.autoencoder import VQModel


class BitDanceTokenizer(BaseDiscreteTokenizer):
    """Wrap BitDance / UniWeTok VQ autoencoder; MLLM uses LFQ ±1 features."""

    def __init__(
        self,
        model: VQModel,
        *,
        codebook_size: int,
        embed_dim: int,
        post_quant_embed_dim: int,
        image_size: int,
        num_tokens: int,
        downsample: int,
        fp16: bool = True,
    ):
        super().__init__()
        self._model = model
        self._model.eval()
        self._model.requires_grad_(False)
        self._vocab_size = codebook_size
        self._embed_dim = embed_dim
        self._post_quant_embed_dim = post_quant_embed_dim
        self._image_size = image_size
        self._num_tokens = num_tokens
        self._downsample = downsample
        self._fp16 = fp16

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

    def encode(self, pixel_values: torch.Tensor) -> torch.LongTensor:
        """Pack LFQ signs into integer codes for inspection (not used by projector path)."""
        feats = self.encode_post_quant_features(pixel_values)
        bits = (feats > 0).to(torch.long)
        # Treat each token's bit vector as an opaque code; clamp to int64 range via hash-like fold.
        # Projector path uses encode_post_quant_features instead.
        codes = torch.zeros(bits.shape[:2], dtype=torch.long, device=bits.device)
        for i in range(bits.shape[-1]):
            codes = (codes * 2 + bits[..., i]) % (2**31 - 1)
        return codes

    def encode_post_quant_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return LFQ ±1 features [B, num_tokens, post_quant_embed_dim]."""
        pixels = self.preprocess(pixel_values)
        param = next(self._model.parameters())
        pixels = pixels.to(device=param.device, dtype=param.dtype)
        with torch.no_grad():
            quant = self._model.encode(pixels)  # [B, C, h, w]
            feats = rearrange(quant, "b c h w -> b (h w) c")
        if feats.shape[1] != self._num_tokens:
            # Safety for unexpected spatial size: interpolate in token space.
            feats = feats.transpose(1, 2)
            feats = F.interpolate(feats, size=self._num_tokens, mode="linear", align_corners=False)
            feats = feats.transpose(1, 2)
        return feats.to(dtype=pixel_values.dtype)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        config_path: str | None = None,
        codebook_size: int | None = None,
        embed_dim: int = 128,
        post_quant_embed_dim: int = 128,
        image_size: int = 256,
        num_tokens: int | None = None,
        fp16: bool = True,
    ) -> "BitDanceTokenizer":
        checkpoint_path = os.path.abspath(checkpoint_path)
        if config_path is None:
            stem = checkpoint_path
            if stem.endswith(".safetensors"):
                stem = stem[: -len(".safetensors")]
            config_path = f"{stem}_config.json"
        config_path = os.path.abspath(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            ae_config = json.load(f)

        ddconfig = ae_config["ddconfig"]
        z_channels = int(ddconfig["z_channels"])
        downsample = 2 ** (len(ddconfig["ch_mult"]) - 1)
        resolved_tokens = int(num_tokens or (image_size // downsample) ** 2)
        resolved_dim = int(post_quant_embed_dim or z_channels)
        resolved_embed = int(embed_dim or z_channels)
        # Binary codebook size 2^{z_channels}; store min(2^z, 2^31-1) for interface.
        resolved_vocab = int(codebook_size) if codebook_size is not None else min(2**z_channels, 2**31 - 1)

        model = VQModel(**ae_config)
        state = load_safetensors(checkpoint_path)
        model.load_state_dict(state, strict=True)
        dtype = torch.bfloat16 if fp16 else torch.float32
        model = model.to(dtype=dtype)
        model.eval()

        print(
            f"Loaded BitDance tokenizer: z_channels={z_channels}, downsample={downsample}, "
            f"post_quant_dim={resolved_dim}, image={image_size}, tokens={resolved_tokens}",
            flush=True,
        )
        return cls(
            model,
            codebook_size=resolved_vocab,
            embed_dim=resolved_embed,
            post_quant_embed_dim=resolved_dim,
            image_size=image_size,
            num_tokens=resolved_tokens,
            downsample=downsample,
            fp16=fp16,
        )
