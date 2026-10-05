"""VILA-U RQVAESIGLIP discrete visual tokenizer wrapper for VTB-Discrete MLLM."""

from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from vision_encoder_eval.mllm.discrete.model.tokenizers.base import BaseDiscreteTokenizer
from vision_encoder_eval.mllm.discrete.model.tokenizers.vilau.rqvaesigliptransformer import (
    RQVAESIGLIPTransformer,
    RQVAESIGLIPTransformerConfig,
)

# SigLIP / VILA-U vision tower normalization.
_VILAU_MEAN = (0.5, 0.5, 0.5)
_VILAU_STD = (0.5, 0.5, 0.5)


class VilaUTokenizer(BaseDiscreteTokenizer):
    """Wrap VILA-U RQVAESIGLIP vision tower (@256 or @384, auto-resolved from checkpoint)."""

    def __init__(
        self,
        model: RQVAESIGLIPTransformer,
        *,
        codebook_size: int,
        embed_dim: int,
        post_quant_embed_dim: int,
        image_size: int,
        num_tokens: int,
        rq_depth: int,
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
        self._rq_depth = rq_depth
        self._fp16 = fp16
        self._processor = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=_VILAU_MEAN, std=_VILAU_STD),
        ])

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
    def rq_depth(self) -> int:
        return self._rq_depth

    @property
    def num_image_tokens(self) -> int:
        return self._num_tokens

    @property
    def image_size(self) -> int:
        return self._image_size

    def codebook_weight(self) -> torch.Tensor:
        """Shared RQ codebook weights: [n_embed, embed_dim]."""
        quantizer = self._model.rqvaesiglip.quantizer
        return quantizer.codebooks[0].weight[:-1, :].detach().clone()

    def preprocess(self, pixel_values: torch.Tensor | list | Image.Image) -> torch.Tensor:
        if isinstance(pixel_values, torch.Tensor):
            if pixel_values.shape[-1] != self._image_size or pixel_values.shape[-2] != self._image_size:
                pixel_values = torch.nn.functional.interpolate(
                    pixel_values,
                    size=(self._image_size, self._image_size),
                    mode="bicubic",
                    align_corners=False,
                )
            if pixel_values.max() <= 1.0 and pixel_values.min() >= 0.0:
                mean = torch.tensor(
                    _VILAU_MEAN, device=pixel_values.device, dtype=pixel_values.dtype
                ).view(1, 3, 1, 1)
                std = torch.tensor(
                    _VILAU_STD, device=pixel_values.device, dtype=pixel_values.dtype
                ).view(1, 3, 1, 1)
                pixel_values = (pixel_values - mean) / std
            return pixel_values

        if not isinstance(pixel_values, list):
            pixel_values = [pixel_values]

        tensors = []
        for img in pixel_values:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert("RGB")
            tensors.append(self._processor(img))
        return torch.stack(tensors, dim=0)

    def _encode_raw(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = self.preprocess(pixel_values)
        param = next(self._model.parameters())
        pixels = pixels.to(device=param.device, dtype=param.dtype)
        with torch.no_grad():
            codes, z_q = self._model.rqvaesiglip.encode_image(pixels)
        return codes, z_q

    def encode(self, pixel_values: torch.Tensor) -> torch.LongTensor:
        """Return residual code indices [B, num_tokens, rq_depth] (debug / inspection only)."""
        codes, _ = self._encode_raw(pixel_values)
        batch_size = codes.shape[0]
        return codes.reshape(batch_size, self._num_tokens, self._rq_depth).long()

    def encode_post_quant_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return RQ-quantized patch features [B, num_tokens, post_quant_embed_dim]."""
        _, z_q = self._encode_raw(pixel_values)
        batch_size = z_q.shape[0]
        feats = z_q.reshape(batch_size, self._num_tokens, self._post_quant_embed_dim)
        return feats.to(pixel_values.dtype)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        codebook_size: int = 16384,
        embed_dim: int = 1024,
        post_quant_embed_dim: int = 1024,
        image_size: int = 256,
        num_tokens: int = 256,
        rq_depth: int = 4,
        fp16: bool = True,
    ) -> "VilaUTokenizer":
        dtype = torch.bfloat16 if fp16 else torch.float32
        config = RQVAESIGLIPTransformerConfig.from_pretrained(checkpoint_path)
        model = RQVAESIGLIPTransformer.from_pretrained(
            checkpoint_path,
            config=config,
            torch_dtype=dtype,
        )
        if fp16:
            model = model.to(dtype=dtype)
        model.eval()

        rq_cfg = config.rqvaesiglip or {}
        resolved_codebook = int(codebook_size or rq_cfg.get("n_embed", 16384))
        resolved_embed = int(embed_dim or rq_cfg.get("embed_dim", 1024))
        resolved_post_q = int(post_quant_embed_dim or config.hidden_size or resolved_embed)
        code_shape = rq_cfg.get("code_shape") or [16, 16, rq_depth]
        resolved_depth = int(rq_depth or code_shape[-1])
        spatial = int(code_shape[0]) * int(code_shape[1])
        resolved_tokens = int(num_tokens or spatial)
        resolved_image = int(image_size or rq_cfg.get("ddconfig", {}).get("resolution", 256))

        print(
            f"Loaded VILA-U tokenizer: codebook={resolved_codebook}, embed_dim={resolved_embed}, "
            f"post_quant_dim={resolved_post_q}, rq_depth={resolved_depth}, "
            f"image={resolved_image}, tokens={resolved_tokens}",
            flush=True,
        )
        return cls(
            model,
            codebook_size=resolved_codebook,
            embed_dim=resolved_embed,
            post_quant_embed_dim=resolved_post_q,
            image_size=resolved_image,
            num_tokens=resolved_tokens,
            rq_depth=resolved_depth,
            fp16=fp16,
        )
