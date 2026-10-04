"""UniAR BSQ discrete tokenizer (deepstack post-quant features → projector)."""

from __future__ import annotations

from vision_encoder_eval.core.runtime import asset_path

import os
import sys
from typing import Any

import torch
import torch.nn.functional as F

from vision_encoder_eval.mllm.discrete.model.tokenizers.base import BaseDiscreteTokenizer

_DEFAULT_UNIAR_REPO = asset_path('runtime', 'repos/UniAR')


def _ensure_uniar_repo(repo_root: str | None = None) -> str:
    root = os.path.abspath(repo_root or os.environ.get("UNIAR_REPO", _DEFAULT_UNIAR_REPO))
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"UniAR repo not found at {root}. Clone ShareLab-SII/UniAR or set UNIAR_REPO."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


class UniARTokenizer(BaseDiscreteTokenizer):
    """Wrap UniAR BSQ vision encoder; MLLM uses bsq_only deepstack features."""

    def __init__(
        self,
        model: Any,
        *,
        codebook_size: int,
        embed_dim: int,
        post_quant_embed_dim: int,
        image_size: int,
        num_tokens: int,
        bsq_only: bool = True,
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
        self._bsq_only = bsq_only
        self._fp16 = fp16
        self.add_module("_uniar", model)

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
        # Match Qwen2/3-VL processor defaults: rescale to [0,1] then (x - 0.5) / 0.5.
        if pixel_values.min() < 0:
            # Already roughly in [-1, 1].
            return pixel_values
        if pixel_values.max() > 1.0:
            pixel_values = pixel_values / 255.0
        mean = pixel_values.new_tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = pixel_values.new_tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        return (pixel_values.clamp(0.0, 1.0) - mean) / std

    def _forward_bsq(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixels = self.preprocess(pixel_values)
        param = next(self._model.parameters())
        pixels = pixels.to(device=param.device, dtype=param.dtype)
        convert = self._model.convert_img_to_patch
        with torch.no_grad():
            flat, grid_thw = convert(pixels)
            hidden, deepstack = self._model(
                flat,
                grid_thw=grid_thw,
                bsq_only=self._bsq_only,
            )
            # hidden: [B*T, H], deepstack: list of [B*T, H]
            if deepstack:
                feats = torch.cat([hidden] + list(deepstack), dim=-1)
            else:
                feats = hidden
            batch_size = pixels.shape[0]
            # convert stacks patches as (b, t*h*w); reshape to [B, T, D]
            total = feats.shape[0]
            tokens = total // batch_size
            feats = feats.reshape(batch_size, tokens, -1)
            if tokens != self._num_tokens:
                # interpolate token count if resolution/grid differs
                feats = feats.transpose(1, 2)
                feats = F.interpolate(feats, size=self._num_tokens, mode="linear", align_corners=False)
                feats = feats.transpose(1, 2)
        return feats.to(dtype=pixel_values.dtype)

    def encode(self, pixel_values: torch.Tensor) -> torch.LongTensor:
        feats = self.encode_post_quant_features(pixel_values)
        return (feats > 0).to(torch.long).sum(dim=-1)

    def encode_post_quant_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return BSQ deepstack features [B, num_tokens, post_quant_embed_dim]."""
        return self._forward_bsq(pixel_values)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        repo_root: str | None = None,
        codebook_size: int = 2**31 - 1,  # informational; BSQ is binary spherical
        embed_dim: int = 1152,
        post_quant_embed_dim: int = 4608,
        image_size: int = 512,
        num_tokens: int = 1024,
        bsq_only: bool = True,
        fp16: bool = True,
    ) -> "UniARTokenizer":
        _ensure_uniar_repo(repo_root)
        checkpoint_path = os.path.abspath(checkpoint_path)

        # Register UniAR configs with transformers Auto*.
        import uniar  # noqa: F401
        from uniar.vision_encoder import load_bsq_image_tokenizer_and_transform

        # Allow either .../uniar/bsq_encoder or .../uniar (with subfolder).
        model_path = checkpoint_path
        subfolder = None
        if os.path.isdir(os.path.join(checkpoint_path, "bsq_encoder")):
            model_path = checkpoint_path
            subfolder = "bsq_encoder"
        elif os.path.basename(checkpoint_path.rstrip("/")) == "bsq_encoder":
            model_path = checkpoint_path
            subfolder = None

        model = load_bsq_image_tokenizer_and_transform(
            model_path,
            resolution=image_size,
            feature_level=None,
            no_merger=False,
            subfolder=subfolder,
        )
        dtype = torch.bfloat16 if fp16 else torch.float32
        model = model.to(dtype=dtype)
        model.eval()

        resolved_dim = int(getattr(model, "embed_dim", post_quant_embed_dim) or post_quant_embed_dim)
        # Spatial tokens for square image: (H / patch)^2 with convert_img spatial_merge=1.
        patch = int(getattr(model.config, "patch_size", 16))
        resolved_tokens = int(num_tokens or (image_size // patch) ** 2)
        bsq_dim = int(getattr(model.config, "bsq_dim", 64))
        resolved_vocab = min(int(codebook_size), 2**31 - 1)

        print(
            f"Loaded UniAR BSQ tokenizer: bsq_dim={bsq_dim}, "
            f"post_quant_dim={resolved_dim}, image={image_size}, tokens={resolved_tokens}, "
            f"bsq_only={bsq_only}",
            flush=True,
        )
        return cls(
            model,
            codebook_size=resolved_vocab,
            embed_dim=int(embed_dim or model.config.hidden_size),
            post_quant_embed_dim=resolved_dim,
            image_size=image_size,
            num_tokens=resolved_tokens,
            bsq_only=bsq_only,
            fp16=fp16,
        )
