"""TokLIP discrete visual tokenizer wrapper for VTB-Discrete MLLM."""

from __future__ import annotations

import os
import sys
from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from src.discrete.model.tokenizers.base import BaseDiscreteTokenizer

_TOKLIP_ROOT = os.path.dirname(os.path.abspath(__file__))


def _ensure_toklip_import_path() -> None:
    if _TOKLIP_ROOT not in sys.path:
        sys.path.insert(0, _TOKLIP_ROOT)


def _ensure_vqgan_symlink(vqgan_checkpoint: str) -> None:
    vq_dir = os.path.join(_TOKLIP_ROOT, "tokenizer", "pretrained_models")
    os.makedirs(vq_dir, exist_ok=True)
    target = os.path.join(vq_dir, "vq_ds16_t2i.pt")
    vqgan_checkpoint = os.path.abspath(vqgan_checkpoint)
    if os.path.isfile(target):
        return
    if os.path.isfile(vqgan_checkpoint):
        os.symlink(vqgan_checkpoint, target)
        return
    raise FileNotFoundError(f"LlamaGen VQGAN checkpoint not found: {vqgan_checkpoint}")


class TokLIPTokenizer(BaseDiscreteTokenizer):
    """Wrap TokLIP visual trunk; MLLM path uses ViT patch features after VQ."""

    def __init__(
        self,
        visual: nn.Module,
        preprocess: Any,
        *,
        codebook_size: int,
        embed_dim: int,
        post_quant_embed_dim: int,
        image_size: int,
        num_tokens: int,
        fp16: bool = True,
    ):
        super().__init__()
        self._visual = visual
        self._visual.eval()
        self._visual.requires_grad_(False)
        self._preprocess = preprocess
        self._vocab_size = codebook_size
        self._embed_dim = embed_dim
        self._post_quant_embed_dim = post_quant_embed_dim
        self._image_size = image_size
        self._num_tokens = num_tokens
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

    def _to_pil_batch(self, pixel_values: torch.Tensor | list | Image.Image) -> list[Image.Image]:
        if isinstance(pixel_values, Image.Image):
            return [pixel_values.convert("RGB")]
        if isinstance(pixel_values, list):
            images = []
            for item in pixel_values:
                if isinstance(item, Image.Image):
                    images.append(item.convert("RGB"))
                else:
                    images.append(Image.fromarray(item).convert("RGB"))
            return images
        if isinstance(pixel_values, torch.Tensor):
            tensors = pixel_values.detach().cpu()
            if tensors.ndim == 3:
                tensors = tensors.unsqueeze(0)
            images: list[Image.Image] = []
            for t in tensors:
                arr = t.float().permute(1, 2, 0).numpy()
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype("uint8")
                else:
                    arr = arr.clip(0, 255).astype("uint8")
                images.append(Image.fromarray(arr))
            return images
        raise TypeError(f"Unsupported pixel_values type: {type(pixel_values)}")

    def preprocess(self, pixel_values: torch.Tensor | list | Image.Image) -> torch.Tensor:
        images = self._to_pil_batch(pixel_values)
        tensors = [self._preprocess(img) for img in images]
        batch = torch.stack(tensors, dim=0)
        param = next(self._visual.parameters())
        return batch.to(device=param.device, dtype=param.dtype)

    def encode(self, pixel_values: torch.Tensor) -> torch.LongTensor:
        pixels = self.preprocess(pixel_values)
        with torch.no_grad():
            _, _, info = self._visual.vq.encode(pixels)
            indices = info[-1]
        return indices.long()

    def encode_post_quant_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return ViT patch features after VQ: [B, num_tokens, post_quant_embed_dim]."""
        pixels = self.preprocess(pixel_values)
        with torch.no_grad():
            feats = self._visual.forward_features(pixels)
        return feats

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        model_config: str,
        image_size: int,
        vqgan_checkpoint: str,
        codebook_size: int = 16384,
        embed_dim: int = 8,
        post_quant_embed_dim: int = 1152,
        fp16: bool = True,
    ) -> "TokLIPTokenizer":
        _ensure_toklip_import_path()
        _ensure_vqgan_symlink(vqgan_checkpoint)

        from open_clip.factory import create_model_and_transforms

        checkpoint_path = os.path.abspath(checkpoint_path)
        device = "cpu"
        model, _, preprocess_val = create_model_and_transforms(
            model_config,
            checkpoint_path,
            precision="fp16" if fp16 else "fp32",
            device=device,
            force_image_size=image_size,
            output_dict=True,
        )
        visual = model.visual.trunk if hasattr(model.visual, "trunk") else model.visual
        if fp16:
            visual = visual.half()
        visual.eval()

        patch_size = 16
        num_tokens = (image_size // patch_size) ** 2
        print(
            f"Loaded TokLIP tokenizer: config={model_config}, codebook={codebook_size}, "
            f"post_quant_dim={post_quant_embed_dim}, image={image_size}, tokens={num_tokens}",
            flush=True,
        )
        return cls(
            visual,
            preprocess_val,
            codebook_size=codebook_size,
            embed_dim=embed_dim,
            post_quant_embed_dim=post_quant_embed_dim,
            image_size=image_size,
            num_tokens=num_tokens,
            fp16=fp16,
        )
