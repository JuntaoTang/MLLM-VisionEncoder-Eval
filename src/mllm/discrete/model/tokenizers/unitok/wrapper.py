# -*- coding: utf-8 -*-
"""UniTok discrete visual tokenizer wrapper for VTB-Discrete MLLM."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from vision_encoder_eval.mllm.discrete.model.tokenizers.base import BaseDiscreteTokenizer
from vision_encoder_eval.mllm.discrete.model.tokenizers.unitok import vitamin
from vision_encoder_eval.mllm.discrete.model.tokenizers.unitok.args import Args
from vision_encoder_eval.mllm.discrete.model.tokenizers.unitok.quant import VectorQuantizerM
from vision_encoder_eval.mllm.discrete.model.tokenizers.unitok.vqvae import AttnProjection


class UniTokTokenizer(BaseDiscreteTokenizer):
    """Wrap UniTok encoder as a discrete tokenizer for MLLM training."""

    def __init__(self, checkpoint_path: str, num_query: int | None = None):
        super().__init__()
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        args = Args()
        args.load_state_dict(ckpt["args"])
        if num_query is not None and num_query != args.num_query:
            print(
                f"Warning: ignoring num_query={num_query}; checkpoint uses num_query={args.num_query}",
                flush=True,
            )

        self.encoder = timm.create_model(
            args.model,
            patch_size=1,
            fc_norm=False,
            drop_rate=0.0,
            num_classes=0,
            global_pool="",
            pos_embed="none",
            class_token=False,
            mlp_layer=vitamin.GeGluMlp,
            reg_tokens=args.num_query,
            img_size=args.img_size,
            drop_path_rate=args.drop_path,
        )
        self.encoder.pos_embed = nn.Parameter(
            torch.zeros(1, 1, self.encoder.embed_dim), requires_grad=False
        )
        self.embed_dim = self.encoder.embed_dim

        if args.quant_proj == "linear":
            self.quant_proj = nn.Linear(self.embed_dim, args.vocab_width)
        else:
            self.quant_proj = AttnProjection(
                self.embed_dim, args.vocab_width, args.num_codebooks,
            )

        self.quantizer = VectorQuantizerM(
            vocab_size=args.vocab_size,
            vocab_width=args.vocab_width,
            beta=args.vq_beta,
            use_entropy_loss=args.le > 0,
            entropy_temp=args.e_temp,
            num_codebooks=args.num_codebooks,
        )

        if args.quant_proj == "linear":
            self.post_quant_proj = nn.Linear(args.vocab_width, self.embed_dim)
        else:
            self.post_quant_proj = AttnProjection(
                args.vocab_width, self.embed_dim, args.num_codebooks,
            )

        model_w = {
            k: v
            for k, v in ckpt["trainer"]["unitok"].items()
            if k.startswith("encoder") or k.startswith("quantizer") or "quant_proj" in k
        }
        self.load_state_dict(model_w, strict=True)
        self.requires_grad_(False)
        self.eval()
        self._cfg = args
        self._image_size = args.img_size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, args.img_size, args.img_size)
            self._num_tokens = int(self.encoder(dummy).shape[1])
        # Convert after dummy probe — Stem/BN expect matching input dtype.
        self.to(torch.bfloat16)
    @property
    def vocab_size(self) -> int:
        return self._cfg.vocab_size

    @property
    def num_image_tokens(self) -> int:
        return self._num_tokens

    @property
    def image_size(self) -> int:
        return self._image_size

    def preprocess(self, pixel_values):
        if pixel_values.max() <= 1.0:
            pixel_values = pixel_values * 2.0 - 1.0
        return pixel_values

    def encode(self, pixel_values):
        return self._encode_features(pixel_values, apply_post_quant=True)

    def encode_quant_features(self, pixel_values):
        """Return VQ codebook features before ``post_quant_proj`` (vocab_width-d)."""
        return self._encode_features(pixel_values, apply_post_quant=False)

    def _encode_features(self, pixel_values, apply_post_quant: bool):
        pixels = self.preprocess(pixel_values)
        param = next(self.encoder.parameters())
        pixels = pixels.to(device=param.device, dtype=param.dtype)
        with torch.no_grad():
            t = self.encoder(pixels)
            t = self.quant_proj(t)
            idx = self.quantizer.f_to_idx(t)
            t = self.quantizer.idx_to_f(idx)
            if apply_post_quant:
                t = self.post_quant_proj(t)
        return t.to(dtype=pixel_values.dtype, device=pixel_values.device)

    @property
    def quant_feature_dim(self) -> int:
        return int(self._cfg.vocab_width)

    def decode(self, indices):
        raise NotImplementedError
