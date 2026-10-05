"""MLLM adapter: VQGAN(frozen) -> vision features -> MLP projector -> LLM.

Supported vision paths include:
  discrete / chameleon : code indices -> code_embed -> projector
  unitok               : UniTok encoder -> VQ -> post_quant_proj -> projector
  unitok_quant         : UniTok encoder -> VQ (pre post_quant_proj) -> projector
  encoder / quantconv  : LlamaGen continuous ablations
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
if TYPE_CHECKING:
    from transformers import PreTrainedModel

from vision_encoder_eval.mllm.discrete.model.tokenizers.base import BaseDiscreteTokenizer

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200

VIS_MODE_ALIASES = {
    "discrete": "discrete",
    "e1": "discrete",
    "unitok": "unitok",
    "unitok_postquant": "unitok",
    "unitok_quant": "unitok_quant",
    "unitok_preproj": "unitok_quant",
    "chameleon": "chameleon",
    "seed": "seed",
    "seed_post_quant": "seed_post_quant",
    "vilau": "vilau",
    "qlip_post_quant": "qlip_post_quant",
    "qlip_pre_quant": "qlip_pre_quant",
    "tokenflow_post_quant": "tokenflow_post_quant",
    "toklip_post_quant": "toklip_post_quant",
    "uniar": "uniar",
    "encoder": "encoder",
    "e2": "encoder",
    "quantconv": "quantconv",
    "e3": "quantconv",
}

# Continuous post-quant features → MLP projector (no codebook embedding table).
POST_QUANT_VIS_MODES = frozenset(
    {
        "seed_post_quant",
        "qlip_post_quant",
        "tokenflow_post_quant",
        "toklip_post_quant",
        "vilau",
        "uniar",
    }
)


class DiscreteVisualAdapter(nn.Module):
    """MLLM with discrete/continuous visual features and MLP projector."""

    def __init__(
        self,
        tokenizer: BaseDiscreteTokenizer,
        llm: PreTrainedModel,
        hidden_size: int,
        vis_mode: str = "discrete",
        projector_cfg: dict | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer.requires_grad_(False)
        self.tokenizer.eval()

        self.llm = llm
        self.hidden_size = hidden_size
        self.projector_cfg = projector_cfg or {}
        self.num_visual_tokens = tokenizer.num_image_tokens

        self.code_embed = self._build_code_embed(tokenizer)

        self.vis_mode = "discrete"
        self.projector = self._build_projector("discrete", hidden_size, self.projector_cfg)
        self.set_vis_mode(vis_mode)

        self.tokenizer_model_max_length: int | None = None

    @staticmethod
    def _build_code_embed(tokenizer: BaseDiscreteTokenizer) -> nn.Embedding | None:
        if hasattr(tokenizer, "codebook_weight"):
            weight = tokenizer.codebook_weight()
            return nn.Embedding.from_pretrained(weight, freeze=True)
        if hasattr(tokenizer, "_vqgan"):
            weight = tokenizer._vqgan.quantize.embedding.weight.detach().clone()
            return nn.Embedding.from_pretrained(weight, freeze=True)
        if hasattr(tokenizer, "_vq_model"):
            weight = tokenizer._vq_model.quantize.embedding.weight.detach().clone()
            return nn.Embedding.from_pretrained(weight, freeze=True)
        if hasattr(tokenizer, "_model") and hasattr(tokenizer._model, "quantize"):
            quantize = tokenizer._model.quantize
            if hasattr(quantize, "embedding"):
                weight = quantize.embedding.weight.detach().clone()
                return nn.Embedding.from_pretrained(weight, freeze=True)
        return None

    @staticmethod
    def _normalize_vis_mode(mode: str) -> str:
        key = mode.lower().strip()
        if key not in VIS_MODE_ALIASES:
            raise ValueError(
                f"Unknown vis_mode {mode!r}; expected discrete, chameleon, seed, seed_post_quant, "
                "vilau, qlip_post_quant, qlip_pre_quant, tokenflow_post_quant, toklip_post_quant, "
                "uniar, encoder, quantconv, unitok, or unitok_quant"
            )
        return VIS_MODE_ALIASES[key]

    def _projector_input_dim(self, vis_mode: str) -> int:
        if vis_mode == "quantconv":
            return 8
        if vis_mode == "unitok_quant":
            if hasattr(self.tokenizer, "quant_feature_dim"):
                return int(self.tokenizer.quant_feature_dim)
            raise ValueError("unitok_quant vis_mode requires a UniTok tokenizer with quant_feature_dim")
        if vis_mode in POST_QUANT_VIS_MODES:
            if hasattr(self.tokenizer, "post_quant_embed_dim"):
                return int(self.tokenizer.post_quant_embed_dim)
            raise ValueError(
                f"{vis_mode} vis_mode requires a tokenizer with post_quant_embed_dim"
            )
        if vis_mode == "qlip_pre_quant":
            if hasattr(self.tokenizer, "pre_quant_embed_dim"):
                return int(self.tokenizer.pre_quant_embed_dim)
            raise ValueError(
                "qlip_pre_quant vis_mode requires a tokenizer with pre_quant_embed_dim"
            )
        if vis_mode == "unitok" and hasattr(self.tokenizer, "embed_dim"):
            return int(self.tokenizer.embed_dim)
        if hasattr(self.tokenizer, "embed_dim"):
            return int(self.tokenizer.embed_dim)
        if hasattr(self.tokenizer, "_vqgan") and self.tokenizer._vqgan.post_quant_conv is not None:
            return self.tokenizer._vqgan.post_quant_conv.out_channels
        if self.code_embed is not None:
            return self.code_embed.embedding_dim
        raise ValueError("Cannot infer projector input dim from tokenizer")

    @staticmethod
    def _resolve_projector_architecture(projector_cfg: dict) -> str:
        arch = str(projector_cfg.get("architecture", "mlp2x")).lower()
        projector_id = str(projector_cfg.get("id", "")).lower()
        if arch in ("fc", "linear") or projector_id in ("fc_projector", "linear_projector"):
            return "fc"
        if arch in ("mlp1", "mlp1x") or projector_id == "mlp1x_projector":
            return "mlp1x"
        if arch in ("mlp2", "mlp2x") or projector_id in ("mlp_projector", "mlp2x_projector", "mlp2x"):
            return "mlp2x"
        raise ValueError(
            f"Unknown projector architecture {arch!r}; "
            "expected fc, mlp1x, or mlp2x"
        )

    @staticmethod
    def _resolve_projector_dims(
        projector_cfg: dict,
        hidden_size: int,
        architecture: str,
    ) -> list[int]:
        """Resolve per-layer output dims for the projector.

        ``projector_cfg.hidden_dims`` overrides architecture presets, e.g.
        ``[1024, 2048, 2048]`` for 256 -> 1024 -> 2048 -> 2048.
        The last value must equal ``hidden_size`` (LLM hidden dim).
        """
        hidden_dims = projector_cfg.get("hidden_dims")
        if hidden_dims is not None:
            dims = [int(d) for d in hidden_dims]
            if not dims:
                raise ValueError("projector.hidden_dims must not be empty")
            if dims[-1] != hidden_size:
                raise ValueError(
                    f"projector.hidden_dims last value must be LLM hidden_size ({hidden_size}), "
                    f"got {dims[-1]}"
                )
            if architecture == "fc" and len(dims) != 1:
                raise ValueError("projector architecture 'fc' requires a single hidden_dims entry")
            if architecture == "mlp1x" and len(dims) != 1:
                raise ValueError("projector architecture 'mlp1x' requires a single hidden_dims entry")
            return dims

        if architecture in ("fc", "mlp1x"):
            return [hidden_size]
        return [hidden_size, hidden_size]

    @staticmethod
    def _build_projector_module(
        in_dim: int,
        hidden_dims: list[int],
        architecture: str,
    ) -> nn.Sequential:
        if architecture == "fc":
            return nn.Sequential(nn.Linear(in_dim, hidden_dims[0]))

        layers: list[nn.Module] = []
        cur = in_dim
        for i, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(cur, out_dim))
            is_last = i == len(hidden_dims) - 1
            if architecture == "mlp1x":
                if is_last:
                    layers.append(nn.GELU())
            else:
                layers.append(nn.GELU())
            cur = out_dim
        return nn.Sequential(*layers)

    def _build_projector(
        self,
        vis_mode: str,
        hidden_size: int,
        projector_cfg: dict | None = None,
    ) -> nn.Sequential:
        projector_cfg = projector_cfg or {}
        architecture = self._resolve_projector_architecture(projector_cfg)
        if vis_mode == "quantconv":
            quantconv_dims = projector_cfg.get("hidden_dims")
            if quantconv_dims is not None:
                return self._build_projector_module(
                    8, [int(d) for d in quantconv_dims], architecture
                )
            return nn.Sequential(
                nn.Linear(8, 256),
                nn.GELU(),
                nn.Linear(256, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
        proj_in = self._projector_input_dim(vis_mode)
        hidden_dims = self._resolve_projector_dims(projector_cfg, hidden_size, architecture)
        return self._build_projector_module(proj_in, hidden_dims, architecture)

    def set_vis_mode(self, mode: str) -> None:
        """Switch vision encoding path (call before loading phase-1 weights)."""
        mode = self._normalize_vis_mode(mode)
        if mode == self.vis_mode:
            return
        self.vis_mode = mode
        self.projector = self._build_projector(mode, self.hidden_size, self.projector_cfg)

    def _truncate_sequence(
        self,
        embeds: torch.Tensor,
        labels: torch.Tensor,
        attn: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        max_len = self.tokenizer_model_max_length
        if max_len is None or embeds.shape[0] <= max_len:
            return embeds, labels, attn
        embeds = embeds[:max_len]
        labels = labels[:max_len]
        if attn is not None:
            attn = attn[:max_len]
        return embeds, labels, attn

    def set_trainable_modules(self, *, llm: bool, projector: bool) -> None:
        """Configure which adapter modules are trainable."""
        for p in self.llm.parameters():
            p.requires_grad_(llm)
        for p in self.projector.parameters():
            p.requires_grad_(projector)
        self.train()
        self.llm.train()
        if llm and hasattr(self.llm, "enable_input_require_grads"):
            self.llm.enable_input_require_grads()

    def set_phase(self, phase: int) -> None:
        """Configure which parameters are trainable (finetune presets)."""
        if phase == 1:
            self.set_trainable_modules(llm=False, projector=True)
        elif phase == 2:
            self.set_trainable_modules(llm=True, projector=True)
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def get_trainable_params(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self) -> None:
        self.llm.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self.llm.is_gradient_checkpointing

    def _compute_vis_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images into LLM-space visual token embeddings."""
        if self.vis_mode == "unitok_quant":
            embeds = self.tokenizer.encode_quant_features(pixel_values)
            proj_dtype = next(self.projector.parameters()).dtype
            if embeds.dtype != proj_dtype:
                embeds = embeds.to(dtype=proj_dtype)
            return self.projector(embeds)

        if self.vis_mode == "unitok":
            embeds = self.tokenizer.encode(pixel_values)
            proj_dtype = next(self.projector.parameters()).dtype
            if embeds.dtype != proj_dtype:
                embeds = embeds.to(dtype=proj_dtype)
            return self.projector(embeds)

        if self.vis_mode == "chameleon":
            if self.code_embed is None:
                raise RuntimeError("chameleon vis_mode requires a frozen code embedding table")
            vis_ids = self.tokenizer.encode(pixel_values)
            code_feats = self.code_embed(vis_ids)
            return self.projector(code_feats)

        if self.vis_mode == "seed":
            if self.code_embed is None:
                raise RuntimeError("seed vis_mode requires a frozen code embedding table")
            vis_ids = self.tokenizer.encode(pixel_values)
            code_feats = self.code_embed(vis_ids)
            return self.projector(code_feats)

        if self.vis_mode in POST_QUANT_VIS_MODES:
            if not hasattr(self.tokenizer, "encode_post_quant_features"):
                raise RuntimeError(
                    f"{self.vis_mode} vis_mode requires tokenizer.encode_post_quant_features"
                )
            embeds = self.tokenizer.encode_post_quant_features(pixel_values)
            proj_dtype = next(self.projector.parameters()).dtype
            if embeds.dtype != proj_dtype:
                embeds = embeds.to(dtype=proj_dtype)
            return self.projector(embeds)

        if self.vis_mode == "qlip_pre_quant":
            if not hasattr(self.tokenizer, "encode_pre_quant_features"):
                raise RuntimeError(
                    "qlip_pre_quant vis_mode requires tokenizer.encode_pre_quant_features"
                )
            embeds = self.tokenizer.encode_pre_quant_features(pixel_values)
            return self.projector(embeds)

        if self.vis_mode == "encoder":
            pixels = self.tokenizer.preprocess(pixel_values)
            if pixels.dtype != next(self.tokenizer._vqgan.encoder.parameters()).dtype:
                pixels = pixels.to(dtype=next(self.tokenizer._vqgan.encoder.parameters()).dtype)
            with torch.no_grad():
                h = self.tokenizer._vqgan.encoder(pixels)
            batch_size, channels, height, width = h.shape
            vis_feats = h.reshape(batch_size, channels, height * width).permute(0, 2, 1)
            return self.projector(vis_feats)

        if self.vis_mode == "quantconv":
            pixels = self.tokenizer.preprocess(pixel_values)
            if pixels.dtype != next(self.tokenizer._vqgan.encoder.parameters()).dtype:
                pixels = pixels.to(dtype=next(self.tokenizer._vqgan.encoder.parameters()).dtype)
            with torch.no_grad():
                h = self.tokenizer._vqgan.encoder(pixels)
                h = self.tokenizer._vqgan.quant_conv(h)
            batch_size, channels, height, width = h.shape
            vis_feats = h.reshape(batch_size, channels, height * width).permute(0, 2, 1)
            return self.projector(vis_feats)

        vis_ids = self.tokenizer.encode(pixel_values)
        code_feats = self.code_embed(vis_ids)
        num_images, num_tokens, code_dim = code_feats.shape
        spatial = int(num_tokens ** 0.5)
        feat_map = code_feats.permute(0, 2, 1).reshape(num_images, code_dim, spatial, spatial)
        feat_map = self.tokenizer._vqgan.post_quant_conv(feat_map)
        vis_feats = feat_map.reshape(num_images, -1, num_tokens).permute(0, 2, 1)
        return self.projector(vis_feats)

    def _maybe_log_vis_discrimination(self, vis_embeds: torch.Tensor) -> None:
        if not self.training or vis_embeds.shape[0] < 2:
            return
        if torch.rand(1, device=vis_embeds.device).item() >= 0.002:
            return
        cos = F.cosine_similarity(
            vis_embeds[0].flatten().unsqueeze(0),
            vis_embeds[1].flatten().unsqueeze(0),
        ).item()
        print(f"[vis] mode={self.vis_mode} cos_sim={cos:.4f}", flush=True)

    def _interleave(
        self,
        input_ids_1d: torch.LongTensor,
        labels_1d: torch.LongTensor,
        attn_1d: Optional[torch.Tensor],
        vis_embeds: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Interleave visual embeddings at IMAGE_TOKEN_INDEX positions."""
        embed = self.llm.get_input_embeddings()
        image_pos = (input_ids_1d == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        num_images = len(image_pos)

        if num_images == 0:
            return embed(input_ids_1d), labels_1d, attn_1d

        if vis_embeds.dim() == 3:
            if vis_embeds.shape[0] != num_images:
                raise ValueError(
                    f"Image count mismatch: prompt has {num_images} <image> slots, "
                    f"but {vis_embeds.shape[0]} image tensors were provided."
                )
            image_embeds = [vis_embeds[i] for i in range(num_images)]
        else:
            if num_images != 1:
                raise ValueError(
                    f"Image count mismatch: prompt has {num_images} <image> slots, "
                    "but only one image embedding was provided."
                )
            image_embeds = [vis_embeds]

        text_chunks, lbl_chunks = [], []
        start = 0
        for pos in image_pos:
            text_chunks.append(input_ids_1d[start:pos])
            lbl_chunks.append(labels_1d[start:pos])
            start = pos + 1
        text_chunks.append(input_ids_1d[start:])
        lbl_chunks.append(labels_1d[start:])

        result_e, result_l, result_a = [], [], []
        attn_idx = 0
        for i in range(len(text_chunks)):
            if len(text_chunks[i]) > 0:
                result_e.append(embed(text_chunks[i]))
                result_l.append(lbl_chunks[i])
                if attn_1d is not None:
                    chunk_len = len(text_chunks[i])
                    result_a.append(attn_1d[attn_idx:attn_idx + chunk_len])
                    attn_idx += chunk_len
            if i < num_images:
                vis_e = image_embeds[i]
                result_e.append(vis_e)
                result_l.append(torch.full((vis_e.shape[0],), IGNORE_INDEX,
                                           device=device, dtype=labels_1d.dtype))
                if attn_1d is not None:
                    result_a.append(torch.ones(vis_e.shape[0], device=device,
                                               dtype=attn_1d.dtype))
                attn_idx += 1

        e = torch.cat(result_e, dim=0)
        l = torch.cat(result_l, dim=0)
        a = torch.cat(result_a, dim=0) if attn_1d is not None else None
        return self._truncate_sequence(e, l, a)

    def _vis_embeds_for_sample(
        self,
        pixel_values: torch.Tensor,
        vis_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        batch_idx: int,
    ) -> torch.Tensor:
        """Map encoded images to one prompt (supports MMMU multi-image, batch size 1)."""
        batch_size = input_ids.shape[0]
        if batch_size == 1 and pixel_values.shape[0] > 1:
            return vis_embeds
        return vis_embeds[batch_idx]

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """Forward pass with inline <image> token replacement."""
        batch_size = input_ids.shape[0]
        device = pixel_values.device

        vis_embeds = self._compute_vis_embeds(pixel_values)
        self._maybe_log_vis_discrimination(vis_embeds)

        batch_e, batch_l, batch_a = [], [], []
        for b in range(batch_size):
            e, l, a = self._interleave(
                input_ids[b], labels[b],
                attention_mask[b] if attention_mask is not None else None,
                self._vis_embeds_for_sample(pixel_values, vis_embeds, input_ids, b),
                device,
            )
            batch_e.append(e)
            batch_l.append(l)
            if attention_mask is not None:
                batch_a.append(a)

        max_len = max(e.shape[0] for e in batch_e)
        padded_e, padded_l, padded_a = [], [], []
        for b in range(batch_size):
            e, l = batch_e[b], batch_l[b]
            pad = max_len - e.shape[0]
            if pad > 0:
                zp = torch.zeros(pad, e.shape[1], device=device, dtype=e.dtype)
                padded_e.append(torch.cat([e, zp], dim=0).unsqueeze(0))
                lip = torch.full((pad,), IGNORE_INDEX, device=device, dtype=l.dtype)
                padded_l.append(torch.cat([l, lip], dim=0).unsqueeze(0))
                if attention_mask is not None:
                    ap = torch.zeros(pad, device=device, dtype=torch.bool)
                    padded_a.append(torch.cat([batch_a[b], ap], dim=0).unsqueeze(0))
            else:
                padded_e.append(e.unsqueeze(0))
                padded_l.append(l.unsqueeze(0))
                if attention_mask is not None:
                    padded_a.append(batch_a[b].unsqueeze(0))

        inputs_embeds = torch.cat(padded_e, dim=0)
        new_labels = torch.cat(padded_l, dim=0)
        new_attn = torch.cat(padded_a, dim=0) if attention_mask is not None else None

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attn,
            labels=new_labels,
            use_cache=False,
            **kwargs,
        )
        return {"loss": outputs.loss, "logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **gen_kwargs,
    ) -> torch.LongTensor:
        """Generate text conditioned on an image (supports inline <image>)."""
        batch_size = input_ids.shape[0]
        device = pixel_values.device

        vis_embeds = self._compute_vis_embeds(pixel_values)

        batch_e, batch_a = [], []
        for b in range(batch_size):
            dummy_labels = torch.zeros(input_ids.shape[1], dtype=torch.long, device=device)
            e, _, a = self._interleave(
                input_ids[b], dummy_labels,
                attention_mask[b] if attention_mask is not None else None,
                self._vis_embeds_for_sample(pixel_values, vis_embeds, input_ids, b),
                device,
            )
            batch_e.append(e)
            if attention_mask is not None:
                batch_a.append(a)

        max_len = max(e.shape[0] for e in batch_e)
        padded_e, padded_a = [], []
        for b in range(batch_size):
            e = batch_e[b]
            pad = max_len - e.shape[0]
            if pad > 0:
                zp = torch.zeros(pad, e.shape[1], device=device, dtype=e.dtype)
                padded_e.append(torch.cat([e, zp], dim=0).unsqueeze(0))
                if attention_mask is not None:
                    ap = torch.zeros(pad, device=device, dtype=torch.bool)
                    padded_a.append(torch.cat([batch_a[b], ap], dim=0).unsqueeze(0))
            else:
                padded_e.append(e.unsqueeze(0))
                if attention_mask is not None:
                    padded_a.append(batch_a[b].unsqueeze(0))

        inputs_embeds = torch.cat(padded_e, dim=0)
        new_attn = torch.cat(padded_a, dim=0) if attention_mask is not None else (
            torch.ones(max_len, dtype=torch.bool, device=device).unsqueeze(0).expand(batch_size, -1)
        )

        prompt_len = inputs_embeds.shape[1]
        for sc in gen_kwargs.get("stopping_criteria") or []:
            if hasattr(sc, "start_len"):
                sc.start_len = prompt_len

        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attn,
            **gen_kwargs,
        )
        prompt_len = inputs_embeds.shape[1]
        if output_ids.shape[1] > prompt_len:
            return output_ids[:, prompt_len:]
        return output_ids
