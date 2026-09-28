"""Build discrete visual tokenizers from training/eval configs."""

from __future__ import annotations

from src.discrete.config import resolve_model_path
from src.discrete.model.tokenizers.base import BaseDiscreteTokenizer
from src.discrete.model.tokenizers.toklip.wrapper import TokLIPTokenizer
from src.discrete.model.tokenizers.unitok.wrapper import UniTokTokenizer
from src.discrete.model.tokenizers.vilau.wrapper import VilaUTokenizer
from src.discrete.model.vision_config import (
    resolve_unitok_checkpoint,
    resolve_unitok_num_query,
    resolve_vis_mode,
)

_TOKLIP_VQGAN_DEFAULT = "/cache/ckpt/download/tokenizer/discrete/toklip/vq_ds16_t2i.pt"


def build_visual_tokenizer(cfg: dict) -> BaseDiscreteTokenizer:
    tokenizer_cfg = cfg["tokenizer"]
    tok_type = tokenizer_cfg.get("type")
    vis_mode = resolve_vis_mode(cfg)

    if vis_mode in ("unitok", "unitok_quant") or tok_type == "unitok":
        ckpt = resolve_unitok_checkpoint(cfg)
        num_query = resolve_unitok_num_query(cfg)
        print(f"Loading UniTok: {ckpt} (num_query={num_query})", flush=True)
        return UniTokTokenizer(checkpoint_path=ckpt, num_query=num_query)

    if tok_type == "vilau":
        ckpt = resolve_model_path(str(tokenizer_cfg["checkpoint_path"]))
        print(f"Loading VILA-U: {ckpt}", flush=True)
        return VilaUTokenizer.from_checkpoint(
            ckpt,
            codebook_size=int(tokenizer_cfg.get("codebook_size", 16384)),
            embed_dim=int(tokenizer_cfg.get("embed_dim", 1024)),
            post_quant_embed_dim=int(tokenizer_cfg.get("post_quant_embed_dim", 1024)),
            image_size=int(tokenizer_cfg.get("image_size", 256)),
            num_tokens=int(tokenizer_cfg.get("num_tokens", 256)),
            rq_depth=int(tokenizer_cfg.get("rq_depth", 4)),
            fp16=bool(tokenizer_cfg.get("fp16", True)),
        )

    if tok_type == "toklip":
        ckpt = resolve_model_path(str(tokenizer_cfg["checkpoint_path"]))
        arch_cfg = cfg.get("arch") or {}
        model_config = (
            tokenizer_cfg.get("model_config")
            or arch_cfg.get("toklip_model_config")
            or "ViT-SO400M-16-SigLIP2-256-toklip"
        )
        vqgan_ckpt = resolve_model_path(
            str(
                tokenizer_cfg.get("vqgan_checkpoint")
                or arch_cfg.get("toklip_vqgan_checkpoint")
                or _TOKLIP_VQGAN_DEFAULT
            )
        )
        print(f"Loading TokLIP: {ckpt}", flush=True)
        return TokLIPTokenizer.from_checkpoint(
            ckpt,
            model_config=str(model_config),
            image_size=int(tokenizer_cfg.get("image_size", 256)),
            vqgan_checkpoint=vqgan_ckpt,
            codebook_size=int(tokenizer_cfg.get("codebook_size", 16384)),
            embed_dim=int(tokenizer_cfg.get("embed_dim", 8)),
            post_quant_embed_dim=int(tokenizer_cfg.get("post_quant_embed_dim", 1152)),
            fp16=bool(tokenizer_cfg.get("fp16", True)),
        )

    if tok_type == "bitdance" or vis_mode == "bitdance":
        from src.discrete.model.tokenizers.bitdance import BitDanceTokenizer

        ckpt = resolve_model_path(str(tokenizer_cfg["checkpoint_path"]))
        config_path = tokenizer_cfg.get("config_path")
        if config_path:
            config_path = resolve_model_path(str(config_path))
        print(f"Loading BitDance: {ckpt}", flush=True)
        return BitDanceTokenizer.from_checkpoint(
            ckpt,
            config_path=config_path,
            codebook_size=tokenizer_cfg.get("codebook_size"),
            embed_dim=int(tokenizer_cfg.get("embed_dim", 128)),
            post_quant_embed_dim=int(tokenizer_cfg.get("post_quant_embed_dim", 128)),
            image_size=int(tokenizer_cfg.get("image_size", 256)),
            num_tokens=tokenizer_cfg.get("num_tokens"),
            fp16=bool(tokenizer_cfg.get("fp16", True)),
        )

    if tok_type == "atoken" or vis_mode == "atoken":
        from src.discrete.model.tokenizers.atoken import ATokenTokenizer

        ckpt = resolve_model_path(str(tokenizer_cfg["checkpoint_path"]))
        config_path = tokenizer_cfg.get("config_path")
        if config_path:
            config_path = resolve_model_path(str(config_path))
        print(f"Loading AToken: {ckpt}", flush=True)
        return ATokenTokenizer.from_checkpoint(
            ckpt,
            config_path=config_path,
            repo_root=tokenizer_cfg.get("repo_root"),
            codebook_size=int(tokenizer_cfg.get("codebook_size", 4096)),
            embed_dim=int(tokenizer_cfg.get("embed_dim", 48)),
            post_quant_embed_dim=int(tokenizer_cfg.get("post_quant_embed_dim", 48)),
            image_size=int(tokenizer_cfg.get("image_size", 256)),
            num_tokens=int(tokenizer_cfg.get("num_tokens", 256)),
            fp16=bool(tokenizer_cfg.get("fp16", True)),
        )

    if tok_type == "uniar" or vis_mode == "uniar":
        from src.discrete.model.tokenizers.uniar import UniARTokenizer

        ckpt = resolve_model_path(str(tokenizer_cfg["checkpoint_path"]))
        print(f"Loading UniAR BSQ: {ckpt}", flush=True)
        return UniARTokenizer.from_checkpoint(
            ckpt,
            repo_root=tokenizer_cfg.get("repo_root"),
            codebook_size=int(tokenizer_cfg.get("codebook_size", 2**31 - 1)),
            embed_dim=int(tokenizer_cfg.get("embed_dim", 1152)),
            post_quant_embed_dim=int(tokenizer_cfg.get("post_quant_embed_dim", 4608)),
            image_size=int(tokenizer_cfg.get("image_size", 512)),
            num_tokens=int(tokenizer_cfg.get("num_tokens", 1024)),
            bsq_only=bool(tokenizer_cfg.get("bsq_only", True)),
            fp16=bool(tokenizer_cfg.get("fp16", True)),
        )

    raise ValueError(
        f"Unsupported discrete tokenizer type={tok_type!r} vis_mode={vis_mode!r}. "
        "Available: unitok, vilau, toklip, bitdance, atoken, uniar."
    )
