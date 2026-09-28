"""Shared helpers for discrete eval (VLMEvalKit model config)."""

from __future__ import annotations

from typing import Any


def build_tokenizer_cfg(tokenizer: dict) -> dict:
    """Copy tokenizer preset fields used by training and eval."""
    fields = (
        "type",
        "checkpoint_path",
        "config_path",
        "repo_root",
        "codebook_size",
        "embed_dim",
        "image_size",
        "quant_proj",
        "num_query",
        "model_config",
        "vqgan_checkpoint",
        "post_quant_embed_dim",
        "num_tokens",
        "rq_depth",
        "bsq_only",
        "fp16",
    )
    return {k: tokenizer[k] for k in fields if k in tokenizer and tokenizer[k] is not None}


def build_vlmeval_model_entry(ctx, checkpoint_dir: str) -> dict:
    """Build VLMEvalKit model entry for a discrete recipe."""
    eval_cfg = ctx.eval or {}
    arch_cfg = ctx.arch or {}
    tok = ctx.tokenizer
    tok_type = tok.get("type")
    vis_mode = arch_cfg.get("vis_mode", tok_type)

    entry: dict[str, Any] = {
        "class": str(eval_cfg.get("vlm_class", "VTB_Discrete_VLM")),
        "model_path": checkpoint_dir,
        "llm_path": ctx.llm["model_name_or_path"],
        "hidden_size": ctx.llm.get("hidden_size", 2048),
        "vis_mode": vis_mode,
        "projector": ctx.projector,
        "max_new_tokens": int(eval_cfg.get("max_new_tokens", 2048)),
        "log_every": int(eval_cfg.get("log_every", 50)),
        "verbose_responses": bool(eval_cfg.get("verbose_responses", False)),
    }

    if tok_type == "unitok":
        entry["unitok_checkpoint_path"] = tok["checkpoint_path"]
        entry["num_query"] = tok.get("num_query", 256)
    elif tok_type == "vilau":
        entry["vilau_checkpoint_path"] = tok["checkpoint_path"]
        entry["embed_dim"] = tok.get("embed_dim", 1024)
        entry["image_size"] = tok.get("image_size", 256)
        entry["num_tokens"] = tok.get("num_tokens", 256)
        entry["rq_depth"] = tok.get("rq_depth", 4)
        if tok.get("post_quant_embed_dim") is not None:
            entry["post_quant_embed_dim"] = tok["post_quant_embed_dim"]
    elif tok_type == "toklip":
        entry["tokenizer_checkpoint_path"] = tok["checkpoint_path"]
        entry["toklip_model_config"] = tok.get("model_config")
        entry["toklip_vqgan_checkpoint_path"] = tok.get("vqgan_checkpoint")
        entry["image_size"] = tok.get("image_size", 256)
        if tok.get("post_quant_embed_dim") is not None:
            entry["post_quant_embed_dim"] = tok["post_quant_embed_dim"]
    elif tok_type in ("bitdance", "atoken", "uniar"):
        entry["tokenizer_checkpoint_path"] = tok["checkpoint_path"]
        entry["image_size"] = tok.get("image_size")
        entry["num_tokens"] = tok.get("num_tokens")
        if tok.get("config_path") is not None:
            entry["tokenizer_config_path"] = tok["config_path"]
        if tok.get("repo_root") is not None:
            entry["tokenizer_repo_root"] = tok["repo_root"]
        if tok.get("post_quant_embed_dim") is not None:
            entry["post_quant_embed_dim"] = tok["post_quant_embed_dim"]
        if tok.get("bsq_only") is not None:
            entry["bsq_only"] = tok["bsq_only"]
    else:
        raise ValueError(f"Unsupported discrete tokenizer for eval: {tok_type!r}")

    return entry


def build_eval_tokenizer_cfg(
    vis_mode: str,
    *,
    kwargs: dict,
    arch_cfg: dict,
    defaults: dict | None = None,
) -> dict:
    """Resolve tokenizer config for VTB_Discrete_VLM from checkpoint kwargs."""
    defaults = defaults or {}
    tok_type = defaults.get("type") or vis_mode.replace("_post_quant", "").replace("_quant", "")
    if vis_mode in ("unitok", "unitok_quant"):
        return {
            "type": "unitok",
            "checkpoint_path": (
                kwargs.get("unitok_checkpoint_path")
                or arch_cfg.get("unitok_checkpoint")
                or defaults.get("checkpoint_path")
            ),
            "num_query": kwargs.get("num_query", defaults.get("num_query", 256)),
            "quant_proj": defaults.get("quant_proj"),
        }
    if vis_mode == "vilau":
        path = kwargs.get("vilau_checkpoint_path") or arch_cfg.get("vilau_checkpoint") or defaults.get("checkpoint_path")
        return {
            "type": "vilau",
            "checkpoint_path": path,
            "codebook_size": kwargs.get("codebook_size", defaults.get("codebook_size", 16384)),
            "embed_dim": kwargs.get("embed_dim", defaults.get("embed_dim", 1024)),
            "post_quant_embed_dim": kwargs.get("post_quant_embed_dim", defaults.get("post_quant_embed_dim", 1024)),
            "image_size": kwargs.get("image_size", defaults.get("image_size", 256)),
            "num_tokens": kwargs.get("num_tokens", defaults.get("num_tokens", 256)),
            "rq_depth": kwargs.get("rq_depth", defaults.get("rq_depth", 4)),
            "fp16": kwargs.get("fp16", defaults.get("fp16", True)),
        }
    if vis_mode == "toklip_post_quant":
        return {
            "type": "toklip",
            "checkpoint_path": (
                kwargs.get("tokenizer_checkpoint_path")
                or arch_cfg.get("tokenizer_checkpoint")
                or defaults.get("checkpoint_path")
            ),
            "model_config": kwargs.get("toklip_model_config") or arch_cfg.get("toklip_model_config") or defaults.get("model_config"),
            "vqgan_checkpoint": (
                kwargs.get("toklip_vqgan_checkpoint_path")
                or arch_cfg.get("toklip_vqgan_checkpoint")
                or defaults.get("vqgan_checkpoint")
            ),
            "image_size": kwargs.get("image_size", defaults.get("image_size", 256)),
            "codebook_size": kwargs.get("codebook_size", defaults.get("codebook_size", 16384)),
            "post_quant_embed_dim": kwargs.get("post_quant_embed_dim", defaults.get("post_quant_embed_dim", 1152)),
            "fp16": kwargs.get("fp16", defaults.get("fp16", True)),
        }
    if vis_mode in ("bitdance", "atoken", "uniar"):
        defaults_dim = {"bitdance": 128, "atoken": 48, "uniar": 4608}[vis_mode]
        defaults_size = {"bitdance": 256, "atoken": 256, "uniar": 512}[vis_mode]
        defaults_tokens = {"bitdance": 64, "atoken": 256, "uniar": 1024}[vis_mode]
        cfg = {
            "type": vis_mode,
            "checkpoint_path": (
                kwargs.get("tokenizer_checkpoint_path")
                or arch_cfg.get("tokenizer_checkpoint")
                or defaults.get("checkpoint_path")
            ),
            "config_path": (
                kwargs.get("tokenizer_config_path")
                or arch_cfg.get("tokenizer_config_path")
                or defaults.get("config_path")
            ),
            "repo_root": (
                kwargs.get("tokenizer_repo_root")
                or arch_cfg.get("tokenizer_repo_root")
                or defaults.get("repo_root")
            ),
            "image_size": kwargs.get("image_size", defaults.get("image_size", defaults_size)),
            "num_tokens": kwargs.get("num_tokens", defaults.get("num_tokens", defaults_tokens)),
            "post_quant_embed_dim": kwargs.get(
                "post_quant_embed_dim", defaults.get("post_quant_embed_dim", defaults_dim)
            ),
            "fp16": kwargs.get("fp16", defaults.get("fp16", True)),
        }
        if vis_mode == "uniar":
            cfg["bsq_only"] = kwargs.get("bsq_only", defaults.get("bsq_only", True))
        return cfg
    raise ValueError(f"Unsupported vis_mode for eval: {vis_mode!r}")
