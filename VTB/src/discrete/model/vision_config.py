"""Helpers for resolving vision backend settings from training configs."""

from __future__ import annotations

from typing import Any


def resolve_vis_mode(cfg: dict[str, Any]) -> str:
    arch_mode = (cfg.get("arch") or {}).get("vis_mode")
    if arch_mode:
        return str(arch_mode)

    tokenizer_type = (cfg.get("tokenizer") or {}).get("type")
    if tokenizer_type == "toklip":
        return "toklip_post_quant"
    if tokenizer_type == "vilau":
        return "vilau"
    if tokenizer_type == "bitdance":
        return "bitdance"
    if tokenizer_type == "atoken":
        return "atoken"
    if tokenizer_type == "uniar":
        return "uniar"
    if tokenizer_type == "unitok":
        quant_proj = (cfg.get("tokenizer") or {}).get("quant_proj", "linear")
        if quant_proj == "attn_quant":
            return "unitok_quant"
        return "unitok"
    return "unitok"


def resolve_unitok_checkpoint(cfg: dict[str, Any]) -> str:
    from src.discrete.config import resolve_model_path

    vision = cfg.get("vision") or {}
    if vision.get("checkpoint"):
        return resolve_model_path(str(vision["checkpoint"]))
    tokenizer = cfg.get("tokenizer") or {}
    if tokenizer.get("checkpoint_path"):
        return resolve_model_path(str(tokenizer["checkpoint_path"]))
    raise ValueError("UniTok checkpoint_path not found in config")


def resolve_unitok_num_query(cfg: dict[str, Any], default: int = 256) -> int:
    vision = cfg.get("vision") or {}
    if vision.get("num_query") is not None:
        return int(vision["num_query"])
    tokenizer = cfg.get("tokenizer") or {}
    if tokenizer.get("num_query") is not None:
        return int(tokenizer["num_query"])
    return default


def build_arch_config(cfg: dict[str, Any], vis_mode: str) -> dict[str, Any]:
    arch_cfg = dict(cfg.get("arch") or {})
    arch_cfg["vis_mode"] = vis_mode
    tokenizer = cfg.get("tokenizer") or {}
    projector = cfg.get("projector") or cfg.get("connector") or {}

    if vis_mode in ("unitok", "unitok_quant"):
        arch_cfg["unitok_checkpoint"] = resolve_unitok_checkpoint(cfg)
        arch_cfg["unitok_quant_proj"] = tokenizer.get("quant_proj")

    if vis_mode == "vilau":
        from src.discrete.config import resolve_model_path

        if tokenizer.get("checkpoint_path"):
            arch_cfg["vilau_checkpoint"] = resolve_model_path(str(tokenizer["checkpoint_path"]))
        arch_cfg["projector_architecture"] = projector.get("architecture", "mlp2x")
        if projector.get("hidden_dims") is not None:
            arch_cfg["projector_hidden_dims"] = projector.get("hidden_dims")

    if vis_mode == "toklip_post_quant":
        from src.discrete.config import resolve_model_path

        if tokenizer.get("checkpoint_path"):
            arch_cfg["tokenizer_checkpoint"] = resolve_model_path(str(tokenizer["checkpoint_path"]))
        if tokenizer.get("model_config"):
            arch_cfg["toklip_model_config"] = tokenizer.get("model_config")
        if tokenizer.get("vqgan_checkpoint"):
            arch_cfg["toklip_vqgan_checkpoint"] = resolve_model_path(str(tokenizer["vqgan_checkpoint"]))
        arch_cfg["projector_architecture"] = projector.get("architecture", "mlp2x")
        if projector.get("hidden_dims") is not None:
            arch_cfg["projector_hidden_dims"] = projector.get("hidden_dims")

    if vis_mode in ("bitdance", "atoken", "uniar"):
        from src.discrete.config import resolve_model_path

        if tokenizer.get("checkpoint_path"):
            arch_cfg["tokenizer_checkpoint"] = resolve_model_path(str(tokenizer["checkpoint_path"]))
        if tokenizer.get("config_path"):
            arch_cfg["tokenizer_config_path"] = resolve_model_path(str(tokenizer["config_path"]))
        if tokenizer.get("repo_root"):
            arch_cfg["tokenizer_repo_root"] = tokenizer.get("repo_root")
        if tokenizer.get("bsq_only") is not None:
            arch_cfg["bsq_only"] = tokenizer.get("bsq_only")
        arch_cfg["projector_architecture"] = projector.get("architecture", "mlp2x")
        if projector.get("hidden_dims") is not None:
            arch_cfg["projector_hidden_dims"] = projector.get("hidden_dims")

    return arch_cfg
