#!/usr/bin/env python3
"""VTB unified entry: continuous (CLIP+LLaVA) or discrete (UniTok/VILA-U/TokLIP)."""

import os
import sys

VTB_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, VTB_ROOT)

from src.utils.config import (
    CONFIGS_ROOT,
    CONTINUOUS_CONFIGS,
    DISCRETE_CONFIGS,
    SHARED_RUNTIME_CONFIG,
    install_offline_hf_env,
)

install_offline_hf_env()

VTB_PYTHON = os.environ.get("VTB_PYTHON", "/home/ma-user/miniconda3/envs/VTB/bin/python")
if os.path.isfile(VTB_PYTHON) and os.path.realpath(sys.executable) != os.path.realpath(VTB_PYTHON):
    os.execv(VTB_PYTHON, [VTB_PYTHON, *sys.argv])

import argparse
import yaml

from src.runner.pipeline import run_pipeline

MODE_CONFIG = {
    "continuous": SHARED_RUNTIME_CONFIG,
    "discrete": SHARED_RUNTIME_CONFIG,
}


def main():
    parser = argparse.ArgumentParser(description="VTB training/eval pipeline")
    parser.add_argument("--mode", choices=["continuous", "discrete"], default="continuous")
    parser.add_argument("--config", default=None, help="Override runtime config path")
    parser.add_argument("--train-config", default=None)
    parser.add_argument("--recipe", default=None, help="Discrete recipe name")
    parser.add_argument("--finetune-tag", default=None)
    parser.add_argument("--stages", nargs="+", default=None)
    parser.add_argument("--eval-model", default=None)
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        if args.mode == "discrete":
            from src.discrete.registry import load_registry
        else:
            from src.utils.mllm_registry import load_registry
        models = load_registry().get("models") or {}
        for name in sorted(models):
            entry = models[name]
            ckpt = (entry.get("checkpoints") or {}).get(entry.get("use_checkpoint", "finetune"), "")
            recipe = entry.get("recipe") or entry.get("mllm_recipe")
            print(f"{name}\t{recipe}\t{ckpt}")
        return 0

    config_path = args.config or MODE_CONFIG[args.mode]
    if not os.path.isfile(config_path):
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    if args.stages is None:
        from src.utils.config import resolve_mode_runtime

        runtime_cfg, _ = resolve_mode_runtime(config_path, args.mode)
        args.stages = runtime_cfg.get("stages")

    return run_pipeline(
        config_path=config_path,
        train_config_path=args.train_config,
        stages_override=args.stages,
        eval_model=args.eval_model,
        mode=args.mode,
        recipe_override=args.recipe,
        finetune_tag_override=args.finetune_tag,
    )


if __name__ == "__main__":
    sys.exit(main())
