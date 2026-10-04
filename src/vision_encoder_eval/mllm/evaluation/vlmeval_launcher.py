"""Launch VLMEvalKit with VTB-Discrete patches."""

from __future__ import annotations
from vision_encoder_eval.core.runtime import mllm_root

import importlib.util
import json
import os
import sys

VTB_ROOT = mllm_root()
VLMEVAL_ROOT = os.path.join(VTB_ROOT, "third_party", "VLMEvalKit")
if not os.path.isfile(os.path.join(VLMEVAL_ROOT, "run.py")):
    raise RuntimeError(f"VLMEvalKit not found at {VLMEVAL_ROOT}")

for path in (VTB_ROOT, VLMEVAL_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def _config_uses_discrete_vlm() -> bool:
    try:
        idx = sys.argv.index("--config")
        cfg_path = sys.argv[idx + 1]
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        for entry in cfg.get("model", {}).values():
            if entry.get("class") == "VTB_Discrete_VLM":
                return True
    except (ValueError, IndexError, OSError, json.JSONDecodeError):
        pass
    return False


def _register_discrete_vlm() -> None:
    if not _config_uses_discrete_vlm():
        return
    import vlmeval.vlm as vlm_module
    from vision_encoder_eval.mllm.evaluation.discrete_vlm import VTB_Discrete_VLM

    vlm_module.VTB_Discrete_VLM = VTB_Discrete_VLM


def _load_vlmeval_run():
    run_path = os.path.join(VLMEVAL_ROOT, "run.py")
    spec = importlib.util.spec_from_file_location("vlmeval_run", run_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load VLMEvalKit run.py from {run_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_vlmeval_dataset_builder(vlmeval_run) -> None:
    import copy as cp

    from vlmeval.dataset import build_dataset
    from vlmeval.dataset.video_dataset_config import supported_video_datasets

    original = vlmeval_run.build_dataset_from_config

    def build_dataset_from_config(cfg, dataset_name):
        config = cp.deepcopy(cfg[dataset_name])
        if config == {}:
            if dataset_name in supported_video_datasets:
                return supported_video_datasets[dataset_name]()
            return build_dataset(dataset_name)
        return original(cfg, dataset_name)

    vlmeval_run.build_dataset_from_config = build_dataset_from_config


def main() -> None:
    from vision_encoder_eval.mllm.evaluation.vlmeval_patches import (
        patch_decode_non_rgb_images,
        patch_lmudata_image_paths,
        patch_mmmu_result_transfer_id,
        patch_subsampled_tsv_md5,
    )

    os.environ.setdefault("VTB_ALLOW_SUBSAMPLED_TSV", "1")
    _register_discrete_vlm()
    os.chdir(VLMEVAL_ROOT)
    vlmeval_run = _load_vlmeval_run()
    _patch_vlmeval_dataset_builder(vlmeval_run)
    patch_subsampled_tsv_md5()
    patch_lmudata_image_paths()
    patch_mmmu_result_transfer_id(vlmeval_run)
    patch_decode_non_rgb_images()
    vlmeval_run.main()


if __name__ == "__main__":
    main()
