#!/usr/bin/env python3
"""Launch VLMEvalKit with VTB_LLaVA registered. Called by src/evaluation/vlmeval_runner.py."""

from __future__ import annotations

import importlib.util
import os
import sys

VTB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAVA_ROOT = os.path.join(VTB_ROOT, "third_party", "LLaVA-NeXT")
VLMEVAL_ROOT = os.path.join(VTB_ROOT, "third_party", "VLMEvalKit")

# VTB + LLaVA on path for adapter imports; VLMEvalKit run.py loaded explicitly below.
for path in (VTB_ROOT, LLAVA_ROOT, VLMEVAL_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.utils.config import apply_cuda_stub_env, apply_offline_hf_env, install_cuda_stub_env, install_offline_hf_env

install_cuda_stub_env()
install_offline_hf_env()
apply_offline_hf_env()
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import vlmeval.vlm as vlm_module
from src.evaluation.vtb_vlm import VTB_LLaVA

vlm_module.VTB_LLaVA = VTB_LLaVA


def _load_vlmeval_run():
    run_path = os.path.join(VLMEVAL_ROOT, "run.py")
    spec = importlib.util.spec_from_file_location("vlmeval_run", run_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load VLMEvalKit run.py from {run_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_vlmeval_dataset_builder(vlmeval_run) -> None:
    """Empty data entries should use build_dataset(), not supported_video_datasets."""
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


def _patch_subsampled_tsv_md5() -> None:
    """Subsampled LMUData caches won't match upstream MD5; skip re-download."""
    if os.environ.get("VTB_ALLOW_SUBSAMPLED_TSV") != "1":
        return
    from vlmeval.dataset.image_base import ImageBaseDataset

    if getattr(ImageBaseDataset, "_vtb_md5_patched", False):
        return

    original = ImageBaseDataset.prepare_tsv

    def prepare_tsv(self, url, file_md5=None):
        return original(self, url, file_md5=None)

    ImageBaseDataset.prepare_tsv = prepare_tsv
    ImageBaseDataset._vtb_md5_patched = True


if __name__ == "__main__":
    from src.evaluation.vlmeval_patches import (
        patch_decode_non_rgb_images,
        patch_lmudata_image_paths,
        patch_mmmu_result_transfer_id,
        patch_subsampled_tsv_md5,
    )

    os.environ.setdefault("VTB_ALLOW_SUBSAMPLED_TSV", "1")
    os.chdir(VLMEVAL_ROOT)
    vlmeval_run = _load_vlmeval_run()
    _patch_vlmeval_dataset_builder(vlmeval_run)
    # Always patch: scoring & path-based local TSVs must never be re-downloaded.
    patch_subsampled_tsv_md5()
    patch_lmudata_image_paths()
    patch_mmmu_result_transfer_id(vlmeval_run)
    patch_decode_non_rgb_images()
    vlmeval_run.main()
