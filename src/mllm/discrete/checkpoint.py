"""Checkpoint helpers for VTB-Discrete (shared layout with continuous)."""

from __future__ import annotations

from vision_encoder_eval.mllm.utils.checkpoint_layout import (
    checkpoint_stage_root,
    finalize_stage_checkpoint,
    prepare_stage_output_dir,
    resolve_latest_stage_dir,
    rotate_stage_backup,
    stage_has_artifacts,
)

__all__ = [
    "checkpoint_stage_root",
    "checkpoint_stage_has_content",
    "finalize_stage_checkpoint",
    "prepare_stage_output_dir",
    "resolve_latest_stage_dir",
    "rotate_stage_backup",
    "stage_has_artifacts",
]


def checkpoint_stage_has_content(checkpoints_base: str, slug: str, stage: str) -> bool:
    return stage_has_artifacts(checkpoint_stage_root(checkpoints_base, slug, stage))
