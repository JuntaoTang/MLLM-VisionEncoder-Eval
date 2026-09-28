"""Checkpoint layout under /cache/ckpt/trained/{mode}/{slug}/{stage}/.

Active weights live directly in the stage directory. Re-training rotates the
previous active checkpoint into ``backup/`` (only one backup is kept).
"""

from __future__ import annotations

import os
import shutil

_CHECKPOINT_MARKERS = (
    "mm_projector.bin",
    "model.safetensors",
    "pytorch_model.bin",
    "trainer_state.json",
    "arch_config.json",
)
_CHECKPOINT_DIR_PREFIX = "checkpoint-"
_BACKUP_DIRNAME = "backup"
_TRAINING_SCRATCH = (".tmp_train",)


def checkpoint_stage_root(checkpoints_base: str, slug: str, stage: str) -> str:
    return os.path.join(checkpoints_base, slug, stage)


def _is_marker_file(path: str, name: str) -> bool:
    return os.path.isfile(os.path.join(path, name))


def _dir_has_marker(path: str) -> bool:
    return any(_is_marker_file(path, marker) for marker in _CHECKPOINT_MARKERS)


def _list_active_entries(stage_dir: str) -> list[str]:
    if not os.path.isdir(stage_dir):
        return []
    return [
        name
        for name in os.listdir(stage_dir)
        if name not in (_BACKUP_DIRNAME, *_TRAINING_SCRATCH)
    ]


def stage_has_artifacts(stage_dir: str) -> bool:
    """True when the active stage dir (excluding backup/) has checkpoint files."""
    if not os.path.isdir(stage_dir):
        return False
    if _dir_has_marker(stage_dir):
        return True
    for name in _list_active_entries(stage_dir):
        path = os.path.join(stage_dir, name)
        if name.startswith(_CHECKPOINT_DIR_PREFIX) and os.path.isdir(path):
            return True
        if name == "saved_weights" and os.path.isdir(path):
            return True
    return False


def resolve_latest_stage_dir(checkpoints_base: str, slug: str, stage: str) -> str:
    """Return the active checkpoint directory (never ``backup/``)."""
    stage_dir = checkpoint_stage_root(checkpoints_base, slug, stage)
    os.makedirs(stage_dir, exist_ok=True)
    return stage_dir


def _find_latest_checkpoint_subdir(stage_dir: str) -> str | None:
    best_step = -1
    best_path: str | None = None
    for name in _list_active_entries(stage_dir):
        if not name.startswith(_CHECKPOINT_DIR_PREFIX):
            continue
        path = os.path.join(stage_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            step = int(name.split("-", 1)[1])
        except (IndexError, ValueError):
            step = -1
        if step >= best_step:
            best_step = step
            best_path = path
    return best_path


def _promote_dir_contents(src: str, dest: str) -> None:
    for name in os.listdir(src):
        src_path = os.path.join(src, name)
        dest_path = os.path.join(dest, name)
        if os.path.exists(dest_path):
            if os.path.isdir(dest_path):
                shutil.rmtree(dest_path)
            else:
                os.remove(dest_path)
        shutil.move(src_path, dest_path)


def finalize_stage_checkpoint(stage_dir: str) -> None:
    """Promote final weights to stage root and remove intermediate checkpoint-* dirs."""
    if not os.path.isdir(stage_dir):
        return

    if not _dir_has_marker(stage_dir):
        latest = _find_latest_checkpoint_subdir(stage_dir)
        if latest and _dir_has_marker(latest):
            _promote_dir_contents(latest, stage_dir)

    for name in list(_list_active_entries(stage_dir)):
        if name.startswith(_CHECKPOINT_DIR_PREFIX) or name in ("saved_weights",):
            path = os.path.join(stage_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)


def rotate_stage_backup(stage_dir: str) -> None:
    """Move the current active checkpoint into ``backup/`` (single-slot)."""
    if not stage_has_artifacts(stage_dir):
        return

    backup_dir = os.path.join(stage_dir, _BACKUP_DIRNAME)
    if os.path.isdir(backup_dir):
        shutil.rmtree(backup_dir)

    os.makedirs(backup_dir, exist_ok=True)
    for name in _list_active_entries(stage_dir):
        shutil.move(os.path.join(stage_dir, name), os.path.join(backup_dir, name))
    print(f"Checkpoint backup: {stage_dir} -> {backup_dir}")


def _resolve_path(path: str | None) -> str | None:
    if not path:
        return None
    if not os.path.isabs(path):
        from src.utils.config import VTB_ROOT

        path = os.path.join(VTB_ROOT, path)
    return os.path.abspath(path)


def _output_dir_from_resume(resume_path: str) -> str:
    base = os.path.basename(resume_path.rstrip("/"))
    if base.startswith(_CHECKPOINT_DIR_PREFIX):
        return os.path.dirname(resume_path)
    if stage_has_artifacts(resume_path):
        return resume_path
    parent = os.path.dirname(resume_path)
    if stage_has_artifacts(parent):
        return parent
    return resume_path


def prepare_stage_output_dir(ctx, stage: str) -> str:
    """Prepare flat stage output dir with optional backup rotation."""
    if stage not in ("pretrain", "finetune"):
        raise ValueError(f"Not a training stage: {stage}")

    base = ctx.checkpoint_base
    slug = ctx.run_slug
    stage_dir = checkpoint_stage_root(base, slug, stage)
    os.makedirs(stage_dir, exist_ok=True)

    resume_key = f"{stage}_resume"
    resume_raw = ctx.checkpoints.get(resume_key)
    if resume_raw:
        resume_path = _resolve_path(str(resume_raw))
        if not resume_path:
            raise ValueError(f"{resume_key} is set but path could not be resolved")
        out_dir = _output_dir_from_resume(resume_path)
        if stage == "pretrain":
            ctx.pretrain_dir = out_dir
        else:
            ctx.finetune_dir = out_dir
        print(f"Checkpoint {stage}: resume in place -> {out_dir}")
        return out_dir

    rotate_stage_backup(stage_dir)

    if stage == "pretrain":
        ctx.pretrain_dir = stage_dir
    else:
        ctx.finetune_dir = stage_dir
        pretrain_dir = resolve_latest_stage_dir(base, slug, "pretrain")
        if stage_has_artifacts(pretrain_dir):
            ctx.pretrain_dir = pretrain_dir

    print(f"Checkpoint {stage}: training -> {stage_dir}")
    return stage_dir
