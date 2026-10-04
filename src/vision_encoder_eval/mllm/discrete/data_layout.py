"""Assemble VLMEvalKit LMUData view from split instructions/images trees."""

from __future__ import annotations

import os


def ensure_lmudata_view(
    data_root: str,
    *,
    instructions_dir: str | None = None,
    images_dir: str | None = None,
    view_dir: str | None = None,
) -> str:
    """Build data/.lmudata with symlinks expected by VLMEvalKit.

    Physical layout:
      instructions/test/*.tsv
      images/test/{dataset}/
    View layout:
      .lmudata/*.tsv -> instructions/test/
      .lmudata/images -> images/test/
    """
    import fcntl

    instructions_dir = instructions_dir or os.path.join(data_root, "instructions", "test")
    images_dir = images_dir or os.path.join(data_root, "images", "test")
    view_dir = view_dir or os.path.join(data_root, ".lmudata")

    if not os.path.isdir(instructions_dir):
        raise FileNotFoundError(f"Eval instructions dir not found: {instructions_dir}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Eval images dir not found: {images_dir}")

    os.makedirs(view_dir, exist_ok=True)
    lock_path = os.path.join(view_dir, ".vtb_lmudata.lock")
    with open(lock_path, "a+", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            for name in sorted(os.listdir(instructions_dir)):
                if not (name.endswith(".tsv") or name.endswith(".json")):
                    continue
                src = os.path.join(instructions_dir, name)
                dst = os.path.join(view_dir, name)
                _symlink_replace(src, dst)

            dst_images = os.path.join(view_dir, "images")
            _symlink_replace(images_dir, dst_images)

            # VLMEvalKit ids may differ in case from on-disk TSV names (GQA).
            try:
                from vision_encoder_eval.mllm.evaluation.dataset_config import (
                    _LMUDATA_TSV_ALIASES,
                    ensure_lmudata_dataset_alias,
                )

                for dataset_name in _LMUDATA_TSV_ALIASES:
                    ensure_lmudata_dataset_alias(view_dir, dataset_name)
            except Exception:
                pass
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
    return view_dir


def _symlink_replace(src: str, dst: str) -> None:
    """Create/replace a symlink; safe under concurrent eval workers."""
    src_abs = os.path.abspath(src)
    last_err: Exception | None = None
    for _ in range(5):
        try:
            if os.path.lexists(dst):
                if os.path.islink(dst) and os.path.realpath(dst) == src_abs:
                    return
                if os.path.isdir(dst) and not os.path.islink(dst):
                    return
                try:
                    os.remove(dst)
                except FileNotFoundError:
                    pass
            os.symlink(src_abs, dst)
            return
        except FileExistsError as exc:
            last_err = exc
            try:
                if os.path.islink(dst) and os.path.realpath(dst) == src_abs:
                    return
            except OSError:
                pass
        except FileNotFoundError as exc:
            # Another worker may have temporarily removed the path mid-race.
            last_err = exc
    try:
        if os.path.islink(dst) and os.path.realpath(dst) == src_abs:
            return
    except OSError:
        pass
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Failed to create symlink {dst} -> {src_abs}")
