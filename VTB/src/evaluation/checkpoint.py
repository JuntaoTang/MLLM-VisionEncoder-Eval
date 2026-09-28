"""Resolve VTB training checkpoints for evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass


HF_WEIGHT_MARKERS = ("model.safetensors", "pytorch_model.bin")
# Sharded HF saves (large LLM / bigG finetune) use an index instead of a single weight file.
HF_SHARD_INDEX_MARKERS = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
ADAPTER_MARKER = "mm_projector.bin"


@dataclass(frozen=True)
class ResolvedCheckpoint:
    path: str
    model_base: str | None = None
    kind: str = "full"  # full | adapter


def _dir_has(path: str, marker: str) -> bool:
    return os.path.isfile(os.path.join(path, marker))


def _dir_has_full_hf_weights(path: str) -> bool:
    if any(_dir_has(path, m) for m in HF_WEIGHT_MARKERS):
        return True
    return any(_dir_has(path, m) for m in HF_SHARD_INDEX_MARKERS)


def _find_latest_subdir(root: str, prefix: str = "checkpoint-") -> str | None:
    if not os.path.isdir(root):
        return None
    ckpts: list[tuple[int, str]] = []
    for name in os.listdir(root):
        if not name.startswith(prefix):
            continue
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            step = int(name.split("-", 1)[1])
        except (IndexError, ValueError):
            step = -1
        ckpts.append((step, path))
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


def resolve_checkpoint_dir(checkpoint_dir: str) -> ResolvedCheckpoint:
    """Resolve a checkpoint directory to a loadable path."""
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    if _dir_has_full_hf_weights(checkpoint_dir):
        return ResolvedCheckpoint(path=checkpoint_dir, kind="full")

    latest = _find_latest_subdir(checkpoint_dir)
    if latest and _dir_has_full_hf_weights(latest):
        return ResolvedCheckpoint(path=latest, kind="full")

    adapter_dir = checkpoint_dir
    if not _dir_has(adapter_dir, ADAPTER_MARKER):
        adapter_dir = latest or adapter_dir
    if _dir_has(adapter_dir, ADAPTER_MARKER) and os.path.isfile(os.path.join(adapter_dir, "config.json")):
        return ResolvedCheckpoint(path=adapter_dir, kind="adapter")

    raise FileNotFoundError(
        f"No eval checkpoint found under {checkpoint_dir}. "
        "Expected model.safetensors / model.safetensors.index.json / pytorch_model.bin "
        "or mm_projector.bin + config.json."
    )


def pick_eval_checkpoint_dir(ctx, llm_path: str) -> ResolvedCheckpoint:
    """Pick checkpoint from eval.use_checkpoint with finetune -> pretrain fallback."""
    use_ckpt = ctx.eval.get("use_checkpoint", "finetune")
    if use_ckpt == "pretrain":
        candidates = [ctx.pretrain_dir]
    elif use_ckpt == "finetune":
        candidates = [ctx.finetune_dir, ctx.pretrain_dir]
    else:
        candidates = [str(use_ckpt)]

    last_error: Exception | None = None
    for idx, directory in enumerate(candidates):
        try:
            resolved = resolve_checkpoint_dir(directory)
            if idx > 0:
                print(f"Warning: using fallback checkpoint from {directory} ({resolved.kind})")
            return resolved
        except FileNotFoundError as exc:
            last_error = exc
            continue

    raise FileNotFoundError(
        f"No checkpoint available for eval (use_checkpoint={use_ckpt!r}). "
        f"Tried: {', '.join(candidates)}"
    ) from last_error
