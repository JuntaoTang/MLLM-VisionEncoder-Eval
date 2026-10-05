#!/usr/bin/env python3
"""Paper protocol: cached ImageNet-1K 5-shot BN linear probing.

Every encoder uses the same deterministic five-example-per-class support set,
one epoch (five updates), and a non-affine BatchNorm1d -> Linear head. Frozen
features are extracted once and cached. Evaluation uses the fixed 5,000-image
validation subset sampled with seed 42.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
import os
from pathlib import Path
import sys
from vision_encoder_eval.core.runtime import asset_path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
MINI_ROOT = Path(asset_path('runtime'))
CACHED_PATH = SCRIPT_DIR / "linear_probe_cached.py"
MANIFEST_PATH = SCRIPT_DIR / "tokenizers.tsv"
DEFAULT_OUTPUT_ROOT = MINI_ROOT / "output" / "linear_probing_runs"
DEFAULT_CACHE_ROOT = DEFAULT_OUTPUT_ROOT / "_feature_cache"
FIVE_SHOT_CACHE_ROOT = DEFAULT_CACHE_ROOT

PROTOCOL_VERSION = "imagenet_5shot_noaug_cached_1epoch_allbn_v1"
CACHE_VERSION = "imagenet_cap100_deterministic_features_v1"
NUM_CLASSES = 1_000
BATCH_SIZE = 1_000
EPOCHS = 1
DEFAULT_CAP_SHOTS = (5,)

# Validated frozen-encoder chunk sizes. These only affect memory/throughput;
# the optimization batch is always 1,000.
FEATURE_MICROBATCH_SIZES = {
    "clip_openai__l14": 256,
    "dinov1_vitb16": 512,
    "dinov1_vitb8": 256,
    "dinov1_vits16": 1000,
    "dinov1_vits8": 512,
    "dinov2_base": 512,
    "dinov2_giant": 64,
    "dinov2_large": 256,
    "dinov2_small": 1000,
    "dinov3_vitl16_lvd1689m": 256,
    "eupe_convnext_b": 512,
    "eupe_vit_b": 512,
    "eupe_vit_s": 1000,
    "eupe_vit_t": 1000,
    "ijepa": 128,
    "mc1_b16_224_2.5b": 512,
    "mc1_b16_224_400m": 512,
    "mc1_b32_224_2.5b": 512,
    "mc1_b32_224_400m": 512,
    "mc1_g14_224_2.5b": 64,
    "mc1_h14_224_2.5b": 128,
    "mc1_h14_224_v1.2": 128,
    "mc1_l14_224_2.5b": 256,
    "mc1_l14_224_400m": 256,
    "mc2_b16_224": 512,
    "mc2_b16_384": 256,
    "mc2_b32_224": 1024,
    "mc2_b32_224_mt5": 1024,
    "mc2_b32_384": 512,
    "mc2_g14_224": 64,
    "mc2_g14_378": 32,
    "mc2_h14_378": 32,
    "mc2_l14_224": 256,
    "mc2_m16_224": 1024,
    "mc2_m16_224_mt5": 1024,
    "mc2_m16_384": 512,
    "mc2_s16_224": 1000,
    "mc2_s16_224_mt5": 1024,
    "mc2_s16_384": 512,
    "pe_core_b16_224": 512,
    "pe_core_g14_448": 16,
    "pe_lang_l14_448": 32,
    "pixio_vitb16": 256,
    "pixio_vith16": 64,
    "pixio_vitl16": 128,
    "raev2_dinov3l_k7": 128,
    "siglip2_b16_224": 512,
    "siglip2_b16_256": 512,
    "siglip2_b16_384": 256,
    "siglip2_b16_512": 128,
    "siglip2_b32_256": 1024,
    "siglip2_g16_256": 32,
    "siglip2_g16_384": 16,
    "siglip2_l16_256": 256,
    "siglip2_l16_384": 128,
    "siglip2_l16_512": 64,
    "siglip2_sm14_224": 128,
    "siglip2_sm14_384": 64,
    "siglip2_sm16_256": 128,
    "siglip2_sm16_384": 64,
    "siglip2_sm16_512": 32,
    "toklip_l": 256,
    "toklip_s": 256,
    "uniar_bsq": 1024,
    "unitok": 256,
    "vilau": 256,
    "webssl_dino1b_full2b_224": 32,
    "webssl_mae1b_full2b_224": 32,
    "webssl_mae300m_full2b_224": 128,
    "webssl_mae3b_full2b_224": 16,
}


def _load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cached = _load_module("flop_sweep_cached_base", CACHED_PATH)
base = cached.base


def _load_manifest() -> tuple[tuple[int, str, str], ...]:
    rows = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            rank, requested_id, model, _head = line.rstrip("\n").split("\t")
            rows.append((int(rank), requested_id, model))
    if len(rows) != 70 or [row[0] for row in rows] != list(range(1, 71)):
        raise RuntimeError("Expected the canonical ranks 1..70 in the manifest")
    return tuple(rows)


MANIFEST = _load_manifest()
MODEL_TO_ROW = {model: (rank, requested_id) for rank, requested_id, model in MANIFEST}


class RangedIndexedDataset(Dataset):
    """Expose original dataset indices for resumable sequential extraction."""

    def __init__(self, dataset, start: int = 0):
        self.dataset = dataset
        self.start = int(start)

    def __len__(self) -> int:
        return len(self.dataset) - self.start

    def __getitem__(self, local_index: int):
        index = self.start + int(local_index)
        image, label = self.dataset[index]
        return index, image, label


class SupportDataset(Dataset):
    """Compact view of deterministic source indices from the official split."""

    def __init__(self, dataset, source_indices: np.ndarray):
        self.dataset = dataset
        self.source_indices = np.asarray(source_indices, dtype=np.int64)
        targets = np.asarray(dataset.get_targets(), dtype=np.int64)
        self._targets = targets[self.source_indices]

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, index: int):
        return self.dataset[int(self.source_indices[index])]

    def get_targets(self) -> np.ndarray:
        return self._targets


class PositionedSubsetDataset(Dataset):
    """Select positions from a compact dataset while retaining destination indices."""

    def __init__(self, dataset, positions: np.ndarray, start: int = 0):
        self.dataset = dataset
        self.positions = np.asarray(positions, dtype=np.int64)
        self.start = int(start)

    def __len__(self) -> int:
        return len(self.positions) - self.start

    def __getitem__(self, local_index: int):
        position = int(self.positions[self.start + int(local_index)])
        image, label = self.dataset[position]
        return position, image, label


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _sha256_indices(indices: np.ndarray) -> str:
    stable = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(stable.tobytes(order="C")).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = cached._build_parser()
    parser.description = __doc__
    model_action = next(action for action in parser._actions if action.dest == "model")
    model_action.choices = tuple(MODEL_TO_ROW)
    epoch_action = next(
        action for action in parser._actions if action.dest == "stop_after_epoch"
    )
    epoch_action.help = "Fixed to 1 by the paper protocol."
    parser.set_defaults(
        output_root=str(DEFAULT_OUTPUT_ROOT),
        cache_root=str(DEFAULT_CACHE_ROOT),
        cache_batch_size=None,
        feature_microbatch_size=None,
        stop_after_epoch=EPOCHS,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--support-seed", type=int, default=0)
    parser.add_argument(
        "--cap-shots",
        default=",".join(str(value) for value in DEFAULT_CAP_SHOTS),
        help="Fixed to 5 by the paper protocol.",
    )
    parser.add_argument(
        "--five-shot-cache-root",
        default=str(FIVE_SHOT_CACHE_ROOT),
        help="Search this cache for a compatible validation feature array.",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=5_000,
        help="Fixed to 5000 by the paper protocol.",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=42,
        help="Seed for uniform validation sampling without replacement (default: 42).",
    )
    return parser


def _parse_cap_shots(text: str) -> list[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("All --cap-shots values must be positive")
        values.append(value)
    if not values:
        raise ValueError("--cap-shots must contain at least one positive integer")
    return sorted(set(values))


def _full_cache_identity(args, bundle, data_root: Path, extra_root: Path) -> dict:
    identity = cached._cache_identity(args, bundle, data_root, extra_root)
    identity.update(
        {
            "version": "imagenet_full_train_deterministic_features_v1",
            "train_split": "complete official ImageNet-1k train split",
            "train_order": "official dataset index order",
        }
    )
    return identity


def _compact_cache_identity(
    args,
    bundle,
    data_root: Path,
    extra_root: Path,
    source_indices: np.ndarray,
    max_cap_shot: int,
) -> dict:
    identity = cached._cache_identity(args, bundle, data_root, extra_root)
    identity.update(
        {
            "version": CACHE_VERSION,
            "train_split": "nested deterministic class-balanced support subset",
            "train_order": "sorted official dataset source indices",
            "max_cap_shot": int(max_cap_shot),
            "support_seed": int(args.support_seed),
            "support_samples": int(len(source_indices)),
            "support_indices_sha256": _sha256_indices(source_indices),
        }
    )
    return identity


def _same_feature_surface(candidate: dict, identity: dict) -> bool:
    ignored = {"version", "train_split", "train_order", "max_cap_shot"}
    ignored.update(
        {
            "support_seed",
            "support_indices_sha256",
            "support_samples",
        }
    )
    candidate_core = {key: value for key, value in candidate.items() if key not in ignored}
    identity_core = {key: value for key, value in identity.items() if key not in ignored}
    return candidate_core == identity_core


def _find_reusable_full_train_cache(
    model_cache_root: Path,
    identity: dict,
    official_train_samples: int,
):
    if not model_cache_root.is_dir():
        return None
    for metadata_path in model_cache_root.glob("*/metadata.json"):
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if not _same_feature_surface(metadata.get("identity", {}), identity):
                continue
            train_meta = metadata["splits"]["train"]
            val_meta = metadata["splits"]["val"]
            if int(train_meta["feature_shape"][0]) != official_train_samples:
                continue
            paths = [
                Path(train_meta["features_path"]),
                Path(train_meta["labels_path"]),
                Path(val_meta["features_path"]),
                Path(val_meta["labels_path"]),
            ]
            if all(path.is_file() for path in paths):
                return metadata
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
    return None


def _find_largest_reusable_compact_cache(
    model_cache_root: Path,
    identity: dict,
    target_source_indices: np.ndarray,
):
    """Find the largest compatible compact cache nested inside the target cache."""
    if not model_cache_root.is_dir():
        return None
    target_source_indices = np.asarray(target_source_indices, dtype=np.int64)
    candidates = []
    for metadata_path in model_cache_root.glob("*/metadata.json"):
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if not _same_feature_surface(metadata.get("identity", {}), identity):
                continue
            train_meta = metadata["splits"]["train"]
            source_path_text = train_meta.get("source_indices_path")
            if not source_path_text:
                continue
            source_path = Path(source_path_text)
            features_path = Path(train_meta["features_path"])
            labels_path = Path(train_meta["labels_path"])
            if not all(path.is_file() for path in (source_path, features_path, labels_path)):
                continue
            source_indices = np.load(source_path, mmap_mode="r")
            if not 0 < len(source_indices) < len(target_source_indices):
                continue
            if np.any(source_indices[1:] <= source_indices[:-1]):
                continue
            target_positions = np.searchsorted(target_source_indices, source_indices)
            if (
                np.any(target_positions >= len(target_source_indices))
                or not np.array_equal(
                    np.asarray(target_source_indices[target_positions]), source_indices
                )
            ):
                continue
            features = np.load(features_path, mmap_mode="r")
            labels = np.load(labels_path, mmap_mode="r")
            if (
                features.dtype != np.float32
                or labels.dtype != np.int64
                or len(features) != len(source_indices)
                or labels.shape != (len(source_indices),)
            ):
                continue
            candidates.append((len(source_indices), metadata_path, metadata))
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
    if not candidates:
        return None
    _count, metadata_path, metadata = max(candidates, key=lambda item: item[0])
    return metadata_path, metadata


def _find_reusable_validation_cache(root: Path, identity: dict):
    if not root.is_dir():
        return None
    for metadata_path in root.glob("*/*/metadata.json"):
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if not _same_feature_surface(metadata.get("identity", {}), identity):
                continue
            split = metadata["splits"]["val"]
            features_path = metadata_path.parent / "val_features.npy"
            labels_path = metadata_path.parent / "val_labels.npy"
            if not features_path.is_file() or not labels_path.is_file():
                continue
            features = np.load(features_path, mmap_mode="r")
            labels = np.load(labels_path, mmap_mode="r")
            if (
                list(features.shape) == split["feature_shape"]
                and list(labels.shape) == split["label_shape"]
                and features.dtype == np.float32
                and labels.dtype == np.int64
            ):
                return {
                    "features_path": str(features_path.resolve()),
                    "labels_path": str(labels_path.resolve()),
                    "feature_shape": list(features.shape),
                    "label_shape": list(labels.shape),
                    "reused_from": str(metadata_path.resolve()),
                }
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
    return None


@torch.no_grad()
def _extract_split_resumable(
    *,
    split: str,
    dataset,
    feature_model,
    feature_dim: int,
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
    progress_label: str,
) -> dict:
    final_features = cache_dir / f"{split}_features.npy"
    final_labels = cache_dir / f"{split}_labels.npy"
    partial_features = cache_dir / f".{split}_features.partial.npy"
    partial_labels = cache_dir / f".{split}_labels.partial.npy"
    progress_path = cache_dir / f".{split}_progress.json"
    expected_shape = (len(dataset), feature_dim)

    if final_features.is_file() and final_labels.is_file():
        features = np.load(final_features, mmap_mode="r")
        labels = np.load(final_labels, mmap_mode="r")
        if features.shape == expected_shape and labels.shape == (len(dataset),):
            if features.dtype == np.float32 and labels.dtype == np.int64:
                return {
                    "features_path": str(final_features.resolve()),
                    "labels_path": str(final_labels.resolve()),
                    "feature_shape": list(features.shape),
                    "label_shape": list(labels.shape),
                    "reused_from": None,
                }

    completed = 0
    if partial_features.is_file() and partial_labels.is_file() and progress_path.is_file():
        with progress_path.open("r", encoding="utf-8") as handle:
            progress = json.load(handle)
        if progress.get("shape") == list(expected_shape):
            completed = int(progress.get("completed", 0))
    if completed < 0 or completed > len(dataset):
        completed = 0

    if completed == 0:
        feature_array = np.lib.format.open_memmap(
            partial_features, mode="w+", dtype=np.float32, shape=expected_shape
        )
        label_array = np.lib.format.open_memmap(
            partial_labels, mode="w+", dtype=np.int64, shape=(len(dataset),)
        )
        _atomic_json(progress_path, {"completed": 0, "shape": list(expected_shape)})
    else:
        feature_array = np.load(partial_features, mmap_mode="r+")
        label_array = np.load(partial_labels, mmap_mode="r+")
        if feature_array.shape != expected_shape or label_array.shape != (len(dataset),):
            raise RuntimeError(f"Invalid resumable {split} arrays in {cache_dir}")

    remaining_dataset = RangedIndexedDataset(dataset, completed)
    loader = DataLoader(
        remaining_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    total_batches = math.ceil(len(dataset) / batch_size)
    initial_batches = completed // batch_size
    progress_bar = tqdm(
        loader,
        total=total_batches,
        initial=initial_batches,
        desc=f"{progress_label} cache-{split}",
        dynamic_ncols=True,
        unit="batch",
    )
    for batch_number, (indices, images, labels) in enumerate(progress_bar, start=1):
        batch_features = feature_model(images).detach().cpu().numpy()
        numpy_indices = indices.numpy()
        if batch_features.dtype != np.float32:
            raise RuntimeError(f"Expected FP32 features, got {batch_features.dtype}")
        if batch_features.shape != (len(numpy_indices), feature_dim):
            raise RuntimeError(f"Unexpected {split} feature shape {batch_features.shape}")
        if not np.isfinite(batch_features).all():
            raise RuntimeError(f"Non-finite {split} feature batch")
        feature_array[numpy_indices] = batch_features
        label_array[numpy_indices] = labels.numpy()
        completed = int(numpy_indices[-1]) + 1
        if batch_number % 25 == 0 or completed == len(dataset):
            feature_array.flush()
            label_array.flush()
            _atomic_json(
                progress_path,
                {"completed": completed, "shape": list(expected_shape)},
            )

    expected_labels = np.asarray(dataset.get_targets(), dtype=np.int64)
    if not np.array_equal(label_array, expected_labels):
        raise RuntimeError(f"Cached {split} labels do not match ImageNet targets")
    feature_array.flush()
    label_array.flush()
    del feature_array, label_array, loader, progress_bar
    os.replace(partial_features, final_features)
    os.replace(partial_labels, final_labels)
    progress_path.unlink(missing_ok=True)
    return {
        "features_path": str(final_features.resolve()),
        "labels_path": str(final_labels.resolve()),
        "feature_shape": list(expected_shape),
        "label_shape": [len(dataset)],
        "reused_from": None,
    }


@torch.no_grad()
def _extract_train_by_extending_compact_cache(
    *,
    dataset,
    feature_model,
    feature_dim: int,
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
    progress_label: str,
    target_source_indices: np.ndarray,
    seed_metadata_path: Path,
    seed_metadata: dict,
) -> dict:
    """Build a larger compact cache by copying a nested cache and encoding only new rows."""
    final_features = cache_dir / "train_features.npy"
    final_labels = cache_dir / "train_labels.npy"
    partial_features = cache_dir / ".train_features.partial.npy"
    partial_labels = cache_dir / ".train_labels.partial.npy"
    progress_path = cache_dir / ".train_extension_progress.json"
    expected_shape = (len(dataset), feature_dim)

    if final_features.is_file() and final_labels.is_file():
        features = np.load(final_features, mmap_mode="r")
        labels = np.load(final_labels, mmap_mode="r")
        if (
            features.shape == expected_shape
            and labels.shape == (len(dataset),)
            and features.dtype == np.float32
            and labels.dtype == np.int64
        ):
            return {
                "features_path": str(final_features.resolve()),
                "labels_path": str(final_labels.resolve()),
                "feature_shape": list(features.shape),
                "label_shape": list(labels.shape),
                "reused_from": str(seed_metadata_path.resolve()),
            }

    seed_train = seed_metadata["splits"]["train"]
    seed_features = np.load(seed_train["features_path"], mmap_mode="r")
    seed_labels = np.load(seed_train["labels_path"], mmap_mode="r")
    seed_sources = np.load(seed_train["source_indices_path"], mmap_mode="r")
    target_sources = np.asarray(target_source_indices, dtype=np.int64)
    seed_positions = np.searchsorted(target_sources, seed_sources)
    if (
        seed_features.shape[1:] != (feature_dim,)
        or len(seed_features) != len(seed_sources)
        or not np.array_equal(np.asarray(target_sources[seed_positions]), seed_sources)
    ):
        raise RuntimeError("Compact seed cache is incompatible with the requested extension")
    expected_labels = np.asarray(dataset.get_targets(), dtype=np.int64)
    if not np.array_equal(seed_labels, expected_labels[seed_positions]):
        raise RuntimeError("Compact seed labels do not match the extended support set")
    missing_mask = np.ones(len(dataset), dtype=bool)
    missing_mask[seed_positions] = False
    missing_positions = np.flatnonzero(missing_mask).astype(np.int64, copy=False)
    seed_fingerprint = str(seed_metadata.get("fingerprint", ""))

    completed = 0
    if partial_features.is_file() and partial_labels.is_file() and progress_path.is_file():
        with progress_path.open("r", encoding="utf-8") as handle:
            progress = json.load(handle)
        if (
            progress.get("shape") == list(expected_shape)
            and progress.get("seed_fingerprint") == seed_fingerprint
            and int(progress.get("missing_count", -1)) == len(missing_positions)
        ):
            completed = int(progress.get("completed_missing", 0))
    if completed < 0 or completed > len(missing_positions):
        completed = 0

    if completed == 0:
        feature_array = np.lib.format.open_memmap(
            partial_features, mode="w+", dtype=np.float32, shape=expected_shape
        )
        label_array = np.lib.format.open_memmap(
            partial_labels, mode="w+", dtype=np.int64, shape=(len(dataset),)
        )
        for start in range(0, len(seed_positions), 8192):
            stop = min(start + 8192, len(seed_positions))
            feature_array[seed_positions[start:stop]] = seed_features[start:stop]
        label_array[:] = expected_labels
        feature_array.flush()
        label_array.flush()
        _atomic_json(
            progress_path,
            {
                "completed_missing": 0,
                "missing_count": int(len(missing_positions)),
                "seed_fingerprint": seed_fingerprint,
                "shape": list(expected_shape),
            },
        )
    else:
        feature_array = np.load(partial_features, mmap_mode="r+")
        label_array = np.load(partial_labels, mmap_mode="r+")
        if feature_array.shape != expected_shape or label_array.shape != (len(dataset),):
            raise RuntimeError(f"Invalid resumable train extension arrays in {cache_dir}")

    remaining_dataset = PositionedSubsetDataset(dataset, missing_positions, completed)
    loader = DataLoader(
        remaining_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    progress_bar = tqdm(
        loader,
        total=math.ceil(len(missing_positions) / batch_size),
        initial=completed // batch_size,
        desc=f"{progress_label} cache-train-extend",
        dynamic_ncols=True,
        unit="batch",
    )
    for batch_number, (positions, images, labels) in enumerate(progress_bar, start=1):
        batch_features = feature_model(images).detach().cpu().numpy()
        numpy_positions = positions.numpy()
        if batch_features.dtype != np.float32:
            raise RuntimeError(f"Expected FP32 features, got {batch_features.dtype}")
        if batch_features.shape != (len(numpy_positions), feature_dim):
            raise RuntimeError(f"Unexpected train feature shape {batch_features.shape}")
        if not np.isfinite(batch_features).all():
            raise RuntimeError("Non-finite train feature batch")
        feature_array[numpy_positions] = batch_features
        if not np.array_equal(labels.numpy(), expected_labels[numpy_positions]):
            raise RuntimeError("Extended train labels do not match ImageNet targets")
        completed += len(numpy_positions)
        if batch_number % 25 == 0 or completed == len(missing_positions):
            feature_array.flush()
            label_array.flush()
            _atomic_json(
                progress_path,
                {
                    "completed_missing": int(completed),
                    "missing_count": int(len(missing_positions)),
                    "seed_fingerprint": seed_fingerprint,
                    "shape": list(expected_shape),
                },
            )

    if not np.array_equal(label_array, expected_labels):
        raise RuntimeError("Extended train-cache labels do not match ImageNet targets")
    feature_array.flush()
    label_array.flush()
    del feature_array, label_array, loader, progress_bar
    os.replace(partial_features, final_features)
    os.replace(partial_labels, final_labels)
    progress_path.unlink(missing_ok=True)
    return {
        "features_path": str(final_features.resolve()),
        "labels_path": str(final_labels.resolve()),
        "feature_shape": list(expected_shape),
        "label_shape": [len(dataset)],
        "reused_from": str(seed_metadata_path.resolve()),
        "seeded_train_samples": int(len(seed_sources)),
        "newly_encoded_train_samples": int(len(missing_positions)),
    }


def _prepare_cache(
    *,
    args,
    identity: dict,
    bundle,
    feature_model,
    train_dataset,
    val_dataset,
    cache_dir: Path,
    progress_label: str,
    train_source_indices: np.ndarray | None = None,
) -> dict:
    metadata_path = cache_dir / "metadata.json"
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("identity") == identity:
            valid = True
            for split in ("train", "val"):
                split_meta = metadata["splits"][split]
                features = Path(split_meta["features_path"])
                labels = Path(split_meta["labels_path"])
                valid &= features.is_file() and labels.is_file()
            if train_source_indices is not None:
                source_path = metadata["splits"]["train"].get("source_indices_path")
                valid &= bool(source_path) and Path(source_path).is_file()
            if valid:
                return metadata

    sample = train_dataset[0][0].unsqueeze(0)
    feature_dim = base._validate_feature_model(feature_model, sample)
    cache_dir.mkdir(parents=True, exist_ok=True)
    compact_seed = None
    if train_source_indices is not None:
        compact_seed = _find_largest_reusable_compact_cache(
            cache_dir.parent, identity, train_source_indices
        )
    if compact_seed is None:
        train_meta = _extract_split_resumable(
            split="train",
            dataset=train_dataset,
            feature_model=feature_model,
            feature_dim=feature_dim,
            cache_dir=cache_dir,
            batch_size=args.cache_batch_size,
            num_workers=args.num_workers,
            progress_label=progress_label,
        )
    else:
        seed_metadata_path, seed_metadata = compact_seed
        print(
            f">> extend compact cache from {seed_metadata_path}; "
            f"encode only {len(train_dataset) - int(seed_metadata['splits']['train']['feature_shape'][0])} new images",
            flush=True,
        )
        train_meta = _extract_train_by_extending_compact_cache(
            dataset=train_dataset,
            feature_model=feature_model,
            feature_dim=feature_dim,
            cache_dir=cache_dir,
            batch_size=args.cache_batch_size,
            num_workers=args.num_workers,
            progress_label=progress_label,
            target_source_indices=train_source_indices,
            seed_metadata_path=seed_metadata_path,
            seed_metadata=seed_metadata,
        )
    val_meta = _find_reusable_validation_cache(
        Path(args.five_shot_cache_root).expanduser().resolve(), identity
    )
    if val_meta is None:
        val_meta = _extract_split_resumable(
            split="val",
            dataset=val_dataset,
            feature_model=feature_model,
            feature_dim=feature_dim,
            cache_dir=cache_dir,
            batch_size=args.cache_batch_size,
            num_workers=args.num_workers,
            progress_label=progress_label,
        )
    if train_meta["feature_shape"][1] != val_meta["feature_shape"][1]:
        raise RuntimeError("Train and validation feature dimensions differ")
    if train_source_indices is not None:
        source_path = cache_dir / "train_source_indices.npy"
        temporary = cache_dir / ".train_source_indices.npy.tmp"
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(train_source_indices, dtype=np.int64))
        os.replace(temporary, source_path)
        train_meta["source_indices_path"] = str(source_path.resolve())
    metadata = {
        "identity": identity,
        "fingerprint": cached._json_fingerprint(identity),
        "splits": {"train": train_meta, "val": val_meta},
    }
    _atomic_json(metadata_path, metadata)
    return metadata


def _nested_class_orders(targets: np.ndarray, seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    targets = np.asarray(targets, dtype=np.int64)
    if not np.array_equal(np.unique(targets), np.arange(NUM_CLASSES)):
        raise RuntimeError("Expected ImageNet labels 0..999")

    # Preserve the exact five selected examples from the earlier 5-shot run.
    legacy_rng = np.random.default_rng(seed)
    first_five = []
    candidates_by_class = []
    for class_id in range(NUM_CLASSES):
        candidates = np.flatnonzero(targets == class_id)
        if len(candidates) < 5:
            raise RuntimeError(f"Class {class_id} has fewer than five examples")
        candidates_by_class.append(candidates)
        first_five.append(legacy_rng.choice(candidates, 5, replace=False))

    orders = []
    for class_id, (candidates, prefix) in enumerate(zip(candidates_by_class, first_five)):
        remaining = candidates[~np.isin(candidates, prefix)]
        tail_rng = np.random.default_rng(np.random.SeedSequence([seed, class_id, 0x51D1D3]))
        tail = tail_rng.permutation(remaining)
        orders.append(np.concatenate((prefix, tail)).astype(np.int64, copy=False))
    counts = np.asarray([len(order) for order in orders], dtype=np.int64)
    return orders, counts


def _support_indices(orders: list[np.ndarray], cap_shot: int) -> np.ndarray:
    selected = np.concatenate([order[: min(cap_shot, len(order))] for order in orders])
    # Sorting matches the earlier SupportDataset representation and makes the
    # support hash independent of class concatenation order.
    return np.sort(selected.astype(np.int64, copy=False))


def _iter_batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


@torch.no_grad()
def _evaluate(head_grid, val_features, val_labels, device: torch.device) -> list[dict]:
    head_grid.eval()
    totals = {
        name: {"top1": 0, "top5": 0}
        for name in head_grid.heads
    }
    count = 0
    all_indices = np.arange(len(val_labels), dtype=np.int64)
    for batch_indices in _iter_batches(all_indices, 1024):
        features = torch.from_numpy(
            np.ascontiguousarray(val_features[batch_indices])
        ).to(device, non_blocking=True)
        labels = torch.from_numpy(
            np.ascontiguousarray(val_labels[batch_indices])
        ).to(device, non_blocking=True)
        logits_by_name = head_grid(features)
        for name, logits in logits_by_name.items():
            predictions = logits.topk(5, dim=1).indices
            totals[name]["top1"] += int((predictions[:, 0] == labels).sum().item())
            totals[name]["top5"] += int((predictions == labels[:, None]).any(dim=1).sum().item())
        count += len(batch_indices)
    classifiers = []
    for name, head in head_grid.heads.items():
        classifiers.append(
            {
                "name": name,
                "base_lr": head.base_lr,
                "effective_lr": head.effective_lr,
                "metrics": {
                    "top-1": 100.0 * totals[name]["top1"] / count,
                    "top-5": 100.0 * totals[name]["top5"] / count,
                },
            }
        )
    classifiers.sort(key=lambda item: item["base_lr"])
    return classifiers


def _save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _train_budget(
    *,
    args,
    output_dir: Path,
    metadata: dict,
    train_features,
    train_labels,
    val_features,
    val_labels,
    validation_source_indices: np.ndarray,
    physical_validation_samples: int,
    protocol_version: str,
    support_indices: np.ndarray,
    support_source_indices: np.ndarray,
    official_train_samples: int,
    cap_shot: int,
    min_class_count: int,
    max_class_count: int,
    progress_label: str,
    device: torch.device,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    class_counts = np.bincount(train_labels[support_indices], minlength=NUM_CLASSES)
    steps_per_epoch = math.ceil(len(support_indices) / BATCH_SIZE)
    total_updates = EPOCHS * steps_per_epoch
    effective_lrs = [
        float(value * BATCH_SIZE / 256.0) for value in base.BASE_LEARNING_RATES
    ]
    protocol = {
        "version": protocol_version,
        "model": args.model,
        "requested_id": MODEL_TO_ROW[args.model][1],
        "rank": MODEL_TO_ROW[args.model][0],
        "dataset": "ImageNet-1k",
        "train_split": "nested deterministic per-class prefix of official train",
        "validation_split": (
            "complete official validation split"
            if len(validation_source_indices) == physical_validation_samples
            else "fixed uniform subset of official validation split"
        ),
        "validation_sampling": (
            "complete split in official order"
            if len(validation_source_indices) == physical_validation_samples
            else "uniform without replacement; selected source indices sorted before evaluation"
        ),
        "validation_seed": (
            None
            if len(validation_source_indices) == physical_validation_samples
            else int(args.validation_seed)
        ),
        "validation_samples": int(len(validation_source_indices)),
        "validation_indices_sha256": _sha256_indices(validation_source_indices),
        "cap_shot": int(cap_shot),
        "actual_train_samples": int(len(support_indices)),
        "minimum_selected_per_class": int(class_counts.min()),
        "maximum_selected_per_class": int(class_counts.max()),
        "minimum_available_per_class": int(min_class_count),
        "maximum_available_per_class": int(max_class_count),
        "is_class_balanced": bool(np.all(class_counts == class_counts[0])),
        "is_full_train_split": bool(
            len(support_source_indices) == official_train_samples
        ),
        "support_seed": int(args.support_seed),
        "support_indices_sha256": _sha256_indices(support_source_indices),
        "reported_encoder_train_forwards": int(len(support_indices)),
        "reported_encoder_validation_forwards": int(len(validation_source_indices)),
        "physical_cache_encoder_train_forwards_per_model": int(len(train_labels)),
        "physical_cache_encoder_validation_forwards_per_model": int(physical_validation_samples),
        "flop_accounting": (
            "report cap-point encoder FLOPs as reported_encoder_train_forwards plus "
            "reported_encoder_validation_forwards; "
            + (
                "the complete cache is an amortized execution optimization"
                if len(train_labels) == official_train_samples
                else "the maximum requested support cache is shared by smaller nested caps"
            )
        ),
        "data_augmentation": False,
        "feature_cache_dtype": "float32",
        "feature_cache_fingerprint": metadata["fingerprint"],
        "probe_head": "BatchNorm1d(affine=False)->Linear for all models",
        "global_batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "max_updates": total_updates,
        "optimizer": "SGD(momentum=0.9, weight_decay=0)",
        "lr_schedule": "independent cosine-to-zero schedule per cap point",
        "base_learning_rates": list(base.BASE_LEARNING_RATES),
        "effective_learning_rates": effective_lrs,
        "seed": int(args.seed),
    }
    protocol["fingerprint"] = base._protocol_fingerprint(protocol)
    protocol_path = output_dir / "protocol.json"
    if protocol_path.is_file():
        with protocol_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous != protocol:
            raise RuntimeError(f"Protocol mismatch in {output_dir}")
    else:
        _atomic_json(protocol_path, protocol)

    result_path = output_dir / "results_eval_linear.json"
    if result_path.is_file():
        with result_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        if result.get("protocol_fingerprint") == protocol["fingerprint"]:
            return result
        raise RuntimeError(f"Result/protocol mismatch in {output_dir}")

    base._seed_everything(args.seed)
    feature_dim = int(train_features.shape[1])
    head_grid, parameter_groups, built_effective_lrs = base._build_heads(
        feature_dim, device, use_batch_norm=True
    )
    if built_effective_lrs != effective_lrs:
        raise RuntimeError("Constructed learning-rate grid differs from protocol")

    optimizer = torch.optim.SGD(parameter_groups, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_updates, eta_min=0.0
    )
    checkpoint_path = output_dir / "running_checkpoint.pt"
    start_epoch = 0
    if checkpoint_path.is_file() and not args.no_resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("protocol_fingerprint") != protocol["fingerprint"]:
            raise RuntimeError(f"Checkpoint/protocol mismatch in {output_dir}")
        head_grid.load_state_dict(checkpoint["head_grid"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["completed_epochs"])

    head_grid.train()
    total_remaining = (EPOCHS - start_epoch) * steps_per_epoch
    progress = tqdm(
        total=total_remaining,
        desc=f"{progress_label} cap={cap_shot} train",
        dynamic_ncols=True,
        unit="batch",
    )
    for epoch in range(start_epoch, EPOCHS):
        epoch_rng = np.random.default_rng(np.random.SeedSequence([args.seed, cap_shot, epoch]))
        shuffled = epoch_rng.permutation(support_indices)
        for batch_indices in _iter_batches(shuffled, BATCH_SIZE):
            features = torch.from_numpy(
                np.ascontiguousarray(train_features[batch_indices])
            ).to(device, non_blocking=True)
            labels = torch.from_numpy(
                np.ascontiguousarray(train_labels[batch_indices])
            ).to(device, non_blocking=True)
            logits = head_grid(features)
            loss = torch.stack(
                [nn.functional.cross_entropy(output, labels) for output in logits.values()]
            ).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            progress.update(1)
            progress.set_postfix(epoch=f"{epoch + 1}/{EPOCHS}", loss=f"{loss.item():.3f}")
        _save_checkpoint(
            checkpoint_path,
            {
                "protocol_fingerprint": protocol["fingerprint"],
                "completed_epochs": epoch + 1,
                "head_grid": head_grid.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
        )
    progress.close()

    classifiers = _evaluate(head_grid, val_features, val_labels, device)
    best = max(classifiers, key=lambda item: item["metrics"]["top-1"])
    result = {
        "protocol_version": protocol_version,
        "protocol_fingerprint": protocol["fingerprint"],
        "cap_shot": int(cap_shot),
        "actual_train_samples": int(len(support_indices)),
        "best_classifier": {
            "name": best["name"],
            "accuracy": best["metrics"]["top-1"],
            "top5_accuracy": best["metrics"]["top-5"],
            "base_lr": best["base_lr"],
            "effective_lr": best["effective_lr"],
        },
        "classifiers": classifiers,
    }
    _atomic_json(result_path, result)
    checkpoint_path.unlink(missing_ok=True)
    del head_grid, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> int:
    args = _build_parser().parse_args()
    if args.model not in MODEL_TO_ROW:
        raise ValueError(f"Model is not in the canonical 70-model panel: {args.model}")
    if args.seed < 0 or args.support_seed < 0 or args.validation_seed < 0:
        raise ValueError("Seeds must be non-negative")
    if args.validation_samples is not None and args.validation_samples <= 0:
        raise ValueError("--validation-samples must be positive")
    if _parse_cap_shots(args.cap_shots) != [5]:
        raise ValueError("The paper protocol is fixed to --cap-shots 5")
    if args.validation_samples != 5_000 or args.validation_seed != 42:
        raise ValueError(
            "The paper protocol requires --validation-samples 5000 "
            "and --validation-seed 42"
        )
    if args.stop_after_epoch != EPOCHS:
        raise ValueError("The paper protocol is fixed to --stop-after-epoch 1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU")
    if args.feature_microbatch_size is None:
        args.feature_microbatch_size = FEATURE_MICROBATCH_SIZES[args.model]
    if args.cache_batch_size is None:
        args.cache_batch_size = min(256, args.feature_microbatch_size)
    if args.num_workers < 0 or args.cache_batch_size <= 0 or args.feature_microbatch_size <= 0:
        raise ValueError("Worker and batch settings must be valid")

    rank, requested_id = MODEL_TO_ROW[args.model]
    total_models = int(os.environ.get("SWEEP_TOKENIZER_TOTAL", "70"))
    model_index = int(os.environ.get("SWEEP_TOKENIZER_INDEX", str(rank)))
    progress_label = f"shot-sweep tokenizer {model_index}/{total_models} {requested_id}"

    base.BATCH_SIZE = BATCH_SIZE
    base.EPOCHS = EPOCHS
    base.PROTOCOL_VERSION = PROTOCOL_VERSION
    base._seed_everything(args.seed)
    base._configure_cuda_math()
    device = torch.device("cuda", torch.cuda.current_device())

    data_root = Path(args.data_root).expanduser().resolve()
    extra_root = (
        Path(args.extra_root).expanduser().resolve()
        if args.extra_root
        else data_root / "extra"
    )
    train_dataset_str = f"ImageNet:split=TRAIN:root={data_root}:extra={extra_root}"
    val_dataset_str = f"ImageNet:split=VAL:root={data_root}:extra={extra_root}"
    bundle = cached._load_feature_bundle(args.model, args, device)
    feature_model = base.FrozenFeatureModel(
        bundle, device=device, microbatch_size=args.feature_microbatch_size
    ).to(device).eval()
    train_dataset = base.make_dataset(
        dataset_str=train_dataset_str, transform=bundle.eval_transform
    )
    val_dataset = base.make_dataset(
        dataset_str=val_dataset_str, transform=bundle.eval_transform
    )
    train_targets = np.asarray(train_dataset.get_targets(), dtype=np.int64)
    orders, available_counts = _nested_class_orders(train_targets, args.support_seed)
    min_class_count = int(available_counts.min())
    max_class_count = int(available_counts.max())
    cap_shots = [
        value for value in _parse_cap_shots(args.cap_shots)
        if value <= max_class_count
    ]
    if not cap_shots:
        raise RuntimeError("No requested cap-shot point is supported by the dataset")
    max_cap_shot = max(cap_shots)
    max_support_source_indices = _support_indices(orders, max_cap_shot)

    cache_root = Path(args.cache_root).expanduser().resolve()
    full_identity = _full_cache_identity(args, bundle, data_root, extra_root)
    metadata = _find_reusable_full_train_cache(
        cache_root / requested_id,
        full_identity,
        len(train_targets),
    )
    if metadata is None:
        compact_train_dataset = SupportDataset(
            train_dataset, max_support_source_indices
        )
        identity = _compact_cache_identity(
            args,
            bundle,
            data_root,
            extra_root,
            max_support_source_indices,
            max_cap_shot,
        )
        fingerprint = cached._json_fingerprint(identity)
        cache_dir = cache_root / requested_id / fingerprint[:16]
        metadata = _prepare_cache(
            args=args,
            identity=identity,
            bundle=bundle,
            feature_model=feature_model,
            train_dataset=compact_train_dataset,
            val_dataset=val_dataset,
            cache_dir=cache_dir,
            progress_label=progress_label,
            train_source_indices=max_support_source_indices,
        )
    else:
        print(
            f">> reuse complete full-train cache for {requested_id}; "
            f"evaluate requested cap points without new extraction",
            flush=True,
        )
    del feature_model, train_dataset, val_dataset
    bundle.encoder = nn.Identity()
    torch.cuda.empty_cache()
    gc.collect()

    train_meta = metadata["splits"]["train"]
    val_meta = metadata["splits"]["val"]
    train_features = np.load(train_meta["features_path"], mmap_mode="r")
    train_labels = np.load(train_meta["labels_path"], mmap_mode="r")
    val_features = np.load(val_meta["features_path"], mmap_mode="r")
    val_labels = np.load(val_meta["labels_path"], mmap_mode="r")
    physical_validation_samples = int(len(val_labels))
    if args.validation_samples is None:
        validation_source_indices = np.arange(
            physical_validation_samples, dtype=np.int64
        )
    else:
        if args.validation_samples > physical_validation_samples:
            raise ValueError(
                f"Requested {args.validation_samples} validation samples, but only "
                f"{physical_validation_samples} are available"
            )
        validation_rng = np.random.default_rng(args.validation_seed)
        validation_source_indices = np.sort(
            validation_rng.choice(
                physical_validation_samples,
                size=args.validation_samples,
                replace=False,
            ).astype(np.int64, copy=False)
        )
        val_features = np.asarray(val_features[validation_source_indices], dtype=np.float32)
        val_labels = np.asarray(val_labels[validation_source_indices], dtype=np.int64)
    source_indices_path = train_meta.get("source_indices_path")
    if source_indices_path:
        cache_source_indices = np.load(source_indices_path, mmap_mode="r")
    else:
        cache_source_indices = np.arange(len(train_targets), dtype=np.int64)
    if len(cache_source_indices) != len(train_labels):
        raise RuntimeError("Cached source-index and label lengths differ")
    if not np.array_equal(train_labels, train_targets[cache_source_indices]):
        raise RuntimeError("Cached train labels do not match their source indices")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    validation_hash = _sha256_indices(validation_source_indices)
    if args.validation_samples is not None:
        validation_indices_path = (
            output_root
            / f"validation_indices_n{len(validation_source_indices)}_seed{args.validation_seed}.npy"
        )
        if validation_indices_path.is_file():
            recorded_indices = np.load(validation_indices_path)
            if not np.array_equal(recorded_indices, validation_source_indices):
                raise RuntimeError(f"Validation-index mismatch in {validation_indices_path}")
        else:
            # Multiple model workers share output_root. A per-process source
            # avoids a race while identical workers atomically publish the
            # same deterministic index array.
            temporary_indices_path = validation_indices_path.with_suffix(
                f".npy.tmp.{os.getpid()}"
            )
            with temporary_indices_path.open("wb") as handle:
                np.save(handle, validation_source_indices)
            os.replace(temporary_indices_path, validation_indices_path)
    protocol_version = (
        PROTOCOL_VERSION
        if args.validation_samples is None
        else f"{PROTOCOL_VERSION}_val{len(validation_source_indices)}_seed{args.validation_seed}_v1"
    )
    model_output = output_root / requested_id
    schedule = {
        "protocol_version": protocol_version,
        "model": args.model,
        "requested_id": requested_id,
        "rank": rank,
        "support_seed": args.support_seed,
        "available_class_count_min": min_class_count,
        "available_class_count_max": max_class_count,
        "official_train_samples": int(len(train_targets)),
        "cached_train_samples": int(len(train_labels)),
        "physical_cached_validation_samples": physical_validation_samples,
        "reported_validation_samples": int(len(validation_source_indices)),
        "validation_seed": (
            None if args.validation_samples is None else int(args.validation_seed)
        ),
        "validation_indices_sha256": validation_hash,
        "cap_shots": cap_shots,
        "actual_samples": {
            str(cap): int(np.minimum(available_counts, cap).sum()) for cap in cap_shots
        },
        "semantics": (
            "select an exact nested class-balanced prefix at each requested cap; "
            "the current sweep stops at the largest requested cap"
        ),
    }
    _atomic_json(model_output / "sweep_schedule.json", schedule)

    summary_rows = []
    for shot_index, cap_shot in enumerate(cap_shots, start=1):
        print(
            f">> shot {shot_index}/{len(cap_shots)} cap={cap_shot}; "
            f"tokenizer {model_index}/{total_models} {requested_id}",
            flush=True,
        )
        support_source_indices = _support_indices(orders, cap_shot)
        support = np.searchsorted(cache_source_indices, support_source_indices)
        if (
            np.any(support >= len(cache_source_indices))
            or not np.array_equal(
                np.asarray(cache_source_indices[support]), support_source_indices
            )
        ):
            raise RuntimeError(
                f"Cache does not cover the requested {cap_shot}-shot support"
            )
        budget_dir = model_output / f"cap_{cap_shot:04d}"
        result = _train_budget(
            args=args,
            output_dir=budget_dir,
            metadata=metadata,
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            validation_source_indices=validation_source_indices,
            physical_validation_samples=physical_validation_samples,
            protocol_version=protocol_version,
            support_indices=support,
            support_source_indices=support_source_indices,
            official_train_samples=len(train_targets),
            cap_shot=cap_shot,
            min_class_count=min_class_count,
            max_class_count=max_class_count,
            progress_label=f"shot {shot_index}/{len(cap_shots)} | tokenizer {model_index}/{total_models}",
            device=device,
        )
        summary_rows.append(
            {
                "cap_shot": cap_shot,
                "actual_train_samples": result["actual_train_samples"],
                "top1": result["best_classifier"]["accuracy"],
                "top5": result["best_classifier"]["top5_accuracy"],
                "base_lr": result["best_classifier"]["base_lr"],
            }
        )
        _atomic_json(model_output / "results_summary.json", {"rows": summary_rows})
    print(f">> completed all shot budgets for tokenizer {model_index}/{total_models} {requested_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
