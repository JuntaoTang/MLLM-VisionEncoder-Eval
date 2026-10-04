# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import io
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image as PILImage
from torchvision.datasets import VisionDataset


logger = logging.getLogger("dinov2")


class HFDataset(VisionDataset):
    """Local Hugging Face image-classification dataset.

    The loader supports the two layouts currently present under data/hf_datasets:
    split directories produced by save_to_disk() and local HF repositories with
    parquet shards under data/ or <split>/data/.
    """

    COLUMN_SPECS: Dict[str, Dict[str, str]] = {
        "caltech101": {"image": "image", "target": "label"},
        "cifar100": {"image": "img", "target": "fine_label"},
        "dtd": {"image": "image", "target": "label"},
        "fgvc_aircraft": {"image": "image", "target": "label"},
        "flowers102": {"image": "image", "target": "label"},
        "food101": {"image": "image", "target": "label"},
        "oxford_pets": {"image": "image", "target": "label"},
        "stanford_cars": {"image": "image", "target": "label"},
        "sun397": {"image": "image", "target": "label"},
    }

    SPLIT_ALIASES = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "test": "test",
    }

    def __init__(
        self,
        *,
        name: str,
        split: str = "TRAIN",
        root: str,
        image_column: Optional[str] = None,
        target_column: Optional[str] = None,
        transforms: Optional[Any] = None,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.name = name
        self.dataset_root = Path(root).expanduser() / name
        self._hf = self._import_datasets()
        self.split = self._resolve_split(split)

        spec = self.COLUMN_SPECS.get(name, {})
        self.image_column = image_column or spec.get("image", "image")
        self.target_column = target_column or spec.get("target", "label")

        self._dataset = self._load_split(self.split)
        self._check_columns()
        self._dataset = self._dataset.cast_column(self.image_column, self._hf["Image"](decode=True))
        self._target_mapping = self._build_target_mapping()
        self._targets = self._map_targets(self._read_raw_targets(self._dataset))

        logger.info(
            "loaded HF dataset name=%s split=%s samples=%d classes=%d image_column=%s target_column=%s",
            self.name,
            self.split,
            len(self._dataset),
            len(self._target_mapping),
            self.image_column,
            self.target_column,
        )

    @staticmethod
    def _import_datasets() -> Dict[str, Any]:
        try:
            from datasets import Image, load_dataset, load_from_disk
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "HFDataset requires the Hugging Face datasets package. "
                "Install dinov2/requirements.txt, which includes datasets>=2.20."
            ) from e
        return {"Image": Image, "load_dataset": load_dataset, "load_from_disk": load_from_disk}

    def _available_splits(self) -> List[str]:
        splits = set()
        for split in ("train", "validation", "test"):
            if (self.dataset_root / split / "state.json").is_file():
                splits.add(split)
            if list((self.dataset_root / "data").glob(f"{split}-*.parquet")):
                splits.add(split)
            if list((self.dataset_root / split / "data").glob(f"{split}-*.parquet")):
                splits.add(split)
        return sorted(splits)

    def _resolve_split(self, split: str) -> str:
        requested = str(split).lower()
        requested = self.SPLIT_ALIASES.get(requested, requested)
        available = self._available_splits()

        if requested in available:
            return requested
        if requested == "validation" and "test" in available:
            return "test"
        if requested == "test" and "validation" in available:
            return "validation"

        raise FileNotFoundError(
            f'No split "{split}" for HF dataset "{self.name}" under {self.dataset_root}. '
            f"Available splits: {available or 'none'}"
        )

    def _parquet_files(self, split: str) -> List[str]:
        files = sorted((self.dataset_root / "data").glob(f"{split}-*.parquet"))
        if not files:
            files = sorted((self.dataset_root / split / "data").glob(f"{split}-*.parquet"))
        return [str(path) for path in files]

    def _load_split(self, split: str):
        split_dir = self.dataset_root / split
        if (split_dir / "state.json").is_file():
            return self._hf["load_from_disk"](str(split_dir))

        files = self._parquet_files(split)
        if files:
            cache_dir = self.dataset_root / ".cache" / "huggingface" / "datasets"
            return self._hf["load_dataset"](
                "parquet",
                data_files=files,
                split="train",
                cache_dir=str(cache_dir),
            )

        raise FileNotFoundError(f'Could not find local data files for split "{split}" in {self.dataset_root}')

    def _check_columns(self) -> None:
        columns = set(self._dataset.column_names)
        missing = [col for col in (self.image_column, self.target_column) if col not in columns]
        if missing:
            raise KeyError(
                f'HF dataset "{self.name}" split "{self.split}" is missing columns {missing}; '
                f"available columns: {sorted(columns)}"
            )

    def _read_raw_targets(self, dataset) -> List[int]:
        try:
            values = dataset[self.target_column]
        except KeyError as e:
            raise KeyError(f'Missing target column "{self.target_column}" in HF dataset "{self.name}"') from e
        return [int(value) for value in values]

    def _build_target_mapping(self) -> Dict[int, int]:
        train_dataset = self._dataset if self.split == "train" else self._load_split("train")
        train_targets = sorted(set(self._read_raw_targets(train_dataset)))
        if not train_targets:
            raise ValueError(f'HF dataset "{self.name}" has no train targets')
        return {raw_target: contiguous_target for contiguous_target, raw_target in enumerate(train_targets)}

    def _map_targets(self, raw_targets: Iterable[int]) -> np.ndarray:
        mapped = []
        for raw_target in raw_targets:
            if raw_target not in self._target_mapping:
                raise ValueError(
                    f'HF dataset "{self.name}" split "{self.split}" contains target {raw_target}, '
                    "which is not present in the train split"
                )
            mapped.append(self._target_mapping[raw_target])
        return np.asarray(mapped, dtype=np.int64)

    def _decode_image(self, value: Any) -> PILImage.Image:
        if isinstance(value, PILImage.Image):
            return value
        if isinstance(value, bytes):
            return PILImage.open(io.BytesIO(value))
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                return PILImage.open(io.BytesIO(value["bytes"]))
            if value.get("path") is not None:
                return PILImage.open(value["path"])
        raise TypeError(f"Unsupported image value type for HF dataset {self.name}: {type(value)!r}")

    def __getitem__(self, index: int):
        sample = self._dataset[index]
        image = self._decode_image(sample[self.image_column])
        if image.mode != "RGB":
            image = image.convert("RGB")

        target = int(self._targets[index])
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self) -> int:
        return len(self._dataset)

    def get_targets(self) -> np.ndarray:
        return self._targets
