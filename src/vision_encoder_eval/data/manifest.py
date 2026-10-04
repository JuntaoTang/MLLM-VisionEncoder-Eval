from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..core.hashing import sha256_json


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    image: str | None = None
    text: str | None = None
    target: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, index: int) -> "SampleRecord":
        sample_id = value.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ManifestError(f"samples[{index}].sample_id must be a non-empty string")
        image = value.get("image")
        text = value.get("text")
        metadata = value.get("metadata", {})
        if image is not None and not isinstance(image, str):
            raise ManifestError(f"samples[{index}].image must be a string or null")
        if text is not None and not isinstance(text, str):
            raise ManifestError(f"samples[{index}].text must be a string or null")
        if not isinstance(metadata, Mapping):
            raise ManifestError(f"samples[{index}].metadata must be a mapping")
        return cls(
            sample_id=sample_id,
            image=image,
            text=text,
            target=value.get("target"),
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "image": self.image,
            "text": self.text,
            "target": self.target,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    split: str
    samples: tuple[SampleRecord, ...]
    schema_version: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetManifest":
        if int(value.get("schema_version", 0)) != 1:
            raise ManifestError("manifest.schema_version must be 1")
        dataset_id = value.get("dataset_id")
        split = value.get("split")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ManifestError("manifest.dataset_id must be a non-empty string")
        if not isinstance(split, str) or not split.strip():
            raise ManifestError("manifest.split must be a non-empty string")
        raw_samples = value.get("samples")
        if not isinstance(raw_samples, list):
            raise ManifestError("manifest.samples must be a list")
        samples = tuple(
            SampleRecord.from_dict(sample, index=index)
            for index, sample in enumerate(raw_samples)
            if isinstance(sample, Mapping)
        )
        if len(samples) != len(raw_samples):
            raise ManifestError("every manifest sample must be a mapping")
        sample_ids = [sample.sample_id for sample in samples]
        duplicates = sorted({item for item in sample_ids if sample_ids.count(item) > 1})
        if duplicates:
            raise ManifestError(f"duplicate sample IDs: {duplicates}")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ManifestError("manifest.metadata must be a mapping")
        return cls(
            dataset_id=dataset_id,
            split=split,
            samples=samples,
            schema_version=1,
            metadata=dict(metadata),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise ManifestError("manifest root must be a mapping")
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "samples": [sample.to_dict() for sample in self.samples],
            "metadata": dict(self.metadata),
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())

    def validate_files(self, *, data_root: str | Path) -> None:
        root = Path(data_root).resolve()
        missing = []
        escaped = []
        for sample in self.samples:
            if sample.image is None:
                continue
            image = (root / sample.image).resolve()
            if image != root and root not in image.parents:
                escaped.append(sample.sample_id)
            elif not image.is_file():
                missing.append(sample.sample_id)
        if escaped:
            raise ManifestError(f"image paths escape data root for sample IDs: {escaped}")
        if missing:
            raise ManifestError(f"image files are missing for sample IDs: {missing}")
