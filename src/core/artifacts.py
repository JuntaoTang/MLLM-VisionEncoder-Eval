from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .hashing import sha256_file


class ArtifactError(ValueError):
    pass


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class FileRecord:
    path: str
    sha256: str
    bytes: int

    @classmethod
    def capture(cls, path: str | Path, *, relative_to: str | Path) -> "FileRecord":
        source = Path(path).resolve()
        root = Path(relative_to).resolve()
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise ArtifactError(f"artifact is outside root: {source}") from exc
        if not source.is_file():
            raise ArtifactError(f"artifact file is missing: {source}")
        return cls(path=relative, sha256=sha256_file(source), bytes=source.stat().st_size)

    def verify(self, *, root: str | Path) -> None:
        source = (Path(root) / self.path).resolve()
        root_path = Path(root).resolve()
        if source != root_path and root_path not in source.parents:
            raise ArtifactError(f"artifact path escapes root: {self.path}")
        if not source.is_file():
            raise ArtifactError(f"artifact file is missing: {source}")
        if source.stat().st_size != self.bytes:
            raise ArtifactError(f"artifact size mismatch: {source}")
        actual = sha256_file(source)
        if actual != self.sha256:
            raise ArtifactError(f"artifact checksum mismatch: {source}")


def validate_result_payload(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "run_id",
        "status",
        "method",
        "dataset",
        "protocol",
        "metrics",
        "audit",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ArtifactError(f"result payload is missing fields: {missing}")
    if payload['schema_version'] != 1:
        raise ArtifactError('result schema_version must be 1')
    if payload["status"] not in {"success", "failed", "skipped"}:
        raise ArtifactError(f"invalid result status: {payload['status']!r}")
    if not isinstance(payload["metrics"], Mapping):
        raise ArtifactError("result metrics must be a mapping")
    if not isinstance(payload["audit"], Mapping):
        raise ArtifactError("result audit must be a mapping")
