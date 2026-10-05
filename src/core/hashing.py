from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible data deterministically."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_files(paths: Iterable[str | Path], *, root: str | Path) -> str:
    """Hash relative names and contents in a stable order."""
    root_path = Path(root).resolve()
    records = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        try:
            relative = path.relative_to(root_path).as_posix()
        except ValueError as exc:
            raise ValueError(f"path is outside hash root: {path}") from exc
        records.append((relative, sha256_file(path)))
    return sha256_json(sorted(records))
