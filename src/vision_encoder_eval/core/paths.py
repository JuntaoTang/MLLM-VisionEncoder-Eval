from __future__ import annotations

from pathlib import Path


def resolve_path(value: str | Path, *, relative_to: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(relative_to) / path
    return path.resolve()


def require_file(path: str | Path, *, label: str = "file") -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def require_directory(path: str | Path, *, label: str = "directory") -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved
