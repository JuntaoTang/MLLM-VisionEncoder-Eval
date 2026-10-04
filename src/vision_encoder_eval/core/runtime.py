"""Configured runtime roots shared by isolated workers."""
from __future__ import annotations

import json
import os
from pathlib import Path


def repository_root() -> Path:
    override = os.environ.get('VEE_RESOURCE_ROOT')
    if override:
        return Path(override).resolve()
    candidate = Path(__file__).resolve().parents[3]
    if (candidate/'pyproject.toml').is_file():
        return candidate
    # Small packaged assets support summary/CPU workers in an installed wheel.
    # Model workers need a separately configured tree containing third_party.
    return Path(__file__).resolve().parents[1]/'resources'


def mllm_root() -> str:
    root = os.environ.get('VEE_MLLM_ROOT')
    if root:
        return str(Path(root).resolve())
    return str(repository_root())


def mllm_configs_root() -> str:
    root = os.environ.get('VEE_MLLM_ROOT')
    if root:
        return str(Path(root).resolve()/'configs')
    return str(Path(__file__).resolve().parents[1]/'resources/mllm_configs')


def asset_path(name: str, suffix: str = '') -> str:
    paths = json.loads(os.environ.get('VEE_PATHS', '{}'))
    if name == 'mllm':
        return str(Path(mllm_root()) / suffix)
    if name == 'third_party':
        return str(repository_root()/'third_party'/suffix)
    if name == 'package':
        return str(Path(__file__).resolve().parents[1]/suffix)
    if name not in paths:
        # Importing numerical kernels should not require every dataset root.
        # Workers validate the paths they consume before loading a model.
        return str(Path('__configure_local_paths__')/name/suffix)
    return str(Path(paths[name]) / suffix)
