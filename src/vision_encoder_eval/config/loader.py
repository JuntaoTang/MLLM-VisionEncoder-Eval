from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..core.artifacts import write_json_atomic
from ..core.hashing import sha256_json
from ..core.paths import resolve_path
from .schema import ConfigError, validate_experiment_config, validate_local_config


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_PATH_PATTERN = re.compile(r"\$\{paths\.([A-Za-z_][A-Za-z0-9_]*)\}")


def _load_document(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            if path.suffix.lower() == ".json":
                value = json.load(handle)
            else:
                try:
                    import yaml
                except ImportError as exc:
                    raise ConfigError(
                        "YAML support requires PyYAML; run scripts/setup.sh first "
                        "or use a .json config"
                    ) from exc
                value = yaml.safe_load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ConfigError(f"config root must be a mapping: {path}")
    return value


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        reserved = {'run_dir', 'workspace'}
        missing = [name for name in _ENV_PATTERN.findall(value) if name not in os.environ and name not in reserved]
        if missing:
            raise ConfigError(f"environment variables are not set: {sorted(set(missing))}")
        return _ENV_PATTERN.sub(lambda match: match.group(0) if match.group(1) in reserved else os.environ[match.group(1)], value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _expand_environment(item) for key, item in value.items()}
    return value


def _parse_override(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ConfigError(f"override must be key=value: {raw!r}")
    raw_key, raw_value = raw.split("=", 1)
    keys = [part for part in raw_key.split(".") if part]
    if not keys:
        raise ConfigError(f"override key is empty: {raw!r}")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return keys, value


def _apply_override(target: dict[str, Any], raw: str) -> None:
    keys, value = _parse_override(raw)
    cursor = target
    for key in keys[:-1]:
        existing = cursor.setdefault(key, {})
        if not isinstance(existing, dict):
            raise ConfigError(f"cannot set nested override below non-mapping key: {key}")
        cursor = existing
    cursor[keys[-1]] = value


def _resolve_local_paths(value: dict[str, Any], *, config_dir: Path) -> None:
    paths = value["paths"]
    for name, raw_path in list(paths.items()):
        paths[name] = str(resolve_path(raw_path, relative_to=config_dir))
    value["output_root"] = str(resolve_path(value["output_root"], relative_to=config_dir))
    if value.get('resource_root'):
        value['resource_root'] = str(resolve_path(value['resource_root'], relative_to=config_dir))
    for environment in value.get("environments", {}).values():
        if environment.get("python"):
            environment["python"] = str(
                resolve_path(environment["python"], relative_to=config_dir)
            )


@dataclass(frozen=True)
class ResolvedConfig:
    value: Mapping[str, Any]
    sha256: str

    def write(self, path: str | Path) -> None:
        write_json_atomic(path, {**self.value, "config_sha256": self.sha256})


def resolve_config(
    *,
    local_path: str | Path,
    experiment_path: str | Path,
    overrides: Iterable[str] = (),
) -> ResolvedConfig:
    local_file = Path(local_path).resolve()
    experiment_file = Path(experiment_path).resolve()
    local = deepcopy(dict(_expand_environment(_load_document(local_file))))
    experiment = deepcopy(dict(_expand_environment(_load_document(experiment_file))))
    validate_local_config(local)
    _resolve_local_paths(local, config_dir=local_file.parent)
    for override in overrides:
        _apply_override(experiment, override)
    def expand_paths(item):
        if isinstance(item, str):
            def replace(match):
                key = match.group(1)
                if key not in local['paths']:
                    raise ConfigError(f'unknown local path: {key}')
                return local['paths'][key]
            return _PATH_PATTERN.sub(replace, item)
        if isinstance(item, list):
            return [expand_paths(v) for v in item]
        if isinstance(item, dict):
            return {k: expand_paths(v) for k,v in item.items()}
        return item
    experiment = expand_paths(experiment)
    # Inputs and child configs are relative to the experiment, never caller cwd.
    for key, value in experiment.get('inputs', {}).items():
        if isinstance(value, str):
            experiment['inputs'][key] = str(resolve_path(value, relative_to=experiment_file.parent))
    for key, value in experiment.get('protocol', {}).items():
        if key.endswith('_path') and isinstance(value, str):
            path = str(resolve_path(value,relative_to=experiment_file.parent))
            experiment['protocol'][key] = path
            experiment.setdefault('inputs', {})[f'protocol_{key}'] = path
    if experiment.get('experiments'):
        experiment['experiments'] = [str(resolve_path(v, relative_to=experiment_file.parent))
                                     for v in experiment['experiments']]
    validate_experiment_config(experiment)
    resolved = {
        "schema_version": 1,
        "local": local,
        "experiment": experiment,
        "sources": {
            "local": str(local_file),
            "experiment": str(experiment_file),
        },
    }
    return ResolvedConfig(value=resolved, sha256=sha256_json(resolved))
