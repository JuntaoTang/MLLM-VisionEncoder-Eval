from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ConfigError(ValueError):
    pass


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be a mapping")
    return value


def validate_local_config(value: Mapping[str, Any]) -> None:
    if int(value.get("schema_version", 0)) != 1:
        raise ConfigError("local.schema_version must be 1")
    paths = _mapping(value.get("paths"), "local.paths")
    for name, path in paths.items():
        if not isinstance(name, str) or not isinstance(path, str) or not path.strip():
            raise ConfigError("local.paths must map names to non-empty strings")
    output_root = value.get("output_root")
    if not isinstance(output_root, str) or not output_root.strip():
        raise ConfigError("local.output_root must be a non-empty string")
    environments = _mapping(value.get("environments", {}), "local.environments")
    for name, environment in environments.items():
        record = _mapping(environment, f"local.environments.{name}")
        python = record.get("python")
        if python is not None and (not isinstance(python, str) or not python.strip()):
            raise ConfigError(f"local.environments.{name}.python must be a string")


def validate_experiment_config(value: Mapping[str, Any]) -> None:
    if int(value.get("schema_version", 0)) != 1:
        raise ConfigError("experiment.schema_version must be 1")
    experiment = _mapping(value.get("experiment"), "experiment.experiment")
    for field in ("name", "kind"):
        item = experiment.get(field)
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"experiment.experiment.{field} must be a non-empty string")
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', experiment['name']):
        raise ConfigError('experiment name must be a safe filename identifier')
    kind = experiment["kind"]
    allowed = {
        "method",
        "mllm_train",
        "mllm_eval",
        "feature_extract",
        "report",
        "suite",
    }
    if kind not in allowed:
        raise ConfigError(f"unsupported experiment kind {kind!r}; expected one of {sorted(allowed)}")
    if kind == "method":
        method = value.get("method")
        if not isinstance(method, str) or not method.strip():
            raise ConfigError("method experiments require a non-empty method field")
    if kind not in {"suite", "report"}:
        dataset = value.get("dataset")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ConfigError(f"{kind} experiments require a non-empty dataset field")
    if kind == 'suite' and (not isinstance(value.get('experiments'), list) or not value['experiments']):
        raise ConfigError('suite requires a non-empty experiments list')
