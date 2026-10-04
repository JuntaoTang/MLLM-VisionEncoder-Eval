"""Flat method registry for implementations that passed legacy parity."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Type

from ..core.registry import register_method
from .base import Method
from importlib import import_module


_CLASS_NAMES = {
    "rsa": 'RSA',
    "cca": 'CCA',
    "gw": 'GW',
    "mutualnn": 'MutualNN',
    "ravel": 'RAVEL',
}

class _LazyClasses(Mapping):
    def __getitem__(self, name):
        return getattr(import_module(f'.{name}', __name__), _CLASS_NAMES[name])
    def __iter__(self):
        return iter(_CLASS_NAMES)
    def __len__(self):
        return len(_CLASS_NAMES)

METHOD_CLASSES = _LazyClasses()


def _resolve_input(raw_path: str, *, output_root: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(output_root) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"feature input does not exist: {path}")
    return path


def _runner_for(method_name):
    def run(resolved: Mapping[str, Any]) -> Mapping[str, Any]:
        import numpy as np
        method_class = METHOD_CLASSES[method_name]
        experiment = resolved["experiment"]
        local = resolved["local"]
        inputs = experiment.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("method experiment requires an inputs mapping")
        visual_path = _resolve_input(
            str(inputs.get("visual_features", "")),
            output_root=str(local["output_root"]),
        )
        text_path = _resolve_input(
            str(inputs.get("text_features", "")),
            output_root=str(local["output_root"]),
        )
        visual = np.load(visual_path, allow_pickle=False)
        text = np.load(text_path, allow_pickle=False)
        protocol = dict(experiment.get("protocol", {}))
        from ..data.features import validate_paired_rows
        row_contract = validate_paired_rows(inputs,visual_path,text_path,visual,text,protocol)
        details = dict(method_class().evaluate(visual, text, protocol))
        config_hash = str(resolved.get("config_sha256", "unknown"))
        name = method_class.name
        return {
            "schema_version": 1,
            "run_id": f"{experiment['experiment']['name']}-{config_hash[:12]}",
            "status": "success",
            "method": name,
            "dataset": experiment["dataset"],
            "protocol": protocol,
            "metrics": {
                "raw_score": details["raw_score"],
                "final_score": details["final_score"],
            },
            "audit": {
                "config_sha256": config_hash,
                "visual_features": str(visual_path),
                "text_features": str(text_path),
                'row_identity': row_contract,
            },
            "details": details,
        }

    return run


_REQUIREMENTS = {
    "rsa": ("numpy",),
    "cca": ("numpy", "torch"),
    "gw": ("numpy", "scipy"),
    "mutualnn": ("numpy",),
    "ravel": ("numpy", "scikit-learn", "torch"),
}

for _name in _CLASS_NAMES:
    register_method(
        _name,
        description=f'Legacy-equivalent {_name} feature evaluation',
        requires=_REQUIREMENTS[_name],
    )(_runner_for(_name))

def run_knn(resolved):
    from .knn import run_knn as implementation
    return implementation(resolved)

register_method(
    "knn",
    description="Exact weighted kNN classification with a fixed nested-shot protocol.",
    requires=("numpy", "faiss"),
)(run_knn)

def _configured_worker(name):
    def run(resolved):
        raise ValueError(f'{name} requires explicit worker steps; use its configs/experiments recipe')
    return run

for _name in ('linear_probe','alignment_probe','ckax','law','tokbench','zero_shot'):
    register_method(_name,description=f'{_name}: configured isolated native evaluation protocol',
                    requires=('separate worker environment',),execution_mode='worker')(_configured_worker(_name))


__all__ = ["CCA", "GW", "MutualNN", "RAVEL", "RSA", "METHOD_CLASSES"]

def __getattr__(name):
    for module, cls in _CLASS_NAMES.items():
        if cls == name:
            return getattr(import_module(f'.{module}', __name__), cls)
    raise AttributeError(name)
