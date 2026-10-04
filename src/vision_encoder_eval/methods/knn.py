"""Exact weighted kNN classification and nested-shot protocol validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def exact_weighted_knn(
    database,
    database_labels,
    query,
    *,
    k: int = 20,
    temperature: float = 0.07,
    num_classes: int = 1000,
) -> np.ndarray:
    """Match the legacy FAISS IndexFlatIP weighted-vote protocol exactly."""
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "exact weighted kNN requires faiss; install the 'knn' extra"
        ) from exc

    database = np.ascontiguousarray(database, dtype=np.float32)
    query = np.ascontiguousarray(query, dtype=np.float32)
    labels_array = np.asarray(database_labels, dtype=np.int64)
    if database.ndim != 2 or query.ndim != 2:
        raise ValueError("database and query must both be [N,D] arrays")
    if database.shape[1] != query.shape[1]:
        raise ValueError(
            f"database/query feature dimensions differ: {database.shape} vs {query.shape}"
        )
    if labels_array.shape != (len(database),):
        raise ValueError(
            f"database labels must have shape {(len(database),)}, got {labels_array.shape}"
        )
    if not np.isfinite(database).all() or not np.isfinite(query).all():
        raise ValueError("kNN features contain NaN/Inf")
    if not 0 < int(k) <= len(database):
        raise ValueError(f"k must be in [1, {len(database)}], got {k}")
    if not float(temperature) > 0:
        raise ValueError("temperature must be positive")
    if int(num_classes) <= 0:
        raise ValueError("num_classes must be positive")
    if labels_array.size and (
        int(labels_array.min()) < 0 or int(labels_array.max()) >= int(num_classes)
    ):
        raise ValueError("database labels are outside [0, num_classes)")

    faiss.normalize_L2(database)
    faiss.normalize_L2(query)
    index = faiss.IndexFlatIP(database.shape[1])
    index.add(database)
    scores, neighbours = index.search(query, int(k))
    labels = labels_array[neighbours]
    weights = np.exp((scores - scores.max(axis=1, keepdims=True)) / float(temperature))
    votes = np.zeros((len(query), int(num_classes)), dtype=np.float32)
    rows = np.arange(len(query))[:, None]
    np.add.at(votes, (np.broadcast_to(rows, labels.shape), labels), weights)
    return votes.argmax(axis=1)


def validate_nested_protocol(
    protocol: Mapping[str, Any],
    *,
    feature_count: int,
) -> tuple[int, ...]:
    raw_shots = protocol.get("train_shots")
    raw_queries = protocol.get("query_indices")
    by_shot = protocol.get("train_indices_by_shot")
    if not isinstance(raw_shots, Sequence) or isinstance(raw_shots, (str, bytes)):
        raise ValueError("protocol.train_shots must be a sequence")
    if not isinstance(raw_queries, Sequence) or isinstance(raw_queries, (str, bytes)):
        raise ValueError("protocol.query_indices must be a sequence")
    if not isinstance(by_shot, Mapping):
        raise ValueError("protocol.train_indices_by_shot must be a mapping")
    shots = tuple(int(shot) for shot in raw_shots)
    if not shots or tuple(sorted(set(shots))) != shots or any(shot <= 0 for shot in shots):
        raise ValueError("protocol.train_shots must be positive, unique, and increasing")
    queries = np.asarray(raw_queries, dtype=np.int64)
    if len(np.unique(queries)) != len(queries):
        raise ValueError("protocol query indices contain duplicates")
    if len(queries) and (queries.min() < 0 or queries.max() >= feature_count):
        raise ValueError("protocol query index is out of range")
    previous: set[int] = set()
    query_set = set(int(item) for item in queries)
    for shot in shots:
        key = str(shot)
        if key not in by_shot:
            raise ValueError(f"protocol is missing train indices for shot {shot}")
        indices = [int(item) for item in by_shot[key]]
        current = set(indices)
        if len(current) != len(indices):
            raise ValueError(f"shot {shot} train indices contain duplicates")
        if current & query_set:
            raise ValueError(f"shot {shot} train/query indices overlap")
        if previous and not previous.issubset(current):
            raise ValueError(f"shot {shot} is not nested over the previous shot")
        if current and (min(current) < 0 or max(current) >= feature_count):
            raise ValueError(f"shot {shot} train index is out of range")
        previous = current
    return shots


def _load_array(raw_path: Any, *, output_root: str) -> tuple[np.ndarray, Path]:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = Path(output_root) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"kNN input does not exist: {path}")
    return np.load(path, allow_pickle=False), path


def run_knn(resolved: Mapping[str, Any]) -> Mapping[str, Any]:
    experiment = resolved["experiment"]
    local = resolved["local"]
    inputs = experiment.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("kNN experiment requires an inputs mapping")
    output_root = str(local["output_root"])
    database, database_path = _load_array(inputs.get("database"), output_root=output_root)
    database_labels, labels_path = _load_array(
        inputs.get("database_labels"), output_root=output_root
    )
    query, query_path = _load_array(inputs.get("query"), output_root=output_root)
    protocol = dict(experiment.get("protocol", {}))
    predictions = exact_weighted_knn(
        database,
        database_labels,
        query,
        k=int(protocol.get("k", 20)),
        temperature=float(protocol.get("temperature", 0.07)),
        num_classes=int(protocol.get("num_classes", 1000)),
    )
    metrics: dict[str, Any] = {}
    audit_paths = {
        "database": str(database_path),
        "database_labels": str(labels_path),
        "query": str(query_path),
    }
    if inputs.get("query_labels") is not None:
        query_labels, query_labels_path = _load_array(
            inputs["query_labels"], output_root=output_root
        )
        query_labels = np.asarray(query_labels, dtype=np.int64)
        if query_labels.shape != predictions.shape:
            raise ValueError("query labels do not match query rows")
        metrics["top1"] = float((predictions == query_labels).mean() * 100.0)
        audit_paths["query_labels"] = str(query_labels_path)
    config_hash = str(resolved.get("config_sha256", "unknown"))
    return {
        "schema_version": 1,
        "run_id": f"{experiment['experiment']['name']}-{config_hash[:12]}",
        "status": "success",
        "method": "knn",
        "dataset": experiment["dataset"],
        "protocol": protocol,
        "metrics": metrics,
        "audit": {"config_sha256": config_hash, **audit_paths},
        "details": {"predictions": predictions.tolist()},
    }
