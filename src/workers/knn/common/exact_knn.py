"""Exact weighted KNN shared by all benchmarks."""
from __future__ import annotations

import numpy as np


def exact_weighted_knn(database, database_labels, query, *, k=20, temperature=0.07, num_classes=1000):
    import faiss

    database = np.ascontiguousarray(database, dtype=np.float32)
    query = np.ascontiguousarray(query, dtype=np.float32)
    faiss.normalize_L2(database)
    faiss.normalize_L2(query)
    index = faiss.IndexFlatIP(database.shape[1])
    index.add(database)
    scores, neighbours = index.search(query, k)
    labels = np.asarray(database_labels, dtype=np.int64)[neighbours]
    weights = np.exp((scores - scores.max(axis=1, keepdims=True)) / temperature)
    votes = np.zeros((len(query), num_classes), dtype=np.float32)
    rows = np.arange(len(query))[:, None]
    np.add.at(votes, (np.broadcast_to(rows, labels.shape), labels), weights)
    return votes.argmax(axis=1)
