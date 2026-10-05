from __future__ import annotations

import numpy as np


def l2_normalize(features: np.ndarray, *, dtype=np.float32) -> np.ndarray:
    values = np.asarray(features, dtype=dtype)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"expected finite [n, d] features, got {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("features contain a zero-norm row")
    return values / norms


def cosine_similarity(features: np.ndarray, *, device: str | None = None) -> np.ndarray:
    normalized = l2_normalize(features)
    if device is not None:
        try:
            import torch

            tensor = torch.as_tensor(normalized, device=device, dtype=torch.float32)
            similarity = (tensor @ tensor.T).cpu().numpy()
            del tensor
        except (ImportError, RuntimeError):
            similarity = normalized @ normalized.T
    else:
        similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    return similarity


def pearson_vector(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"vector shape mismatch: {a.shape} vs {b.shape}")
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 0:
        raise ValueError("Pearson correlation is undefined for a constant vector")
    return float(np.dot(a, b) / denominator)


def fractional_topk_distribution(similarity: np.ndarray, k: int) -> np.ndarray:
    matrix = np.asarray(similarity, dtype=np.float32)
    n = int(matrix.shape[0])
    if matrix.shape != (n, n) or not 0 < k < n:
        raise ValueError(f"invalid similarity/k: {matrix.shape}, k={k}")
    distribution = np.zeros((n, n), dtype=np.float32)
    for row_index, row in enumerate(matrix):
        boundary = np.partition(row, -k)[-k]
        greater = np.flatnonzero(row > boundary)
        equal = np.flatnonzero(row == boundary)
        remaining = k - len(greater)
        distribution[row_index, greater] = np.float32(1.0 / k)
        distribution[row_index, equal] = np.float32(remaining / (len(equal) * k))
    return distribution


def fractional_overlap(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"distribution shape mismatch: {a.shape} vs {b.shape}")
    return float(np.minimum(a, b).sum(axis=1, dtype=np.float64).mean())


def topk_neighbors(
    similarity: np.ndarray,
    k: int,
    *,
    device: str | None = None,
) -> np.ndarray:
    matrix = np.asarray(similarity, dtype=np.float32)
    n = int(matrix.shape[0])
    if matrix.shape != (n, n) or not 0 < k < n:
        raise ValueError(f"invalid similarity/k: {matrix.shape}, k={k}")
    try:
        import torch

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.as_tensor(matrix, device=selected_device, dtype=torch.float32)
        indices = torch.topk(tensor, k=k, dim=1, largest=True, sorted=True).indices
        result = indices.cpu().numpy().astype(np.int32)
        del tensor, indices
        return result
    except (ImportError, RuntimeError):
        return np.argsort(-matrix, axis=1, kind="stable")[:, :k].astype(np.int32)


def binary_overlap(left_neighbors: np.ndarray, right_neighbors: np.ndarray) -> float:
    left = np.asarray(left_neighbors, dtype=np.int32)
    right = np.asarray(right_neighbors, dtype=np.int32)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(f"neighbor shape mismatch: {left.shape} vs {right.shape}")
    n, k = left.shape
    graph = np.zeros((n, n), dtype=bool)
    graph[np.arange(n)[:, None], right] = True
    return float(np.count_nonzero(graph[np.arange(n)[:, None], left]) / float(n * k))
