from __future__ import annotations

from typing import Any

import numpy as np

from .preprocessing import EIGENVALUE_FLOOR


def fit_full_rank_patch_pca(
    patches: np.ndarray,
    *,
    eigenvalue_floor: float = EIGENVALUE_FLOOR,
    image_batch: int = 8,
) -> dict[str, Any]:
    """Fit full numerical-rank PCA to all patch tokens without r_max truncation."""
    values = np.asarray(patches)
    if values.ndim != 3 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError(f"expected finite [n, tokens, dim] patches, got {values.shape}")
    n_images, n_tokens, dim = values.shape
    n_flat = int(n_images) * int(n_tokens)
    if n_flat <= 150_000:
        from sklearn.decomposition import PCA

        flat = np.ascontiguousarray(np.asarray(values, dtype=np.float32).reshape(n_flat, dim))
        mean = flat.mean(axis=0, keepdims=True)
        centered = flat - mean
        rank = int(np.linalg.matrix_rank(centered))
        component_count = min(dim, n_flat - 1, rank)
        if component_count < 1:
            raise ValueError("patch PCA has zero usable rank")
        pca = PCA(n_components=component_count, whiten=False, svd_solver="full", random_state=42)
        pca.fit(centered)
        eigenvalues = np.asarray(pca.explained_variance_, dtype=np.float64)
        usable = int(np.sum(eigenvalues > float(eigenvalue_floor)))
        if usable < 1:
            raise ValueError("patch PCA has zero numerically usable rank")
        return {
            "backend": "sklearn_full",
            "mean": (mean.ravel() + np.asarray(pca.mean_, dtype=np.float64)).astype(np.float64),
            "components": np.asarray(pca.components_.T[:, :usable], dtype=np.float64),
            "explained_variance": eigenvalues[:usable],
            "meta": {
                "input_shape": [n_flat, dim],
                "n_images": int(n_images),
                "n_tokens": int(n_tokens),
                "dim_in": int(dim),
                "full_usable_rank": usable,
                "eigenvalue_floor": float(eigenvalue_floor),
                "truncated": False,
                "solver": "sklearn_full",
            },
        }
    mean = np.zeros(dim, dtype=np.float64)
    for start in range(0, n_images, image_batch):
        chunk = np.asarray(values[start : start + image_batch], dtype=np.float32).reshape(-1, dim)
        mean += np.asarray(chunk, dtype=np.float64).sum(axis=0)
    mean /= n_flat
    covariance = np.zeros((dim, dim), dtype=np.float64)
    for start in range(0, n_images, image_batch):
        chunk = np.asarray(values[start : start + image_batch], dtype=np.float32).reshape(-1, dim)
        centered = np.asarray(chunk, dtype=np.float64) - mean
        covariance += centered.T @ centered
    covariance /= max(n_flat - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    usable = int(np.sum(eigenvalues > float(eigenvalue_floor)))
    if usable < 1:
        raise ValueError("patch PCA has zero numerically usable rank")
    return {
        "backend": "covariance_eigh",
        "mean": mean,
        "components": np.asarray(eigenvectors[:, :usable], dtype=np.float64),
        "explained_variance": np.asarray(eigenvalues[:usable], dtype=np.float64),
        "meta": {
            "input_shape": [n_flat, dim],
            "n_images": int(n_images),
            "n_tokens": int(n_tokens),
            "dim_in": int(dim),
            "full_usable_rank": usable,
            "eigenvalue_floor": float(eigenvalue_floor),
            "truncated": False,
            "solver": "covariance_eigh",
        },
    }


def transform_patches_full_rank(
    patches: np.ndarray,
    pca_state: dict[str, Any],
    *,
    whitening_eps: float = 1e-4,
    image_batch: int = 8,
) -> np.ndarray:
    values = np.asarray(patches)
    if values.ndim != 3:
        raise ValueError(f"expected [n, tokens, dim] patches, got {values.shape}")
    n_images, n_tokens, dim = values.shape
    mean = np.asarray(pca_state["mean"], dtype=np.float64).reshape(1, -1)
    components = np.asarray(pca_state["components"], dtype=np.float64)
    eigenvalues = np.asarray(pca_state["explained_variance"], dtype=np.float64)
    if mean.shape[1] != dim or components.shape[0] != dim:
        raise ValueError("patch PCA state does not match input dimension")
    scale = np.sqrt(eigenvalues + float(whitening_eps)).reshape(1, 1, -1)
    output = np.empty((n_images, n_tokens, components.shape[1]), dtype=np.float32)
    for start in range(0, n_images, image_batch):
        stop = min(n_images, start + image_batch)
        chunk = np.asarray(values[start:stop], dtype=np.float32).reshape(-1, dim)
        projected = (np.asarray(chunk, dtype=np.float64) - mean) @ components
        projected = projected.reshape(stop - start, n_tokens, -1).astype(np.float32)
        projected = np.asarray(projected, dtype=np.float64) / scale
        norms = np.linalg.norm(projected, axis=2, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("zero patch-token norm after whitening")
        output[start:stop] = (projected / norms).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("non-finite patch representation")
    return output


def symmetric_chamfer_similarity(
    patches: np.ndarray,
    *,
    device: str | None = None,
    normalize_tokens: bool = True,
) -> np.ndarray:
    import torch

    values = np.asarray(patches)
    if values.ndim != 3:
        raise ValueError(f"expected [n, tokens, dim] patches, got {values.shape}")
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.as_tensor(np.ascontiguousarray(values), device=selected_device, dtype=torch.float32)
    if normalize_tokens:
        tensor = tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    n, n_tokens, _ = tensor.shape
    bytes_per_gallery = max(int(n_tokens) * int(n_tokens) * 4, 1)
    chunk = max(1, min(n, int(8e8 / bytes_per_gallery)))
    similarity = torch.empty((n, n), device=selected_device, dtype=torch.float32)
    for i in range(n):
        query = tensor[i]
        j0 = i
        while j0 < n:
            stop = min(n, j0 + chunk)
            gallery = tensor[j0:stop]
            try:
                dots = torch.einsum("td,jrd->tjr", query, gallery)
                values_ij = 0.5 * (
                    dots.amax(dim=2).mean(dim=0) + dots.amax(dim=0).mean(dim=1)
                )
                similarity[i, j0:stop] = values_ij
                similarity[j0:stop, i] = values_ij
                j0 = stop
                del dots, values_ij
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or chunk <= 1:
                    raise
                if selected_device.startswith("cuda"):
                    torch.cuda.empty_cache()
                chunk = max(1, chunk // 2)
    similarity.fill_diagonal_(float("-inf"))
    finite = similarity[~torch.isinf(similarity)]
    if finite.numel() and not torch.isfinite(finite).all():
        raise RuntimeError("Chamfer similarity contains NaN/Inf")
    result = similarity.cpu().numpy()
    del tensor, similarity, finite
    return result
