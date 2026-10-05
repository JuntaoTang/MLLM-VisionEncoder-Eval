from __future__ import annotations

from typing import Any

import numpy as np


EIGENVALUE_FLOOR = 1e-10


def full_rank_pca_whiten_l2(
    features: np.ndarray,
    *,
    whitening_eps: float = 1e-4,
    eigenvalue_floor: float = EIGENVALUE_FLOOR,
    random_state: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Full numerical-rank PCA, stabilized whitening, then row L2."""
    from sklearn.decomposition import PCA

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError(f"expected finite [n, d] features with n>=2, got {values.shape}")
    centered = values - values.mean(axis=0, keepdims=True)
    maximum = min(centered.shape[0] - 1, centered.shape[1])
    pca = PCA(
        n_components=maximum,
        whiten=False,
        svd_solver="full",
        random_state=random_state,
    )
    scores = pca.fit_transform(centered).astype(np.float32)
    eigenvalues = np.asarray(pca.explained_variance_, dtype=np.float64)
    usable = int(np.sum(eigenvalues > float(eigenvalue_floor)))
    if usable < 1:
        raise ValueError("PCA has zero numerically usable rank")
    scores = scores[:, :usable]
    eigenvalues = eigenvalues[:usable]
    transformed = np.asarray(scores, dtype=np.float64)
    transformed /= np.sqrt(eigenvalues + float(whitening_eps))[None, :]
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("zero norm after PCA whitening")
    transformed /= norms
    if not np.isfinite(transformed).all():
        raise ValueError("non-finite PCA-whitened features")
    return transformed, {
        "input_shape": list(values.shape),
        "output_shape": list(transformed.shape),
        "full_usable_rank": usable,
        "eigenvalue_floor": float(eigenvalue_floor),
        "whitening_eps": float(whitening_eps),
        "truncated": False,
        "solver": "sklearn_full",
    }
