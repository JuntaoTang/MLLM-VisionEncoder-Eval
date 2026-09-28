from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AngularAudit:
    n: int
    dtype_in: str
    dtype_out: str
    zero_norm_count: int
    cosine_min_before_clip: float
    cosine_max_before_clip: float
    cosine_clip_count: int
    max_asymmetry: float
    diagonal_max_abs: float
    finite: bool


def clip_cosine_matrix(cosine_raw: np.ndarray) -> tuple[np.ndarray, int, float, float]:
    values = np.asarray(cosine_raw)
    if not np.isfinite(values).all():
        raise ValueError("Cosine matrix contains NaN or Inf before clipping")
    raw_min = float(np.min(values))
    raw_max = float(np.max(values))
    clipped = np.clip(values, -1.0, 1.0)
    clip_count = int(np.count_nonzero(clipped != values))
    return clipped, clip_count, raw_min, raw_max


def angular_distance_matrix(
    features: np.ndarray,
    *,
    dtype: np.dtype | str = np.float64,
    zero_norm_eps: float = 1e-12,
) -> tuple[np.ndarray, AngularAudit]:
    """Compute the paper angular distance matrix arccos(cosine).

    Zero-norm rows are rejected because their angular distance is undefined.
    """

    x = np.asarray(features, dtype=dtype)
    if x.ndim != 2:
        raise ValueError(f"Expected a [n, d] feature matrix, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Feature matrix contains NaN or Inf")
    norms = np.linalg.norm(x, axis=1)
    zero_norm = norms <= zero_norm_eps
    zero_count = int(zero_norm.sum())
    if zero_count:
        raise ValueError(f"Angular distance is undefined for {zero_count} zero-norm rows")

    x_norm = x / norms[:, None]
    cosine_raw = x_norm @ x_norm.T
    cosine, clip_count, raw_min, raw_max = clip_cosine_matrix(cosine_raw)
    dist = np.arccos(cosine)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    if not np.isfinite(dist).all():
        raise ValueError("Angular distance contains NaN or Inf")
    audit = AngularAudit(
        n=int(dist.shape[0]),
        dtype_in=str(np.asarray(features).dtype),
        dtype_out=str(dist.dtype),
        zero_norm_count=zero_count,
        cosine_min_before_clip=raw_min,
        cosine_max_before_clip=raw_max,
        cosine_clip_count=clip_count,
        max_asymmetry=float(np.max(np.abs(dist - dist.T))),
        diagonal_max_abs=float(np.max(np.abs(np.diag(dist)))),
        finite=bool(np.isfinite(dist).all()),
    )
    return dist, audit
