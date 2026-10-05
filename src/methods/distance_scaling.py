from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScaleAudit:
    visual_distance_median: float
    text_distance_median: float
    scale_factor: float
    scaled_visual_distance_median: float
    median_abs_error: float


def offdiag_values(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {x.shape}")
    return x[~np.eye(x.shape[0], dtype=bool)]


def offdiag_median(matrix: np.ndarray) -> float:
    values = offdiag_values(matrix)
    if values.size == 0:
        raise ValueError("Need at least two points for an off-diagonal median")
    if not np.isfinite(values).all():
        raise ValueError("Distance matrix contains NaN or Inf off diagonal")
    return float(np.median(values))


def median_ratio_match_visual_to_text(dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, ScaleAudit]:
    mx = offdiag_median(dx)
    my = offdiag_median(dy)
    if mx <= 0.0:
        raise ValueError(f"Visual off-diagonal median must be positive, got {mx}")
    scale = my / mx
    scaled = np.asarray(dx, dtype=np.float64) * scale
    np.fill_diagonal(scaled, 0.0)
    scaled_median = offdiag_median(scaled)
    return scaled, ScaleAudit(
        visual_distance_median=mx,
        text_distance_median=my,
        scale_factor=float(scale),
        scaled_visual_distance_median=float(scaled_median),
        median_abs_error=float(abs(scaled_median - my)),
    )
