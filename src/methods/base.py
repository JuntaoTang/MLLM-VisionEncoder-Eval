from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class Method(ABC):
    name: str
    requires_pool: bool = False

    @abstractmethod
    def evaluate(self, visual_features: np.ndarray, text_features: np.ndarray, config: dict[str, Any]) -> dict[str, Any]: ...

    def finalize_pool(self, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
        """Finalize scores that depend on the complete candidate pool."""

def validate_inputs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]: raise ValueError(f"expected [n,d] features, got {x.shape} and {y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError("features contain NaN/Inf")
    return x, y
