"""Few-shot demo selection helpers."""

from __future__ import annotations

import random
from typing import Sequence, TypeVar

T = TypeVar("T")


def sample_demos(pool: Sequence[T], k: int, *, seed: int, exclude_index=None) -> list[T]:
    """Sample ``k`` demos from ``pool``, optionally excluding one index field."""
    if k <= 0:
        return []
    candidates = list(pool)
    if exclude_index is not None:
        candidates = [c for c in candidates if getattr(c, "index", None) != exclude_index]
    if not candidates:
        return []
    rng = random.Random(seed)
    if k >= len(candidates):
        return list(candidates)
    return rng.sample(candidates, k)
