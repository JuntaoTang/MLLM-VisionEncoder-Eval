"""Abstract model interface for OpenFlamingo-style evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from PIL import Image


class BaseOFEvalModel(ABC):
    """Minimal interface: generate continuations or score completions."""

    @abstractmethod
    def generate(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        *,
        max_new_tokens: int,
        num_beams: int = 3,
        stop_strings: Sequence[str] | None = None,
    ) -> str:
        """Continue ``prompt`` conditioned on interleaved images (one per ``<image>``)."""

    @abstractmethod
    def score_completions(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        completions: Sequence[str],
        *,
        length_normalize: bool = False,
    ) -> list[float]:
        """Return log-prob of each completion (higher=better).

        By default returns sum log-prob over completion tokens; if
        ``length_normalize``, returns mean log-prob (better for ranking
        captions of unequal length).
        """
