"""Abstract interface for discrete visual tokenizers (VQGAN, VQVAE, dVAE, etc.)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseDiscreteTokenizer(nn.Module, ABC):
    """Abstract base for discrete visual tokenizers."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Size of the discrete codebook."""
        ...

    @property
    @abstractmethod
    def num_image_tokens(self) -> int:
        """Number of tokens produced per image (H_feat * W_feat)."""
        ...

    @property
    @abstractmethod
    def image_size(self) -> int:
        """Expected input image size (height = width = image_size)."""
        ...

    @abstractmethod
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images to discrete token indices or features."""
        ...

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encode(pixel_values)
