"""Minimal UniTok checkpoint args loader (inference-only)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Union


class Args:
    """Subset of UniTok training args stored in checkpoint metadata."""

    model: str = "vitamin_large"
    img_size: int = 256
    num_query: int = 0
    drop_path: float = 0.1
    vocab_size: int = 32768
    vocab_width: int = 64
    vq_beta: float = 0.25
    num_codebooks: int = 8
    quant_proj: str = "attn"
    le: float = 0.0
    e_temp: float = 0.01

    def state_dict(self, key_ordered: bool = True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        for k, v in self.__class__.__dict__.items():
            if not k.startswith("_") and not callable(v):
                d[k] = getattr(self, k)
        return d

    def load_state_dict(self, state_dict: dict) -> None:
        for k, v in state_dict.items():
            setattr(self, k, v)
