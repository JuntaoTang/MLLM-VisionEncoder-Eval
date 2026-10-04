from __future__ import annotations
import torch
import numpy as np
from torch import nn
NUM_CLASSES = 1000

class LinearHead(nn.Module):
    def __init__(self, in_dim: int, base_lr: float, effective_lr: float):
        super().__init__()
        self.base_lr = float(base_lr)
        self.effective_lr = float(effective_lr)
        self.linear = nn.Linear(in_dim, NUM_CLASSES, bias=True)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)

class BatchNormalizedLinearHead(LinearHead):
    """MAE-style non-affine BatchNorm followed by a linear classifier."""

    def __init__(self, in_dim: int, base_lr: float, effective_lr: float):
        super().__init__(in_dim, base_lr, effective_lr)
        self.batch_norm = nn.BatchNorm1d(
            in_dim,
            eps=1e-6,
            momentum=0.1,
            affine=False,
            track_running_stats=True,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(self.batch_norm(features))

class LinearHeadGrid(nn.Module):
    def __init__(self, heads: dict[str, LinearHead]):
        super().__init__()
        self.heads = nn.ModuleDict(heads)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(features) for name, head in self.heads.items()}

class LinearPostprocessor(nn.Module):
    def __init__(self, head: LinearHead):
        super().__init__()
        self.head = head

    def forward(self, features: torch.Tensor, targets: torch.Tensor):
        return {"preds": self.head(features), "target": targets}


def _nested_class_orders(targets: np.ndarray, seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    targets = np.asarray(targets, dtype=np.int64)
    if not np.array_equal(np.unique(targets), np.arange(NUM_CLASSES)):
        raise RuntimeError("Expected ImageNet labels 0..999")

    # Preserve the exact five selected examples from the earlier 5-shot run.
    legacy_rng = np.random.default_rng(seed)
    first_five = []
    candidates_by_class = []
    for class_id in range(NUM_CLASSES):
        candidates = np.flatnonzero(targets == class_id)
        if len(candidates) < 5:
            raise RuntimeError(f"Class {class_id} has fewer than five examples")
        candidates_by_class.append(candidates)
        first_five.append(legacy_rng.choice(candidates, 5, replace=False))

    orders = []
    for class_id, (candidates, prefix) in enumerate(zip(candidates_by_class, first_five)):
        remaining = candidates[~np.isin(candidates, prefix)]
        tail_rng = np.random.default_rng(np.random.SeedSequence([seed, class_id, 0x51D1D3]))
        tail = tail_rng.permutation(remaining)
        orders.append(np.concatenate((prefix, tail)).astype(np.int64, copy=False))
    counts = np.asarray([len(order) for order in orders], dtype=np.int64)
    return orders, counts


def _support_indices(orders: list[np.ndarray], cap_shot: int) -> np.ndarray:
    selected = np.concatenate([order[: min(cap_shot, len(order))] for order in orders])
    # Sorting matches the earlier SupportDataset representation and makes the
    # support hash independent of class concatenation order.
    return np.sort(selected.astype(np.int64, copy=False))


def _iter_batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]
