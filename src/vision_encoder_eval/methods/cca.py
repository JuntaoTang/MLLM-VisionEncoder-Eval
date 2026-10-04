from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import Method, validate_inputs


DEFAULT_ALPHAS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0)


@dataclass
class SVDBundle:
    u: Any
    s: Any
    vh: Any
    evaluation: Any
    train_size: int
    feature_dim: int


def make_split(n: int, seed: int) -> dict[str, np.ndarray]:
    train_size = 3 * n // 5
    validation_size = n // 5
    order = np.random.default_rng(seed).permutation(n).astype(np.int64)
    validation_end = train_size + validation_size
    return {
        "train": order[:train_size],
        "validation": order[train_size:validation_end],
        "test": order[validation_end:],
        "train_validation": order[:validation_end],
        "order": order,
    }


def fit_svd_bundle(values, fit_indices, evaluation_indices, device) -> SVDBundle:
    import torch

    train = torch.as_tensor(np.asarray(values[fit_indices]), dtype=torch.float64, device=device)
    evaluation = torch.as_tensor(
        np.asarray(values[evaluation_indices]), dtype=torch.float64, device=device
    )
    if not torch.isfinite(train).all() or not torch.isfinite(evaluation).all():
        raise ValueError("non-finite CCA feature values")
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, correction=1, keepdim=True)
    std = torch.where(std > 1e-12, std, torch.ones_like(std))
    train = (train - mean) / std
    evaluation = (evaluation - mean) / std
    u, s, vh = torch.linalg.svd(train, full_matrices=False)
    return SVDBundle(u, s, vh, evaluation, len(fit_indices), values.shape[1])


def heldout_score(
    visual: SVDBundle,
    text: SVDBundle,
    u_cross,
    *,
    alpha: float,
    components: int,
    oversampling: int,
    power_iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    import torch

    train_size = visual.train_size
    ridge = (train_size - 1) * alpha
    factor_x = visual.s / torch.sqrt(visual.s.square() + ridge)
    factor_y = text.s / torch.sqrt(text.s.square() + ridge)
    whitened_cross = factor_x[:, None] * u_cross * factor_y[None, :]
    q = min(components + oversampling, min(whitened_cross.shape))
    torch.manual_seed(random_seed)
    left, train_correlations, right = torch.svd_lowrank(
        whitened_cross, q=q, niter=power_iterations
    )
    count = min(components, len(train_correlations))
    left = left[:, :count]
    right = right[:, :count]
    train_correlations = train_correlations[:count]
    inverse_x = np.sqrt(train_size - 1) / torch.sqrt(visual.s.square() + ridge)
    inverse_y = np.sqrt(train_size - 1) / torch.sqrt(text.s.square() + ridge)
    wx = visual.vh.T @ (inverse_x[:, None] * left)
    wy = text.vh.T @ (inverse_y[:, None] * right)
    zx = visual.evaluation @ wx
    zy = text.evaluation @ wy
    zx = zx - zx.mean(dim=0, keepdim=True)
    zy = zy - zy.mean(dim=0, keepdim=True)
    numerator = (zx * zy).sum(dim=0)
    denominator = torch.sqrt(zx.square().sum(dim=0) * zy.square().sum(dim=0))
    correlations = numerator / torch.clamp(denominator, min=1e-30)
    if not torch.isfinite(correlations).all():
        raise ValueError("non-finite held-out canonical correlations")
    values = [float(value) for value in correlations.detach().cpu().tolist()]
    return {
        "score": float(np.mean(values)),
        "mean_abs_score": float(np.mean(np.abs(values))),
        "canonical_correlations": values,
        "train_regularized_canonical_correlations": [
            float(value) for value in train_correlations.detach().cpu().tolist()
        ],
    }


def cell_seed(base_seed: int, visual_id: str, text_id: str, alpha_index: int, phase: str) -> int:
    payload = f"{base_seed}|{visual_id}|{text_id}|{alpha_index}|{phase}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


class CCA(Method):
    """Validation-selected ridge CCA with an untouched held-out test split."""

    name = "cca"

    def evaluate(self, visual_features, text_features, config):
        import torch

        visual, text = validate_inputs(visual_features, text_features)
        cfg = dict(config or {})
        n = int(visual.shape[0])
        if n < 10:
            raise ValueError("CV-Ridge CCA requires at least 10 aligned samples")
        seed = int(cfg.get("seed", 42))
        components = int(cfg.get("components", cfg.get("n_components", 10)))
        alphas = tuple(float(value) for value in cfg.get("alphas", DEFAULT_ALPHAS))
        if not alphas or any(value <= 0 for value in alphas):
            raise ValueError("CCA alphas must be positive")
        device_name = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        device = torch.device(device_name)
        oversampling = int(cfg.get("oversampling", 10))
        power_iterations = int(cfg.get("power_iterations", 4))
        visual_id = str(cfg.get("visual_id", "visual"))
        text_id = str(cfg.get("text_id", "text"))
        split = make_split(n, seed)
        visual_selection = fit_svd_bundle(visual, split["train"], split["validation"], device)
        visual_final = fit_svd_bundle(visual, split["train_validation"], split["test"], device)
        text_selection = fit_svd_bundle(text, split["train"], split["validation"], device)
        text_final = fit_svd_bundle(text, split["train_validation"], split["test"], device)
        selection_cross = visual_selection.u.T @ text_selection.u
        validation = []
        for index, alpha in enumerate(alphas):
            result = heldout_score(
                visual_selection,
                text_selection,
                selection_cross,
                alpha=alpha,
                components=components,
                oversampling=oversampling,
                power_iterations=power_iterations,
                random_seed=cell_seed(seed, visual_id, text_id, index, "validation"),
            )
            validation.append({"alpha": alpha, **result})
        best_index = max(range(len(validation)), key=lambda index: (validation[index]["score"], -index))
        chosen_alpha = alphas[best_index]
        final_cross = visual_final.u.T @ text_final.u
        test = heldout_score(
            visual_final,
            text_final,
            final_cross,
            alpha=chosen_alpha,
            components=components,
            oversampling=oversampling,
            power_iterations=power_iterations,
            random_seed=cell_seed(seed, visual_id, text_id, best_index, "test"),
        )
        return {
            "raw_score": test["score"],
            "final_score": test["score"],
            "test_score": test["score"],
            "test_mean_abs_score": test["mean_abs_score"],
            "test_canonical_correlations": test["canonical_correlations"],
            "chosen_alpha": chosen_alpha,
            "chosen_alpha_index": best_index,
            "validation_score": validation[best_index]["score"],
            "validation_grid": validation,
            "n_samples_used": n,
            "split_sizes": {
                "train": len(split["train"]),
                "validation": len(split["validation"]),
                "test": len(split["test"]),
            },
            "preprocessing": "fit_fold_center_and_scale_no_pca_no_row_l2",
            "components": components,
            "device": device_name,
            "score_direction": "higher_is_better",
            "method_variant": "validation_selected_ridge_cca",
        }
