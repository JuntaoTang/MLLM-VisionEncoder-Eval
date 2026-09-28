from __future__ import annotations

import numpy as np

from methods.base import Method
from src.neighbors import binary_overlap, topk_neighbors
from src.patch_similarity import (
    fit_full_rank_patch_pca,
    symmetric_chamfer_similarity,
    transform_patches_full_rank,
)
from src.preprocessing import EIGENVALUE_FLOOR, full_rank_pca_whiten_l2


class RAVEL(Method):
    """Full-rank PCA-whitened patch-set RAVEL with binary neighborhood overlap."""

    name = "ravel"

    def evaluate(self, visual_features, text_features, config):
        patches = np.asarray(visual_features)
        text = np.asarray(text_features)
        if patches.ndim != 3 or text.ndim != 2 or patches.shape[0] != text.shape[0]:
            raise ValueError(
                f"expected patches [n,t,d] and text [n,d], got {patches.shape} and {text.shape}"
            )
        if not np.isfinite(patches).all() or not np.isfinite(text).all():
            raise ValueError("RAVEL inputs contain NaN/Inf")
        cfg = dict(config or {})
        n = int(patches.shape[0])
        k = int(cfg.get("k", 100))
        if not 0 < k < n:
            raise ValueError(f"k must be in [1, n-1], got k={k}, n={n}")
        whitening_eps = float(cfg.get("whitening_eps", 1e-4))
        eigenvalue_floor = float(cfg.get("eigenvalue_floor", EIGENVALUE_FLOOR))
        device = cfg.get("device")
        text_transformed, text_meta = full_rank_pca_whiten_l2(
            text,
            whitening_eps=whitening_eps,
            eigenvalue_floor=eigenvalue_floor,
            random_state=int(cfg.get("random_state", 42)),
        )
        text_similarity = np.asarray(text_transformed @ text_transformed.T, dtype=np.float32)
        np.fill_diagonal(text_similarity, -np.inf)
        text_neighbors = topk_neighbors(text_similarity, k, device=device)
        text_neighbors_source = "recomputed_from_features"
        stored_text_neighbors = cfg.get("text_neighbors")
        text_neighbors_path = cfg.get("text_neighbors_path")
        if stored_text_neighbors is not None and text_neighbors_path:
            raise ValueError("provide only one of text_neighbors or text_neighbors_path")
        if text_neighbors_path:
            text_neighbors = np.load(text_neighbors_path, allow_pickle=False)
            text_neighbors_source = str(text_neighbors_path)
        elif stored_text_neighbors is not None:
            text_neighbors = np.asarray(stored_text_neighbors)
            text_neighbors_source = "config.text_neighbors"
        text_neighbors = np.asarray(text_neighbors, dtype=np.int32)
        if text_neighbors.shape != (n, k):
            raise ValueError(
                f"text neighbors must have shape {(n, k)}, got {text_neighbors.shape}"
            )
        pca_state = fit_full_rank_patch_pca(
            patches,
            eigenvalue_floor=eigenvalue_floor,
            image_batch=int(cfg.get("pca_image_batch", 8)),
        )
        patch_transformed = transform_patches_full_rank(
            patches,
            pca_state,
            whitening_eps=whitening_eps,
            image_batch=int(cfg.get("transform_image_batch", 8)),
        )
        visual_similarity = symmetric_chamfer_similarity(
            patch_transformed,
            device=device,
            normalize_tokens=False,
        )
        visual_neighbors = topk_neighbors(visual_similarity, k, device=device)
        score = binary_overlap(visual_neighbors, text_neighbors)
        return {
            "raw_score": score,
            "final_score": score,
            "k": k,
            "whitening_eps": whitening_eps,
            "eigenvalue_floor": eigenvalue_floor,
            "visual_full_usable_rank": int(pca_state["meta"]["full_usable_rank"]),
            "text_full_usable_rank": int(text_meta["full_usable_rank"]),
            "pca_truncation": False,
            "patch_similarity": "symmetric_bidirectional_chamfer",
            "text_similarity": "cosine",
            "text_neighbors_source": text_neighbors_source,
            "overlap": "binary_topk_intersection_divided_by_k",
            "rank_weighting": False,
            "hub_correction": False,
            "chance_adjustment": False,
            "score_direction": "higher_is_better",
            "method_variant": "full_rank_pca_whiten_patch_set_binary_overlap",
        }
