from __future__ import annotations

import numpy as np

from .base import Method, validate_inputs
from .neighbors import cosine_similarity, pearson_vector


class RSA(Method):
    """Paper RSA: row-L2, within-modality cosine, Pearson upper triangle."""

    name = "rsa"

    def evaluate(self, visual_features, text_features, config):
        visual, text = validate_inputs(visual_features, text_features)
        cfg = dict(config or {})
        device = cfg.get("device")
        n = int(visual.shape[0])
        if n < 2:
            raise ValueError("RSA requires at least two aligned samples")
        visual_similarity = cosine_similarity(visual, device=device)
        text_similarity = cosine_similarity(text, device=device)
        upper = np.triu_indices(n, 1)
        score = pearson_vector(
            np.clip(visual_similarity[upper], -1.0, 1.0),
            np.clip(text_similarity[upper], -1.0, 1.0),
        )
        return {
            "raw_score": score,
            "final_score": score,
            "n_samples_used": n,
            "preprocessing": "row_l2_only",
            "similarity": "within_modality_cosine",
            "correlation": "pearson_upper_triangle",
            "score_direction": "higher_is_better",
            "method_variant": "paper_raw_cosine_rsa",
        }
