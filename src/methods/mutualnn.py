from __future__ import annotations

from .base import Method, validate_inputs
from .neighbors import cosine_similarity, fractional_overlap, fractional_topk_distribution


class MutualNN(Method):
    """Paper MutualNN with fractional boundary ties and chance adjustment."""

    name = "mutualnn"

    def evaluate(self, visual_features, text_features, config):
        visual, text = validate_inputs(visual_features, text_features)
        cfg = dict(config or {})
        n = int(visual.shape[0])
        k = int(cfg.get("k", 10))
        device = cfg.get("device")
        if not 0 < k < n:
            raise ValueError(f"k must be in [1, n-1], got k={k}, n={n}")
        visual_graph = fractional_topk_distribution(cosine_similarity(visual, device=device), k)
        text_graph = fractional_topk_distribution(cosine_similarity(text, device=device), k)
        raw_overlap = fractional_overlap(visual_graph, text_graph)
        chance = k / float(n - 1)
        final_score = (raw_overlap - chance) / (1.0 - chance)
        return {
            "raw_score": raw_overlap,
            "final_score": float(final_score),
            "chance": float(chance),
            "k": k,
            "preprocessing": "row_l2_only",
            "tie_mode": "fractional",
            "rank_mass": "uniform",
            "chance_adjustment": True,
            "score_direction": "higher_is_better",
            "method_variant": "paper_fractional_uniform_chance_adjusted",
        }
