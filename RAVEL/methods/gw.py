from __future__ import annotations

from dataclasses import asdict

import numpy as np

from methods.base import Method, validate_inputs
from src.distances.scaling import median_ratio_match_visual_to_text
from src.ot.initialization import build_initialization
from src.ot.l1_gw import SolverConfig, solve_l1_gw


class GW(Method):
    """Paper exact L1 angular GW with uniform initialization."""

    name = "gw"

    @staticmethod
    def _paper_angular_distance(features):
        values = np.asarray(features, dtype=np.float64)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("angular distance is undefined for zero-norm rows")
        normalized = values / norms
        cosine_raw = normalized @ normalized.T
        cosine = np.clip(cosine_raw, -1.0 + 1e-7, 1.0 - 1e-7)
        distance = np.arccos(cosine)
        distance = 0.5 * (distance + distance.T)
        np.fill_diagonal(distance, 0.0)
        offdiag = distance[~np.eye(len(distance), dtype=bool)]
        return distance, {
            "kind": "angular",
            "n": len(distance),
            "clip_epsilon": 1e-7,
            "cosine_min_before_clip": float(cosine_raw.min()),
            "cosine_max_before_clip": float(cosine_raw.max()),
            "clip_count": int(np.count_nonzero(cosine != cosine_raw)),
            "offdiag_median": float(np.median(offdiag)),
            "finite": bool(np.isfinite(distance).all()),
        }

    def evaluate(self, visual_features, text_features, config):
        visual, text = validate_inputs(visual_features, text_features)
        cfg = dict(config or {})
        visual_distance, visual_audit = self._paper_angular_distance(visual)
        text_distance, text_audit = self._paper_angular_distance(text)
        visual_distance, scale_audit = median_ratio_match_visual_to_text(
            visual_distance, text_distance
        )
        requested_backend = str(cfg.get("backend", "torch_cuda"))
        backend = requested_backend
        fallback = None
        if requested_backend == "torch_cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    backend = "numpy"
                    fallback = "torch_cuda_unavailable"
            except ImportError:
                backend = "numpy"
                fallback = "torch_unavailable"
        initialization = build_initialization("uniform", len(visual))
        solver = SolverConfig(
            max_iter=int(cfg.get("max_iter", 1000)),
            tolerance=float(cfg.get("tolerance", 1e-9)),
            line_search=str(cfg.get("line_search", "exact_quadratic")),
            backend=backend,
            max_exact_gradient_n=int(cfg.get("max_exact_gradient_n", len(visual))),
            row_batch_size=int(cfg.get("row_batch_size", min(64, len(visual)))),
            init="uniform",
        )
        result = solve_l1_gw(
            visual_distance,
            text_distance,
            solver,
            initial_transport=initialization.transport,
            initialization_metadata=initialization.metadata,
            initialization_seconds=initialization.seconds,
        )
        log = result.log
        return {
            "raw_score": float(result.final_objective),
            "final_score": -float(result.final_objective),
            "objective": float(result.final_objective),
            "score_direction": "lower_raw_higher_final",
            "distance": "angular",
            "calibration": "visual_to_text_median_ratio",
            "loss": "l1",
            "initialization": "uniform",
            "paired_lambda": 0.0,
            "requested_backend": requested_backend,
            "solver_backend": backend,
            "backend_fallback": fallback,
            "converged": bool(log.converged),
            "actual_iterations": int(log.actual_iterations),
            "initial_objective": float(log.initial_objective),
            "row_marginal_max_error": float(log.row_marginal_max_error),
            "column_marginal_max_error": float(log.column_marginal_max_error),
            "solver_config": asdict(solver),
            "visual_distance_audit": visual_audit,
            "text_distance_audit": text_audit,
            "scale_audit": asdict(scale_audit),
            "method_variant": "paper_exact_l1_angular_uniform",
        }
