"""Feasible transport-plan initializers for the exact L1 GW solver.

The routines in this module deliberately do not depend on the solver.  This
keeps initialization auditable and makes it possible to validate a plan before
the conditional-gradient iterations start.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InitializationResult:
    transport: np.ndarray
    metadata: dict[str, object]
    seconds: float


def audit_coupling(plan: np.ndarray, *, atol: float = 1e-8) -> dict[str, object]:
    """Return a complete finite/nonnegative/uniform-marginal audit."""

    t = np.asarray(plan, dtype=np.float64)
    shape_ok = bool(t.ndim == 2 and t.shape[0] == t.shape[1])
    if not shape_ok:
        return {
            "shape": list(t.shape),
            "shape_ok": False,
            "finite": False,
            "nonnegative": False,
            "row_marginal_max_error": float("inf"),
            "column_marginal_max_error": float("inf"),
            "min_value": float("nan"),
            "valid": False,
        }
    n = int(t.shape[0])
    finite = bool(np.isfinite(t).all())
    min_value = float(np.min(t)) if t.size else float("nan")
    if finite:
        target = 1.0 / n
        row_error = float(np.max(np.abs(t.sum(axis=1) - target)))
        column_error = float(np.max(np.abs(t.sum(axis=0) - target)))
    else:
        row_error = float("inf")
        column_error = float("inf")
    nonnegative = bool(finite and min_value >= -atol)
    valid = bool(
        finite
        and nonnegative
        and row_error <= atol
        and column_error <= atol
    )
    return {
        "shape": list(t.shape),
        "shape_ok": True,
        "finite": finite,
        "nonnegative": nonnegative,
        "row_marginal_max_error": row_error,
        "column_marginal_max_error": column_error,
        "min_value": min_value,
        "valid": valid,
    }


def uniform_coupling(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return np.full((n, n), 1.0 / float(n * n), dtype=np.float64)


def random_permutation_coupling(n: int, seed: int) -> np.ndarray:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(n)
    plan = np.zeros((n, n), dtype=np.float64)
    plan[np.arange(n), permutation] = 1.0 / float(n)
    return plan


def sinkhorn_balance(
    kernel: np.ndarray,
    *,
    max_iter: int = 500,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, int, float]:
    """Scale a strictly nonnegative kernel to uniform marginals."""

    k = np.asarray(kernel, dtype=np.float64).copy()
    if k.ndim != 2 or k.shape[0] != k.shape[1]:
        raise ValueError(f"Sinkhorn kernel must be square, got {k.shape}")
    if not np.isfinite(k).all() or np.any(k < 0):
        raise ValueError("Sinkhorn kernel must be finite and nonnegative")
    n = int(k.shape[0])
    # A tiny strictly-positive floor prevents a disconnected random or mutual
    # kNN graph from producing zero rows/columns during scaling.
    k = np.maximum(k, np.finfo(np.float64).tiny)
    target = 1.0 / float(n)
    last_error = float("inf")
    for iteration in range(1, int(max_iter) + 1):
        row_sum = k.sum(axis=1)
        if np.any(row_sum <= 0) or not np.isfinite(row_sum).all():
            raise FloatingPointError("Sinkhorn encountered an invalid row sum")
        k *= (target / row_sum)[:, None]
        column_sum = k.sum(axis=0)
        if np.any(column_sum <= 0) or not np.isfinite(column_sum).all():
            raise FloatingPointError("Sinkhorn encountered an invalid column sum")
        k *= (target / column_sum)[None, :]
        last_error = max(
            float(np.max(np.abs(k.sum(axis=1) - target))),
            float(np.max(np.abs(k.sum(axis=0) - target))),
        )
        if last_error <= tolerance:
            return k, iteration, last_error
    return k, int(max_iter), last_error


def random_sinkhorn_coupling(
    n: int,
    seed: int,
    *,
    epsilon: float = 0.5,
    max_iter: int = 500,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, dict[str, object]]:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    rng = np.random.default_rng(int(seed))
    logits = rng.standard_normal((n, n))
    logits -= float(np.max(logits))
    kernel = np.exp(logits / float(epsilon))
    kernel += np.finfo(np.float64).tiny
    plan, iterations, error = sinkhorn_balance(
        kernel, max_iter=max_iter, tolerance=tolerance
    )
    return plan, {
        "sinkhorn_epsilon": float(epsilon),
        "sinkhorn_iterations": int(iterations),
        "sinkhorn_final_marginal_error": float(error),
    }


def _topk_indices(similarity: np.ndarray, k: int) -> np.ndarray:
    n, m = similarity.shape
    if k <= 0 or k > m:
        raise ValueError(f"k must be in [1, {m}], got {k}")
    candidates = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    scores = np.take_along_axis(similarity, candidates, axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.take_along_axis(candidates, order, axis=1)


def mutualnn_warm_start_coupling(
    vision_features: np.ndarray,
    text_features: np.ndarray,
    *,
    k: int = 10,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_max_iter: int = 500,
    sinkhorn_tolerance: float = 1e-12,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build an informed feasible plan from mutual cross-modal kNN edges.

    MutualNN is used only to construct the starting kernel.  Sinkhorn scaling
    then enforces the exact uniform marginals required by the GW solver.  A
    tiny dense floor makes the feasibility step well-defined even when the
    mutual graph is disconnected.
    """

    x = np.asarray(vision_features, dtype=np.float64)
    y = np.asarray(text_features, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"MutualNN features must be [n,d] with equal n, got {x.shape} and {y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("MutualNN features contain NaN or Inf")
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    if np.any(x_norm <= 1e-12) or np.any(y_norm <= 1e-12):
        raise ValueError("MutualNN features contain a zero-norm row")
    x = x / x_norm
    y = y / y_norm
    n = int(x.shape[0])
    if k <= 0 or k >= n:
        raise ValueError(f"MutualNN k must be in [1, n-1], got k={k}, n={n}")

    # This follows robust_fair_alignment.mutual_nn: build an independent
    # within-modality graph for each side (uncentered L2 cosine), then compare
    # neighbor sets.  It remains well-defined for heterogeneous feature
    # dimensions, unlike a raw cross-modal dot product.
    x_sim = x @ x.T
    y_sim = y @ y.T
    np.fill_diagonal(x_sim, -np.inf)
    np.fill_diagonal(y_sim, -np.inf)
    x_neighbors = _topk_indices(x_sim, int(k))
    y_neighbors = _topk_indices(y_sim, int(k))
    y_inverse: list[list[int]] = [[] for _ in range(n)]
    for j, row in enumerate(y_neighbors):
        for neighbor in row:
            y_inverse[int(neighbor)].append(j)

    # Sparse graph-profile overlap.  Convert the affinity to a Gibbs kernel so
    # ``sinkhorn_epsilon`` is an actual temperature parameter: epsilon->0
    # concentrates mass on high-overlap profiles, while large epsilon tends to
    # the uniform kernel.  The dense matrix is formed only after accumulation.
    affinity = np.zeros((n, n), dtype=np.float64)
    mutual_edges = 0
    for i, row in enumerate(x_neighbors):
        overlap_counts: dict[int, int] = {}
        for neighbor in row:
            for j in y_inverse[int(neighbor)]:
                overlap_counts[j] = overlap_counts.get(j, 0) + 1
        for j, count in overlap_counts.items():
            if count <= 0:
                continue
            # Requiring a reciprocal profile edge gives the "mutual" part of
            # MutualNN.  We still keep non-mutual overlap as a smaller signal
            # so every row has useful structure before Sinkhorn balancing.
            reciprocal = i in set(y_neighbors[j].tolist())
            if reciprocal:
                mutual_edges += 1
                affinity[i, j] = max(affinity[i, j], 1.0 + float(count) / float(k))
            else:
                affinity[i, j] = max(affinity[i, j], 0.25 * float(count) / float(k))
    if sinkhorn_epsilon <= 0 or not np.isfinite(sinkhorn_epsilon):
        raise ValueError(f"sinkhorn_epsilon must be finite and positive, got {sinkhorn_epsilon}")
    kernel = np.exp(np.clip(affinity / float(sinkhorn_epsilon), -700.0, 700.0))
    plan, iterations, error = sinkhorn_balance(
        kernel,
        max_iter=sinkhorn_max_iter,
        tolerance=sinkhorn_tolerance,
    )
    return plan, {
        "mutualnn_k": int(k),
        "mutualnn_centering": "none",
        "mutualnn_normalization": "l2",
        "mutualnn_cross_distance": "within_modality_cosine_graph_profile",
        "mutualnn_edges": int(mutual_edges),
        "sinkhorn_epsilon": float(sinkhorn_epsilon),
        "sinkhorn_kernel": "exp(graph_profile_affinity / epsilon)",
        "sinkhorn_iterations": int(iterations),
        "sinkhorn_final_marginal_error": float(error),
    }


def build_initialization(
    name: str,
    n: int,
    *,
    seed: int | None = None,
    vision_features: np.ndarray | None = None,
    text_features: np.ndarray | None = None,
    mutualnn_k: int = 10,
    sinkhorn_epsilon: float = 0.5,
    sinkhorn_max_iter: int = 500,
    sinkhorn_tolerance: float = 1e-12,
) -> InitializationResult:
    """Create one named initialization and validate it before returning."""

    started = time.perf_counter()
    metadata: dict[str, object] = {
        "initialization": str(name),
        "initialization_seed": "" if seed is None else int(seed),
    }
    if name == "uniform":
        plan = uniform_coupling(n)
        metadata["initialization_source"] = "uniform_product_coupling"
    elif name == "random_sinkhorn":
        if seed is None:
            raise ValueError("random_sinkhorn requires an explicit seed")
        plan, extra = random_sinkhorn_coupling(
            n,
            seed,
            epsilon=sinkhorn_epsilon,
            max_iter=sinkhorn_max_iter,
            tolerance=sinkhorn_tolerance,
        )
        metadata.update(extra)
        metadata["initialization_source"] = "random_logits_sinkhorn"
    elif name == "random_permutation":
        if seed is None:
            raise ValueError("random_permutation requires an explicit seed")
        plan = random_permutation_coupling(n, seed)
        metadata["initialization_source"] = "random_permutation_matrix"
    elif name == "identity_diagnostic":
        plan = np.eye(n, dtype=np.float64) / float(n)
        metadata["initialization_seed"] = 0
        metadata["initialization_source"] = "identity_index_coupling"
    elif name == "mutualnn_warm_start":
        if vision_features is None or text_features is None:
            raise ValueError("mutualnn_warm_start requires vision_features and text_features")
        plan, extra = mutualnn_warm_start_coupling(
            vision_features,
            text_features,
            k=mutualnn_k,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_max_iter=sinkhorn_max_iter,
            sinkhorn_tolerance=sinkhorn_tolerance,
        )
        metadata.update(extra)
        metadata["initialization_seed"] = "deterministic"
        metadata["initialization_source"] = "mutualnn_cross_modal_k10_sinkhorn"
    else:
        raise ValueError(f"Unsupported initialization {name!r}")
    audit = audit_coupling(plan)
    if not bool(audit["valid"]):
        raise ValueError(f"Initializer {name} produced an infeasible coupling: {audit}")
    metadata.update(audit)
    return InitializationResult(
        transport=np.asarray(plan, dtype=np.float64),
        metadata=metadata,
        seconds=float(time.perf_counter() - started),
    )
