from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import time

import numpy as np


@dataclass(frozen=True)
class SolverConfig:
    max_iter: int = 1000
    tolerance: float = 1e-9
    line_search: str = "exact_quadratic"
    init: str = "uniform"
    log_every: int = 1
    max_exact_gradient_n: int = 1000
    backend: str = "numpy"
    row_batch_size: int = 64
    cache_query_indices: bool = False
    init_seed: int | None = None
    sinkhorn_epsilon: float = 0.5
    sinkhorn_max_iter: int = 500
    sinkhorn_tolerance: float = 1e-12
    mutualnn_k: int = 10


@dataclass
class SolverLog:
    objectives: list[float]
    rel_changes: list[float]
    step_sizes: list[float]
    converged: bool
    actual_iterations: int
    wall_clock_seconds: float
    solver_name: str = "exact_l1_conditional_gradient"
    solver_version: str = "0.1"
    initial_objective: float = float("nan")
    initialization_seconds: float = float("nan")
    initialization_metadata: dict[str, object] | None = None
    fw_gaps: list[float] | None = None
    row_marginal_max_error: float = float("nan")
    column_marginal_max_error: float = float("nan")
    transport_min_value: float = float("nan")


@dataclass
class SolverResult:
    gw_distance: float
    final_objective: float
    transport_plan: np.ndarray
    log: SolverLog


def uniform_marginals(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=np.float64)


def uniform_coupling(n: int) -> np.ndarray:
    p = uniform_marginals(n)
    return np.outer(p, p)


def validate_distance_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"{name} must be square, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if np.max(np.abs(x - x.T)) > 1e-8:
        raise ValueError(f"{name} must be symmetric")
    if np.max(np.abs(np.diag(x))) > 1e-8:
        raise ValueError(f"{name} diagonal must be zero")
    return x


def validate_transport_plan(plan: np.ndarray, atol: float = 1e-8) -> dict[str, float | bool]:
    t = np.asarray(plan, dtype=np.float64)
    if t.ndim != 2 or t.shape[0] != t.shape[1]:
        raise ValueError(f"Transport plan must be square for this reproduction, got {t.shape}")
    n = t.shape[0]
    p = uniform_marginals(n)
    row_err = float(np.max(np.abs(t.sum(axis=1) - p)))
    col_err = float(np.max(np.abs(t.sum(axis=0) - p)))
    min_value = float(t.min())
    return {
        "nonnegative": bool(min_value >= -atol),
        "row_marginal_max_error": row_err,
        "column_marginal_max_error": col_err,
        "min_value": min_value,
        "valid": bool(min_value >= -atol and row_err <= atol and col_err <= atol),
    }


def _l1_gw_cost_matrix(
    dx: np.ndarray,
    dy: np.ndarray,
    transport: np.ndarray,
) -> np.ndarray:
    """Return M[i, j] = sum_{k,l} |dx[i,k] - dy[j,l]| * transport[k,l].

    For every fixed row of ``dx``, the weighted absolute-value queries are
    answered by sorting that row and using prefix sums. This is an exact
    computation of the L1 kernel with O(n^3) arithmetic and O(n^2) working
    memory, without materializing an n^4 tensor.
    """

    cx = validate_distance_matrix(dx, "dx")
    cy = validate_distance_matrix(dy, "dy")
    t = np.asarray(transport, dtype=np.float64)
    if t.shape != (cx.shape[0], cy.shape[0]):
        raise ValueError(f"Transport shape {t.shape} does not match distance matrices")
    n_source = cx.shape[0]
    n_target = cy.shape[0]
    cost = np.empty((n_source, n_target), dtype=np.float64)
    target_columns = np.arange(n_target)

    for i in range(n_source):
        source_row = cx[i]
        order = np.argsort(source_row, kind="mergesort")
        source_sorted = source_row[order]
        sorted_weights = t[order, :]
        prefix_mass = np.cumsum(sorted_weights, axis=0)
        prefix_weighted_source = np.cumsum(source_sorted[:, None] * sorted_weights, axis=0)
        total_mass = prefix_mass[-1]
        total_weighted_source = prefix_weighted_source[-1]

        # Each target-distance column is queried against the same sorted
        # source row, so searchsorted avoids a dense four-index tensor.
        query_indices = np.searchsorted(source_sorted, cy, side="right")
        prefix_mass_padded = np.vstack([np.zeros((1, n_target)), prefix_mass])
        prefix_weighted_padded = np.vstack([np.zeros((1, n_target)), prefix_weighted_source])
        mass_left = prefix_mass_padded[query_indices, target_columns[None, :]]
        weighted_left = prefix_weighted_padded[query_indices, target_columns[None, :]]
        target_values = cy
        cost[i] = (
            target_values * (2.0 * mass_left - total_mass[None, :])
            + total_weighted_source[None, :]
            - 2.0 * weighted_left
        ).sum(axis=1)
    return cost


def l1_gw_objective_fast(dx: np.ndarray, dy: np.ndarray, transport: np.ndarray) -> float:
    """Exact L1 GW objective using the prefix-sum acceleration."""

    cx = validate_distance_matrix(dx, "dx")
    cy = validate_distance_matrix(dy, "dy")
    t = np.asarray(transport, dtype=np.float64)
    if t.shape != (cx.shape[0], cy.shape[0]):
        raise ValueError(f"Transport shape {t.shape} does not match distance matrices")
    cost = _l1_gw_cost_matrix(cx, cy, t)
    return float(np.sum(cost * t))


def l1_gw_objective(
    dx: np.ndarray,
    dy: np.ndarray,
    transport: np.ndarray,
    *,
    max_bruteforce_n: int = 64,
) -> float:
    """Independent direct L1 GW objective audit for small inputs.

    The direct four-index contraction is intentionally retained as an
    independent audit implementation. Production optimization uses
    :func:`l1_gw_objective_fast`.
    """

    cx = validate_distance_matrix(dx, "dx")
    cy = validate_distance_matrix(dy, "dy")
    t = np.asarray(transport, dtype=np.float64)
    if t.shape != (cx.shape[0], cy.shape[0]):
        raise ValueError(f"Transport shape {t.shape} does not match distance matrices")
    if max(cx.shape[0], cy.shape[0]) > max_bruteforce_n:
        raise RuntimeError(
            f"Independent brute-force L1 objective is limited to n<={max_bruteforce_n}; "
            "use l1_gw_objective_fast for production-scale exact evaluation."
        )
    total = 0.0
    for i in range(cx.shape[0]):
        for j in range(cx.shape[1]):
            diff = np.abs(cx[i, j] - cy)
            total += float(np.sum(diff * t[i, :, None] * t[j, None, :]))
    return total


def square_gw_objective(dx: np.ndarray, dy: np.ndarray, transport: np.ndarray) -> float:
    cx = validate_distance_matrix(dx, "dx")
    cy = validate_distance_matrix(dy, "dy")
    t = np.asarray(transport, dtype=np.float64)
    total = 0.0
    n = cx.shape[0]
    for i in range(n):
        diff = cx[i, :, None, None] - cy[None, None, :, :]
        total += float(np.einsum("abcd,ac,bd->", diff * diff, t[i : i + 1, :], t, optimize=True))
    return total


def l1_gw_gradient(dx: np.ndarray, dy: np.ndarray, transport: np.ndarray, max_exact_n: int = 1000) -> np.ndarray:
    cx = validate_distance_matrix(dx, "dx")
    cy = validate_distance_matrix(dy, "dy")
    t = np.asarray(transport, dtype=np.float64)
    if t.shape != (cx.shape[0], cy.shape[0]):
        raise ValueError(f"Transport shape {t.shape} does not match distance matrices")
    if max(cx.shape[0], cy.shape[0]) > max_exact_n:
        raise RuntimeError(
            f"Exact L1 gradient is guarded at n<={max_exact_n}; input shape "
            f"{cx.shape[0]}x{cy.shape[0]} exceeds the configured safety limit."
        )
    forward = _l1_gw_cost_matrix(cx, cy, t)
    if np.allclose(cx, cx.T, rtol=0.0, atol=1e-10) and np.allclose(cy, cy.T, rtol=0.0, atol=1e-10):
        return 2.0 * forward
    reverse = _l1_gw_cost_matrix(cx.T, cy.T, t)
    return forward + reverse


def _assignment_uniform(cost: np.ndarray) -> np.ndarray:
    c = np.asarray(cost, dtype=np.float64)
    n = c.shape[0]
    if c.shape[0] != c.shape[1]:
        raise ValueError("Only equal-size uniform OT is supported")
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(c)
    except Exception:
        if n > 8:
            raise RuntimeError("scipy is required for linear OT when n > 8")
        best_cols = None
        best_cost = float("inf")
        for cols_candidate in permutations(range(n)):
            value = float(c[np.arange(n), cols_candidate].sum())
            if value < best_cost:
                best_cost = value
                best_cols = cols_candidate
        rows = np.arange(n)
        cols = np.asarray(best_cols, dtype=np.int64)
    plan = np.zeros((n, n), dtype=np.float64)
    plan[rows, cols] = 1.0 / n
    return plan


def _permutation_objective(dx: np.ndarray, dy: np.ndarray, plan: np.ndarray) -> float:
    """Exact objective of a uniform permutation coupling in O(n^2)."""

    n = plan.shape[0]
    permutation = np.argmax(plan, axis=1)
    return float(np.mean(np.abs(dx - dy[np.ix_(permutation, permutation)])))


def _objective_backend(dx: np.ndarray, dy: np.ndarray, transport: np.ndarray, backend: str, workspace=None) -> float:
    if backend in {"torch", "torch_cuda"}:
        if backend == "torch_cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("torch_cuda backend requested but CUDA is unavailable")
            except ImportError as exc:
                raise RuntimeError("torch_cuda backend requested but PyTorch is unavailable") from exc
        if workspace is not None:
            return workspace.objective(transport)
        from src.ot.l1_gw_torch import l1_gw_objective_torch

        return l1_gw_objective_torch(dx, dy, transport)
    if backend != "numpy":
        raise ValueError(f"Unsupported L1 GW backend: {backend}")
    return l1_gw_objective_fast(dx, dy, transport)


def _gradient_backend(
    dx: np.ndarray,
    dy: np.ndarray,
    transport: np.ndarray,
    *,
    backend: str,
    max_exact_n: int,
    workspace=None,
) -> np.ndarray:
    if backend in {"torch", "torch_cuda"}:
        if backend == "torch_cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("torch_cuda backend requested but CUDA is unavailable")
            except ImportError as exc:
                raise RuntimeError("torch_cuda backend requested but PyTorch is unavailable") from exc
        if workspace is not None:
            return workspace.gradient(transport)
        if max(dx.shape[0], dy.shape[0]) > max_exact_n:
            raise RuntimeError(
                f"Exact L1 gradient is guarded at n<={max_exact_n}; input shape "
                f"{dx.shape[0]}x{dy.shape[0]} exceeds the configured safety limit."
            )
        from src.ot.l1_gw_torch import l1_gw_gradient_torch

        return l1_gw_gradient_torch(dx, dy, transport)
    return l1_gw_gradient(dx, dy, transport, max_exact_n)


def _quadratic_step(f0: float, linear: float, quadratic: float) -> float:
    candidates = [0.0, 1.0]
    if quadratic > 1e-18:
        eta = -linear / (2.0 * quadratic)
        if 0.0 < eta < 1.0:
            candidates.append(float(eta))
    values = [(f0 + eta * linear + eta * eta * quadratic, eta) for eta in candidates]
    return min(values, key=lambda item: item[0])[1]


def solve_l1_gw(
    dx: np.ndarray,
    dy: np.ndarray,
    config: SolverConfig | None = None,
    *,
    initial_transport: np.ndarray | None = None,
    initialization_metadata: dict[str, object] | None = None,
    initialization_seconds: float | None = None,
    workspace=None,
) -> SolverResult:
    cfg = config or SolverConfig()
    cx = validate_distance_matrix(dx, "dx")
    cy = validate_distance_matrix(dy, "dy")
    if cx.shape != cy.shape:
        raise ValueError(f"This reproduction expects equal sample counts, got {cx.shape} and {cy.shape}")
    n = cx.shape[0]
    init_started = time.perf_counter()
    if initial_transport is None:
        from src.ot.initialization import build_initialization

        initialization = build_initialization(
            cfg.init,
            n,
            seed=cfg.init_seed,
            mutualnn_k=cfg.mutualnn_k,
            sinkhorn_epsilon=cfg.sinkhorn_epsilon,
            sinkhorn_max_iter=cfg.sinkhorn_max_iter,
            sinkhorn_tolerance=cfg.sinkhorn_tolerance,
        )
        transport = initialization.transport
        init_meta = initialization.metadata
        init_seconds = initialization.seconds
    else:
        transport = np.asarray(initial_transport, dtype=np.float64)
        init_meta = dict(initialization_metadata or {})
        init_seconds = (
            float(initialization_seconds)
            if initialization_seconds is not None
            else float(time.perf_counter() - init_started)
        )
    init_audit = validate_transport_plan(transport)
    if not bool(init_audit["valid"]):
        raise ValueError(f"Invalid initial transport plan: {init_audit}")
    if initialization_seconds is not None and initial_transport is not None:
        init_seconds = float(initialization_seconds)
    if workspace is None and cfg.backend in {"torch", "torch_cuda"}:
        from src.ot.l1_gw_torch import L1TorchWorkspace

        workspace = L1TorchWorkspace(
            cx,
            cy,
            row_batch_size=cfg.row_batch_size,
            cache_query_indices=cfg.cache_query_indices,
        )
    objectives = [_objective_backend(cx, cy, transport, cfg.backend, workspace)]
    rel_changes: list[float] = []
    step_sizes: list[float] = []
    fw_gaps: list[float] = []
    t0 = time.perf_counter()
    converged = False

    for iteration in range(1, cfg.max_iter + 1):
        grad = _gradient_backend(cx, cy, transport, backend=cfg.backend, max_exact_n=cfg.max_exact_gradient_n, workspace=workspace)
        target = _assignment_uniform(grad)
        direction = target - transport
        previous = objectives[-1]
        fw_gaps.append(float(-np.sum(grad * direction)))
        if cfg.line_search in {"exact_quadratic", "quadratic_interpolation"}:
            linear = float(np.sum(grad * direction))
            # F is homogeneous quadratic in the coupling.  Since ``target``
            # is a permutation coupling, its objective has an exact O(n^2)
            # evaluation; this identity avoids a second O(n^3) prefix-sum
            # pass while preserving the original exact line-search value:
            # F(D) = F(target) - F(T) - <grad(T), D>.
            target_objective = _permutation_objective(cx, cy, target)
            quadratic = target_objective - previous - linear
            eta = _quadratic_step(previous, linear, quadratic)
            # F is homogeneous quadratic in the coupling.  Reusing the
            # exact line polynomial avoids a second full prefix-sum objective
            # evaluation after the update while remaining algebraically exact.
            objective = previous + eta * linear + eta * eta * quadratic
        else:
            eta = 2.0 / (iteration + 1.0)
            objective = float("nan")
        transport = transport + eta * direction
        if cfg.line_search not in {"exact_quadratic", "quadratic_interpolation"}:
            objective = _objective_backend(cx, cy, transport, cfg.backend, workspace)
        rel = abs(previous - objective) / max(abs(previous), 1e-30)
        objectives.append(float(objective))
        rel_changes.append(float(rel))
        step_sizes.append(float(eta))
        if rel <= cfg.tolerance:
            converged = True
            break

    elapsed = time.perf_counter() - t0
    log = SolverLog(
        objectives=objectives,
        rel_changes=rel_changes,
        step_sizes=step_sizes,
        converged=converged,
        actual_iterations=len(objectives) - 1,
        wall_clock_seconds=float(elapsed),
        initial_objective=float(objectives[0]),
        initialization_seconds=float(init_seconds),
        initialization_metadata={**init_meta, **init_audit},
        fw_gaps=fw_gaps,
        row_marginal_max_error=float(validate_transport_plan(transport)["row_marginal_max_error"]),
        column_marginal_max_error=float(validate_transport_plan(transport)["column_marginal_max_error"]),
        transport_min_value=float(np.min(transport)),
    )
    return SolverResult(
        gw_distance=float(objectives[-1]),
        final_objective=float(objectives[-1]),
        transport_plan=transport,
        log=log,
    )
