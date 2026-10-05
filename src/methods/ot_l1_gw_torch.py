from __future__ import annotations

import numpy as np

from .ot_l1_gw import validate_distance_matrix


def _as_cuda_tensor(array: np.ndarray):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch_cuda backend requested but CUDA is unavailable")
    device = torch.device("cuda")
    return torch.as_tensor(array, dtype=torch.float64, device=device)


class L1TorchWorkspace:
    """Reusable CUDA state for repeated exact L1 GW evaluations.

    The distance matrices and the per-source-row sorting are invariant over
    conditional-gradient iterations. Keeping them on the accelerator avoids
    repeated host-to-device copies and repeated ``sort`` calls while the
    transport plan changes.
    """

    def __init__(
        self,
        dx: np.ndarray,
        dy: np.ndarray,
        *,
        row_batch_size: int = 64,
        column_batch_size: int = 256,
        cache_query_indices: bool = False,
        device=None,
    ) -> None:
        import torch

        cx = validate_distance_matrix(dx, "dx")
        cy = validate_distance_matrix(dy, "dy")
        if cx.shape != cy.shape:
            raise ValueError(f"Distance matrices must have equal shape, got {cx.shape} and {cy.shape}")
        if row_batch_size <= 0:
            raise ValueError("row_batch_size must be positive")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        self.row_batch_size = int(row_batch_size)
        self.column_batch_size = int(column_batch_size)
        self.n = int(cx.shape[0])
        self.tx = torch.as_tensor(cx, dtype=torch.float64, device=self.device)
        self.ty = torch.as_tensor(cy, dtype=torch.float64, device=self.device)
        self.source_sorted, self.source_order = torch.sort(self.tx, dim=1, stable=True)
        self.columns = torch.arange(self.n, device=self.device)
        self._zero_prefix = torch.zeros(
            (self.row_batch_size, 1, self.n), dtype=torch.float64, device=self.device
        )
        # ``solve_l1_gw`` evaluates the initial objective and then its
        # gradient at the same plan.  Keep one cost matrix around so the
        # expensive exact contraction is not performed twice.  The cache key
        # is the identity of the caller's NumPy plan; every FW update creates
        # a new array, so stale values cannot be reused across iterations.
        self._cost_cache_key = None
        self._cost_cache = None
        self.query_indices = None
        if cache_query_indices:
            if self.n > np.iinfo(np.int16).max:
                raise ValueError("int16 query-index cache requires n <= 32767")
            self.query_indices = torch.empty(
                (self.n, self.n, self.n), dtype=torch.int16, device=self.device
            )
            for start in range(0, self.n, self.row_batch_size):
                stop = min(start + self.row_batch_size, self.n)
                batch_size = stop - start
                boundaries = self.source_sorted[start:stop].unsqueeze(1).expand(batch_size, self.n, self.n)
                values = self.ty.unsqueeze(0).expand(batch_size, self.n, self.n)
                self.query_indices[start:stop] = torch.searchsorted(
                    boundaries.contiguous(), values.contiguous(), right=True
                ).to(torch.int16)

    def transport_tensor(self, transport: np.ndarray):
        import torch

        if isinstance(transport, torch.Tensor):
            if transport.device != self.device or transport.dtype != torch.float64:
                return transport.to(device=self.device, dtype=torch.float64)
            return transport
        return torch.as_tensor(transport, dtype=torch.float64, device=self.device)

    def cost_matrix(self, transport):
        """Evaluate the exact L1 GW linearization with bounded temporaries.

        The original implementation expanded a dense ``[row_batch, n, n]``
        view of the transport plan.  At n=5000 that view becomes a multi-GB
        gather and is needlessly expensive for the permutation plans produced
        by the conditional-gradient solver.  This implementation keeps the
        same prefix-sum identity but processes one source row at a time for a
        dense plan, has a specialised uniform-plan path, and uses a tiled
        pairwise row-L1 kernel once a permutation coupling is reached.
        """

        import torch

        tt = self.transport_tensor(transport)
        if tuple(tt.shape) != (self.n, self.n):
            raise ValueError(f"Transport shape {tuple(tt.shape)} does not match {(self.n, self.n)}")
        # The solver passes the same NumPy plan to ``objective`` and then
        # ``gradient``.  Keying by the NumPy object's identity lets those two
        # calls share the exact cost matrix without ever reusing a value for a
        # mutated/new plan.  Torch inputs are treated as uncached because a
        # tensor may be modified in-place by a caller between evaluations.
        cache_key = id(transport) if isinstance(transport, np.ndarray) else None
        if cache_key is not None and cache_key == self._cost_cache_key and self._cost_cache is not None:
            return self._cost_cache
        if self._is_uniform_plan(tt):
            result = self._uniform_cost_matrix()
        else:
            permutation = self._permutation_indices(tt)
            if permutation is not None:
                result = self._permutation_cost_matrix(permutation)
            else:
                result = self._dense_cost_matrix(tt)
        if cache_key is not None:
            self._cost_cache_key = cache_key
            self._cost_cache = result
        return result

    def _is_uniform_plan(self, transport, *, atol: float = 1e-14) -> bool:
        import torch

        expected = 1.0 / float(self.n * self.n)
        # Checking a scalar plus marginals avoids a full host copy while still
        # rejecting plans that only happen to have uniform row sums.
        return bool(
            torch.allclose(transport[0, 0], torch.as_tensor(expected, dtype=torch.float64, device=self.device), atol=atol, rtol=0.0)
            and torch.max(torch.abs(transport - expected)) <= atol
        )

    def _permutation_indices(self, transport, *, atol: float = 5e-12):
        import torch

        values, indices = torch.max(transport, dim=1)
        target = 1.0 / float(self.n)
        if bool(torch.max(torch.abs(values - target)) > atol):
            return None
        if bool(torch.min(transport) < -atol):
            return None
        # A valid uniform permutation has one nonzero per row and column.  The
        # column check is done on-device and only the small index vector is
        # copied to the host for the uniqueness test.
        if bool(torch.max(torch.abs(transport.sum(dim=1) - target)) > atol):
            return None
        if bool(torch.max(torch.abs(transport.sum(dim=0) - target)) > atol):
            return None
        if torch.unique(indices).numel() != self.n:
            return None
        residual = transport.clone()
        residual[torch.arange(self.n, device=self.device), indices] = 0.0
        if bool(torch.max(torch.abs(residual)) > atol):
            return None
        return indices

    def _permutation_cost_matrix(self, permutation):
        """Exact cost for T[i, permutation[i]]=1/n using tiled row L1."""

        import torch

        # T[k,l]=1/n iff l=permutation[k], hence the GW linearisation is the
        # mean row-wise L1 distance between DX[i,:] and DY[j, permutation[:]].
        target_rows = self.ty[:, permutation]
        out = torch.empty((self.n, self.n), dtype=torch.float64, device=self.device)
        row_block = max(1, int(self.row_batch_size))
        col_block = max(1, int(self.column_batch_size))
        inner_block = max(256, min(self.n, 2048))
        with torch.no_grad():
            for istart in range(0, self.n, row_block):
                istop = min(self.n, istart + row_block)
                left = self.tx[istart:istop]
                for jstart in range(0, self.n, col_block):
                    jstop = min(self.n, jstart + col_block)
                    right = target_rows[jstart:jstop]
                    block = torch.zeros((istop - istart, jstop - jstart), dtype=torch.float64, device=self.device)
                    for kstart in range(0, self.n, inner_block):
                        kstop = min(self.n, kstart + inner_block)
                        block += torch.abs(left[:, None, kstart:kstop] - right[None, :, kstart:kstop]).sum(dim=2)
                    out[istart:istop, jstart:jstop] = block / float(self.n)
        return out

    def _uniform_cost_matrix(self):
        """Exact cost for the product coupling 1/n^2 without T expansion."""

        import torch

        out = torch.empty((self.n, self.n), dtype=torch.float64, device=self.device)
        row_block = max(1, int(self.row_batch_size))
        col_block = max(1, int(self.column_batch_size))
        zero = torch.zeros((row_block, 1), dtype=torch.float64, device=self.device)
        columns = torch.arange(self.n, device=self.device)
        with torch.no_grad():
            for istart in range(0, self.n, row_block):
                istop = min(self.n, istart + row_block)
                a = self.tx[istart:istop]
                order = self.source_order[istart:istop]
                a_sorted = self.source_sorted[istart:istop]
                prefix_mass = torch.cumsum(torch.ones_like(a_sorted), dim=1)
                prefix_weighted = torch.cumsum(a_sorted, dim=1)
                prefix_mass = torch.cat([zero[: istop - istart], prefix_mass], dim=1)
                prefix_weighted = torch.cat([zero[: istop - istart], prefix_weighted], dim=1)
                total_mass = float(self.n)
                total_weighted = prefix_weighted[:, -1]
                for jstart in range(0, self.n, col_block):
                    jstop = min(self.n, jstart + col_block)
                    values = self.ty[jstart:jstop]
                    boundaries = a_sorted[:, None, :].expand(istop - istart, jstop - jstart, self.n)
                    queries = values[None, :, :].expand(istop - istart, jstop - jstart, self.n)
                    qidx = torch.searchsorted(boundaries.contiguous(), queries.contiguous(), right=True)
                    ml = torch.take_along_dim(prefix_mass[:, None, :].expand(istop - istart, jstop - jstart, self.n + 1), qidx, dim=2)
                    wl = torch.take_along_dim(prefix_weighted[:, None, :].expand(istop - istart, jstop - jstart, self.n + 1), qidx, dim=2)
                    block = (values[None, :, :] * (2.0 * ml - total_mass) + total_weighted[:, None, None] - 2.0 * wl).sum(dim=2)
                    out[istart:istop, jstart:jstop] = block / float(self.n * self.n)
        return out

    def _dense_cost_matrix(self, transport):
        """Exact dense-plan path with no [row_batch,n,n] expansion."""

        import torch

        out = torch.empty((self.n, self.n), dtype=torch.float64, device=self.device)
        col_block = max(1, int(self.column_batch_size))
        columns = torch.arange(self.n, device=self.device)
        with torch.no_grad():
            for i in range(self.n):
                a_sorted = self.source_sorted[i]
                order = self.source_order[i]
                weights = transport[order, :]
                prefix_mass = torch.cumsum(weights, dim=0)
                prefix_weighted = torch.cumsum(a_sorted[:, None] * weights, dim=0)
                prefix_mass = torch.cat([torch.zeros((1, self.n), dtype=torch.float64, device=self.device), prefix_mass], dim=0)
                prefix_weighted = torch.cat([torch.zeros((1, self.n), dtype=torch.float64, device=self.device), prefix_weighted], dim=0)
                total_mass = prefix_mass[-1]
                total_weighted = prefix_weighted[-1]
                for jstart in range(0, self.n, col_block):
                    jstop = min(self.n, jstart + col_block)
                    values = self.ty[jstart:jstop]
                    boundaries = a_sorted[None, :].expand(jstop - jstart, self.n)
                    qidx = torch.searchsorted(boundaries.contiguous(), values.contiguous(), right=True)
                    # Each target row's l-th query uses transport column l.
                    col = columns[None, :].expand(jstop - jstart, self.n)
                    ml = prefix_mass[qidx, col]
                    wl = prefix_weighted[qidx, col]
                    out[i, jstart:jstop] = (
                        values * (2.0 * ml - total_mass[None, :])
                        + total_weighted[None, :]
                        - 2.0 * wl
                    ).sum(dim=1)
                del weights, prefix_mass, prefix_weighted
        return out

    def objective(self, transport) -> float:
        import torch

        # Keep the original NumPy object as the cache key (see
        # ``cost_matrix``); conversion to a device tensor happens there.
        tt = self.transport_tensor(transport)
        cost = self.cost_matrix(transport)
        return float(torch.sum(cost * tt).detach().cpu())

    def gradient(self, transport, *, symmetric: bool = True) -> np.ndarray:
        cost = self.cost_matrix(transport)
        if symmetric:
            return (2.0 * cost).detach().cpu().numpy()
        raise NotImplementedError("Cached nonsymmetric torch gradient is not implemented")


def l1_gw_cost_matrix_torch(
    dx: np.ndarray,
    dy: np.ndarray,
    transport: np.ndarray,
    *,
    row_batch_size: int = 32,
) -> np.ndarray:
    """Exact L1 GW cost matrix using torch searchsorted/prefix sums.

    The returned matrix is the same quantity as
    ``src.ot.l1_gw._l1_gw_cost_matrix``. CUDA is used when available, otherwise
    PyTorch CPU is used. This is still an exact ``abs(a-b)`` computation.
    """

    import torch

    workspace = L1TorchWorkspace(dx, dy, row_batch_size=row_batch_size)
    return workspace.cost_matrix(transport).detach().cpu().numpy()


def l1_gw_objective_torch(dx: np.ndarray, dy: np.ndarray, transport: np.ndarray) -> float:
    cost = l1_gw_cost_matrix_torch(dx, dy, transport)
    return float(np.sum(cost * np.asarray(transport, dtype=np.float64)))


def l1_gw_objective_direct_torch(
    dx: np.ndarray,
    dy: np.ndarray,
    transport: np.ndarray,
    *,
    row_chunk_size: int = 2,
    column_chunk_size: int = 32,
) -> float:
    """Independent direct ``abs(a-b)`` audit with bounded 4-D chunks.

    This intentionally does not use sorting or prefix sums. It evaluates
    ``sum_{i,k,j,l} abs(dx[i,k]-dy[j,l]) * pi[i,j] * pi[k,l]`` directly in
    bounded chunks, which makes it suitable as a one-shot audit at n=1000.
    """

    import torch

    cx = validate_distance_matrix(dx, "dx")
    cy = validate_distance_matrix(dy, "dy")
    t_np = np.asarray(transport, dtype=np.float64)
    if t_np.shape != (cx.shape[0], cy.shape[0]):
        raise ValueError(f"Transport shape {t_np.shape} does not match distance matrices")
    if row_chunk_size <= 0 or column_chunk_size <= 0:
        raise ValueError("Chunk sizes must be positive")
    tx = _as_cuda_tensor(cx)
    ty = _as_cuda_tensor(cy)
    tt = _as_cuda_tensor(t_np)
    total = torch.zeros((), dtype=torch.float64, device=tx.device)
    for i_start in range(0, tx.shape[0], row_chunk_size):
        i_stop = min(i_start + row_chunk_size, tx.shape[0])
        source_chunk = tx[i_start:i_stop]
        for j_start in range(0, ty.shape[0], column_chunk_size):
            j_stop = min(j_start + column_chunk_size, ty.shape[0])
            target_chunk = ty[j_start:j_stop]
            # [i_chunk, j_chunk, k, l], with pi[k,l] as the inner weight.
            pairwise_abs = torch.abs(source_chunk[:, None, :, None] - target_chunk[None, :, None, :])
            cost_chunk = (pairwise_abs * tt[None, None, :, :]).sum(dim=(2, 3))
            total = total + (cost_chunk * tt[i_start:i_stop, j_start:j_stop]).sum()
            del pairwise_abs, cost_chunk
    return float(total.detach().cpu())


def l1_gw_gradient_torch(dx: np.ndarray, dy: np.ndarray, transport: np.ndarray) -> np.ndarray:
    cx = validate_distance_matrix(dx, "dx")
    cy = validate_distance_matrix(dy, "dy")
    forward = l1_gw_cost_matrix_torch(cx, cy, transport)
    if np.allclose(cx, cx.T, rtol=0.0, atol=1e-10) and np.allclose(cy, cy.T, rtol=0.0, atol=1e-10):
        return 2.0 * forward
    reverse = l1_gw_cost_matrix_torch(cx.T, cy.T, transport)
    return forward + reverse
