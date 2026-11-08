"""ADMM solver for CVaR-constrained quadratic programs."""

import time
import numpy as np
import scipy as sp

from ._types import Settings, History, Result
from ._linear_system import LinearSystem
from ._projection import proj_sum_largest


def solve(
    P: np.ndarray | sp.sparse.spmatrix | None,
    q: np.ndarray,
    A: np.ndarray,
    B: np.ndarray | sp.sparse.spmatrix,
    l: np.ndarray,
    u: np.ndarray,
    beta: float,
    kappa: float,
    *,
    warm_start: np.ndarray | None = None,
    verbose: bool = False,
    settings: Settings | None = None,
) -> Result:
    """Solve a CVaR-constrained quadratic program.

    Solves:
        minimize    (1/2) x^T P x + q^T x
        subject to  CVaR_beta(Ax) <= kappa
                    l <= Bx <= u

    Args:
        P: Quadratic cost matrix (n x n), or None for linear problems.
        q: Linear cost vector (n,).
        A: Scenario matrix (m x n).
        B: Linear constraint matrix (p x n).
        l: Lower bounds (p,).
        u: Upper bounds (p,).
        beta: CVaR quantile level in (0, 1).
        kappa: CVaR upper bound.
        warm_start: Initial x guess.
        verbose: Print iteration progress.
        settings: Solver settings.

    Returns:
        Result with optimal x, objective value, and convergence history.
    """
    _validate(P, q, A, B, l, u, beta)
    cfg = settings or Settings()

    # Scale problem (local copies, never mutate inputs)
    P_s, q_s, A_s, scale = _scale(P, q, A)
    m, n = A_s.shape
    k = int((1 - beta) * m)
    alpha = kappa * k / scale

    # Precompute and factorize
    AtA = A_s.T @ A_s
    BtB = B.T @ B
    system = LinearSystem(P_s, AtA, BtB, cfg.rho)

    # Initialize ADMM variables
    p = B.shape[0]
    if warm_start is not None:
        x = warm_start.copy()
        z = A_s @ x
        z_t = _densify(B @ x)
    else:
        x = np.zeros(n)
        z = np.zeros(m)
        z_t = np.zeros(p)
    u_z = np.zeros(m)
    u_t = np.zeros(p)
    rho = cfg.rho

    history = History()
    t0 = time.time()
    status = "max_iter"

    if verbose:
        _print_header()

    for i in range(cfg.max_iter):
        z_prev = z.copy()
        z_t_prev = z_t.copy()

        # x-update: solve linear system
        rhs = -q_s + rho * (A_s.T @ (z - u_z) + B.T @ (z_t - u_t))
        x = system.solve(rhs)

        # Cache matrix-vector products (computed once per iteration)
        Ax = A_s @ x
        Bx = _densify(B @ x)

        # z-update: over-relax then project onto CVaR set
        z_hat = cfg.alpha_relax * Ax + (1 - cfg.alpha_relax) * z_prev
        z = proj_sum_largest(z_hat + u_z, k, alpha)

        # z_tilde-update: over-relax then clip to [l, u]
        z_t_hat = cfg.alpha_relax * Bx + (1 - cfg.alpha_relax) * z_t_prev
        z_t = np.clip(z_t_hat + u_t, l, u)

        # Dual updates (reuse z_hat, z_t_hat)
        u_z += z_hat - z
        u_t += z_t_hat - z_t

        # Convergence check (every check_interval iterations)
        if i % cfg.check_interval == 0:
            r_norm, s_norm, eps_pri, eps_dual = _compute_residuals(
                Ax, Bx, z, z_t, z_prev, z_t_prev, u_z, u_t, A_s, B, rho, m, p, cfg,
            )
            obj = _objective(P_s, q_s, x, scale)

            history.r_norm.append(r_norm)
            history.s_norm.append(s_norm)
            history.eps_pri.append(eps_pri)
            history.eps_dual.append(eps_dual)
            history.objective.append(obj)
            history.rho.append(rho)

            if verbose:
                _print_iter(i, r_norm, eps_pri, s_norm, eps_dual, rho, obj)

            if r_norm <= eps_pri and s_norm <= eps_dual:
                status = "optimal"
                break

            if time.time() - t0 > cfg.time_limit:
                status = "time_limit"
                break

            # Adaptive rho (reuses residuals just computed)
            if cfg.adaptive_rho:
                rho_new = _adapt_rho(rho, r_norm, s_norm, cfg)
                if rho_new != rho:
                    u_z *= rho / rho_new
                    u_t *= rho / rho_new
                    rho = rho_new
                    system.factorize(rho)

    solve_time = time.time() - t0
    final_obj = _objective(P_s, q_s, x, scale)

    if verbose:
        _print_footer(final_obj, solve_time, status)

    return Result(
        x=x,
        value=final_obj,
        status=status,
        iterations=i + 1,
        time=solve_time,
        history=history,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate(P, q, A, B, l, u, beta):
    """Check dimensional compatibility and parameter ranges."""
    n = q.shape[0]
    if P is not None:
        if P.shape != (n, n):
            raise ValueError(f"P must be ({n}, {n}), got {P.shape}")
    if A.shape[1] != n:
        raise ValueError(f"A has {A.shape[1]} cols, expected {n}")
    if B.shape[1] != n:
        raise ValueError(f"B has {B.shape[1]} cols, expected {n}")
    if l.shape[0] != B.shape[0]:
        raise ValueError(f"l has length {l.shape[0]}, B has {B.shape[0]} rows")
    if u.shape[0] != B.shape[0]:
        raise ValueError(f"u has length {u.shape[0]}, B has {B.shape[0]} rows")
    if not 0 < beta < 1:
        raise ValueError(f"beta must be in (0, 1), got {beta}")


def _scale(P, q, A):
    """Return scaled copies of P, q, A. Never mutates inputs."""
    scale = max(np.max(np.abs(A)), 1.0)
    A_s = A / scale
    q_s = q / scale
    P_s = P / scale if P is not None else None
    return P_s, q_s, A_s, scale


def _densify(x):
    """Convert sparse matrix/array to dense 1D array if needed."""
    if sp.sparse.issparse(x):
        return np.asarray(x.todense()).ravel()
    return np.asarray(x).ravel()


def _objective(P, q, x, scale):
    """Compute objective value in original (unscaled) units."""
    if P is None:
        return float(q @ x) * scale
    if sp.sparse.issparse(P):
        return float(0.5 * x @ P @ x + q @ x) * scale
    return float(0.5 * np.dot(x, P @ x) + q @ x) * scale


def _compute_residuals(Ax, Bx, z, z_t, z_prev, z_t_prev, u_z, u_t, A, B, rho, m, p, cfg):
    """Compute primal/dual residuals and tolerances.

    Follows Boyd et al. (2011), section 3.3.1. The relative term of eps_dual
    uses the unscaled dual variable y = rho * u.
    """
    # Primal residual
    r = np.concatenate([Ax - z, Bx - z_t])
    r_norm = np.linalg.norm(r)

    # Dual residual
    s = rho * (A.T @ (z - z_prev) + _densify(B.T @ (z_t - z_t_prev)))
    s_norm = np.linalg.norm(s)

    # Tolerances
    sqrt_mp = (m + p) ** 0.5
    sqrt_n = A.shape[1] ** 0.5
    eps_pri = sqrt_mp * cfg.abstol + cfg.reltol * max(
        np.linalg.norm(np.concatenate([Ax, Bx])),
        np.linalg.norm(np.concatenate([z, z_t])),
    )
    eps_dual = sqrt_n * cfg.abstol + cfg.reltol * np.linalg.norm(
        rho * _densify(A.T @ u_z + B.T @ u_t)
    )

    return r_norm, s_norm, eps_pri, eps_dual


def _adapt_rho(rho, r_norm, s_norm, cfg):
    """Compute new rho based on primal/dual residual balance."""
    if r_norm > cfg.mu * s_norm:
        return rho * cfg.tau
    elif s_norm > cfg.mu * r_norm:
        return rho / cfg.tau
    return rho


_COL_FMT = "{:<6} {:<12.3e} {:<12.3e} {:<12.3e} {:<12.3e} {:<12.2e} {:<12.3e}"
_HDR_FMT = "{:<6} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}"
_SEP = "=" * 83


def _print_header():
    print(_SEP)
    print("CVQP solver".center(83))
    print(_SEP)
    print(_HDR_FMT.format("iter", "r_norm", "eps_pri", "s_norm", "eps_dual", "rho", "obj"))
    print("-" * 83)


def _print_iter(i, r_norm, eps_pri, s_norm, eps_dual, rho, obj):
    print(_COL_FMT.format(i, r_norm, eps_pri, s_norm, eps_dual, rho, obj))


def _print_footer(obj, solve_time, status):
    print(_SEP)
    print(f"Optimal value: {obj:.3e}")
    print(f"Solve time: {solve_time:.2f}s")
    print(f"Status: {status}")
