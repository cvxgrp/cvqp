"""Problem generators for CVQP experiments."""

import numpy as np
import cvxpy as cp
import scipy.sparse as sp


def make_portfolio(
    n_vars: int,
    n_scenarios: int,
    seed: int | None = None,
    alpha: float = 0.8,
    gamma: float = 1.0,
    nu: float = 0.2,
    sigma: float = 2.0,
    beta: float = 0.95,
    kappa: float = 0.3,
) -> tuple[dict, cp.Problem]:
    """Generate a portfolio optimization instance with CVaR constraint.

    Returns:
        (data, cvxpy_problem) where data is a dict with keys
        matching cvqp.solve() arguments.
    """
    rng = np.random.default_rng(seed)

    n_normal = int(alpha * n_scenarios)
    n_stress = n_scenarios - n_normal

    R = np.empty((n_scenarios, n_vars))
    R[:n_normal] = rng.normal(nu, 1.0, size=(n_normal, n_vars))
    R[n_normal:] = rng.normal(-nu, sigma, size=(n_stress, n_vars))

    mu = np.mean(R, axis=0)

    if n_scenarios >= 1e6:
        Sigma = np.diag(np.var(R, axis=0))
        P = gamma * sp.diags(np.diag(Sigma))
    else:
        Sigma = np.cov(R.T)
        P = gamma * Sigma

    A = -R
    B = sp.vstack([sp.csr_matrix(np.ones(n_vars)), sp.eye(n_vars, format="csr")])
    l = np.concatenate([np.ones(1), np.zeros(n_vars)])
    u = np.concatenate([np.ones(1), np.full(n_vars, np.inf)])

    data = dict(P=P, q=-mu, A=A, B=B, l=l, u=u, beta=beta, kappa=kappa)

    x = cp.Variable(n_vars)
    k = int((1 - beta) * n_scenarios)
    objective = cp.Minimize((gamma / 2) * cp.quad_form(x, Sigma, assume_PSD=True) - mu @ x)
    constraints = [cp.sum(x) == 1, x >= 0, cp.sum_largest(A @ x, k) <= kappa * k]

    return data, cp.Problem(objective, constraints)


def make_quantile_regression(
    n_vars: int,
    n_scenarios: int,
    seed: int | None = None,
    tau: float = 0.9,
) -> tuple[dict, cp.Problem]:
    """Generate a quantile regression instance.

    Returns:
        (data, cvxpy_problem) where data is a dict with keys
        matching cvqp.solve() arguments.
    """
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n_scenarios, n_vars))
    beta_true = rng.standard_normal(n_vars) / np.sqrt(1 + np.arange(n_vars))
    y = X @ beta_true + 0.1 * rng.standard_t(df=5, size=n_scenarios)

    A = np.hstack([-X, -np.ones((n_scenarios, 1)), y.reshape(-1, 1)])
    q = np.hstack([np.mean(X, axis=0), 1, 0])
    B = sp.csr_matrix(([1], ([0], [n_vars + 1])), shape=(1, n_vars + 2))

    data = dict(P=None, q=q, A=A, B=B, l=np.ones(1), u=np.ones(1), beta=tau, kappa=0.0)

    x = cp.Variable(n_vars + 2)
    k = int((1 - tau) * n_scenarios)
    objective = cp.Minimize(q @ x)
    constraints = [cp.sum_largest(A @ x, k) <= 0.0, x[-1] == 1]

    return data, cp.Problem(objective, constraints)
