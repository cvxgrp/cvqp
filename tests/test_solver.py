"""Tests for the CVQP solver."""

import pytest
import numpy as np
import scipy.sparse as sp
import cvxpy as cp

import cvqp

OBJECTIVE_TOL = 1e-2
FEASIBILITY_TOL = 5e-3


def _make_problem(matrix_type="dense", m=50, n=8):
    """Generate a small test problem."""
    rng = np.random.RandomState(42)

    if matrix_type == "linear":
        P = None
    else:
        P = np.diag(np.ones(n) + rng.rand(n) * 0.1)
        if matrix_type == "sparse_P":
            P = sp.csr_matrix(P)

    q = rng.randn(n) * 0.05
    A = rng.randn(m, n) * 0.1

    # B is sparse in most real problems (identity + few constraint rows)
    B = sp.eye(n, format="csr") if matrix_type != "dense" else np.eye(n)

    l = -np.ones(n) * 2
    u = np.ones(n) * 2
    beta = 0.95
    kappa = 0.5

    return dict(P=P, q=q, A=A, B=B, l=l, u=u, beta=beta, kappa=kappa)


def _solve_cvxpy(P, q, A, B, l, u, beta, kappa):
    """Solve with CVXPY for reference."""
    n = q.shape[0]
    x = cp.Variable(n)

    if P is None:
        objective = cp.Minimize(q @ x)
    else:
        P_dense = P.toarray() if sp.issparse(P) else P
        objective = cp.Minimize(0.5 * cp.quad_form(x, P_dense) + q @ x)

    constraints = [
        cp.cvar(A @ x, beta) <= kappa,
        l <= B @ x,
        B @ x <= u,
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status != cp.OPTIMAL:
        return None, None

    P_dense = P.toarray() if sp.issparse(P) else P
    if P is None:
        obj = float(q @ x.value)
    else:
        obj = float(0.5 * x.value @ P_dense @ x.value + q @ x.value)
    return x.value, obj


def _check_feasible(result, A, B, l, u, beta, kappa):
    """Verify all constraints are satisfied."""
    x = result.x
    Bx = B @ x if not sp.issparse(B) else np.asarray((B @ x)).ravel()

    assert np.all(Bx >= l - FEASIBILITY_TOL), "Lower bound violated"
    assert np.all(Bx <= u + FEASIBILITY_TOL), "Upper bound violated"

    Az = A @ x
    k = int((1 - beta) * len(Az))
    cvar_val = np.mean(np.sort(Az)[-k:])
    assert cvar_val <= kappa + FEASIBILITY_TOL, f"CVaR {cvar_val:.3f} > {kappa}"


class TestSolver:
    """Test cvqp.solve on small problems."""

    @pytest.mark.parametrize("matrix_type", ["dense", "sparse_B", "sparse_P", "linear"])
    def test_correctness(self, matrix_type):
        prob = _make_problem(matrix_type)
        result = cvqp.solve(**prob)

        assert result.status == "optimal"
        _check_feasible(result, prob["A"], prob["B"], prob["l"], prob["u"],
                        prob["beta"], prob["kappa"])

        # Compare to CVXPY
        _, cvxpy_obj = _solve_cvxpy(**prob)
        if cvxpy_obj is not None:
            gap = abs(cvxpy_obj - result.value) / max(abs(cvxpy_obj), 1e-8)
            assert gap <= OBJECTIVE_TOL, f"Objective gap {gap:.2e}"

    def test_warm_start(self):
        prob = _make_problem(m=30, n=5)
        rng = np.random.RandomState(123)

        result_cold = cvqp.solve(**prob)
        result_warm = cvqp.solve(**prob, warm_start=rng.randn(5) * 0.1)

        assert result_cold.status == "optimal"
        assert result_warm.status == "optimal"
        _check_feasible(result_cold, prob["A"], prob["B"], prob["l"], prob["u"],
                        prob["beta"], prob["kappa"])
        _check_feasible(result_warm, prob["A"], prob["B"], prob["l"], prob["u"],
                        prob["beta"], prob["kappa"])

    def test_inputs_not_mutated(self):
        prob = _make_problem()
        A_copy = prob["A"].copy()
        q_copy = prob["q"].copy()

        cvqp.solve(**prob)

        np.testing.assert_array_equal(prob["A"], A_copy)
        np.testing.assert_array_equal(prob["q"], q_copy)

    def test_result_fields(self):
        prob = _make_problem()
        result = cvqp.solve(**prob)

        assert isinstance(result, cvqp.Result)
        assert isinstance(result.x, np.ndarray)
        assert isinstance(result.value, float)
        assert result.status in ("optimal", "time_limit", "max_iter")
        assert result.iterations > 0
        assert result.time > 0
        assert isinstance(result.history, cvqp.History)
        assert len(result.history.r_norm) > 0

    def test_settings(self):
        prob = _make_problem()
        settings = cvqp.Settings(max_iter=10, check_interval=1)
        result = cvqp.solve(**prob, settings=settings)
        assert result.iterations <= 10

    def test_validation(self):
        with pytest.raises(ValueError, match="beta"):
            cvqp.solve(P=np.eye(2), q=np.zeros(2), A=np.ones((3, 2)),
                       B=np.eye(2), l=-np.ones(2), u=np.ones(2),
                       beta=0.0, kappa=1.0)
        with pytest.raises(ValueError, match="cols"):
            cvqp.solve(P=None, q=np.zeros(2), A=np.ones((3, 3)),
                       B=np.eye(2), l=-np.ones(2), u=np.ones(2),
                       beta=0.5, kappa=1.0)
