"""Tests for projection functions."""

import pytest
import numpy as np
import cvxpy as cp

from cvqp import proj_sum_largest, proj_cvar


class TestProjSumLargest:
    """Test projection onto sum-of-k-largest constraint."""

    def test_basic(self):
        x = np.array([6.0, 2.0, 5.0, 4.0, 1.0])
        k, alpha = 2, 7.0

        result = proj_sum_largest(x, k, alpha)

        # Constraint satisfied
        assert sum(sorted(result, reverse=True)[:k]) <= alpha + 1e-6

        # Matches CVXPY
        y = cp.Variable(x.shape)
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(y - x)),
            [cp.sum_largest(y, k) <= alpha],
        )
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if prob.status == cp.OPTIMAL:
            np.testing.assert_allclose(result, y.value, rtol=1e-5)

    def test_already_feasible(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = proj_sum_largest(x, 2, 10.0)  # 4+3=7 < 10
        np.testing.assert_array_almost_equal(result, x)

    def test_k_equals_1(self):
        x = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = proj_sum_largest(x, 1, 3.0)
        assert max(result) <= 3.0 + 1e-6

    def test_k_equals_n(self):
        x = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = proj_sum_largest(x, len(x), 10.0)
        assert np.sum(result) <= 10.0 + 1e-6

    def test_alpha_zero(self):
        x = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = proj_sum_largest(x, 2, 0.0)
        np.testing.assert_allclose(sorted(result)[-2:], [0.0, 0.0], atol=1e-6)

    def test_validation(self):
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="k must be"):
            proj_sum_largest(x, 0, 1.0)
        with pytest.raises(ValueError, match="1D"):
            proj_sum_largest(np.array([[1, 2], [3, 4]]), 1, 1.0)
        with pytest.raises(TypeError):
            proj_sum_largest([1.0, 2.0], 1, 1.0)


class TestProjCvar:
    """Test projection onto CVaR constraint."""

    def test_basic(self):
        x = np.array([6.0, 2.0, 5.0, 4.0, 1.0])
        beta, kappa = 0.6, 4.0

        result = proj_cvar(x, beta, kappa)

        k = int((1 - beta) * len(x))
        cvar = np.mean(np.sort(result)[-k:])
        assert cvar <= kappa + 1e-6

    def test_equivalence_with_sum_largest(self):
        x = np.array([6.0, 2.0, 5.0, 4.0, 1.0])
        beta, kappa = 0.6, 3.0

        result_cvar = proj_cvar(x, beta, kappa)

        k = int((1 - beta) * len(x))
        alpha = kappa * k
        result_sum = proj_sum_largest(x, k, alpha)

        np.testing.assert_array_almost_equal(result_cvar, result_sum)

    def test_validation(self):
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="beta"):
            proj_cvar(x, -0.1, 1.0)
        with pytest.raises(ValueError, match="beta"):
            proj_cvar(x, 1.0, 1.0)
