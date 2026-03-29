"""Smoke test for the C++ extension (no cvxpy dependency)."""

import numpy as np
from cvqp import proj_sum_largest, proj_cvar


def test_import():
    """C++ extension loads correctly."""
    from cvqp._cpp import proj_sum_largest as cpp_func
    assert callable(cpp_func)


def test_proj_sum_largest():
    x = np.array([6.0, 2.0, 5.0, 4.0, 1.0])
    result = proj_sum_largest(x, 2, 7.0)
    assert sum(sorted(result, reverse=True)[:2]) <= 7.0 + 1e-6


def test_proj_sum_largest_feasible():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = proj_sum_largest(x, 2, 10.0)
    np.testing.assert_array_almost_equal(result, x)


def test_proj_cvar():
    x = np.array([6.0, 2.0, 5.0, 4.0, 1.0])
    result = proj_cvar(x, 0.6, 4.0)
    k = int(0.4 * len(x))
    assert np.mean(np.sort(result)[-k:]) <= 4.0 + 1e-6
