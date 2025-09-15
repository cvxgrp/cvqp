"""
Projection onto sum-of-k-largest and CVaR constraints.
"""

import numpy as np
from .libs import proj_sum_largest as proj_sum_largest_cpp


def proj_sum_largest(x: np.ndarray, k: int, alpha: float) -> np.ndarray:
    """Project onto sum-of-k-largest constraint: sum of k largest elements <= alpha."""
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Input x must be a numpy array, got {type(x)}")
    if x.ndim != 1:
        raise ValueError(f"Input x must be a 1D array, got shape {x.shape}")
    if not 0 < k <= len(x):
        raise ValueError(f"k must be between 0 and len(x), got {k}")
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")

    sorted_inds = np.argsort(x)[::-1]
    x_sorted = x[sorted_inds]

    # Early return if constraint already satisfied
    if np.sum(x_sorted[:k]) <= alpha:
        return x.copy()

    # Use C++ implementation
    x_projected, *_ = proj_sum_largest_cpp(x_sorted, k, alpha, k, 0, len(x), False)
    result = np.empty_like(x)
    result[sorted_inds] = x_projected
    return result


def proj_cvar(x: np.ndarray, beta: float, kappa: float) -> np.ndarray:
    """Project onto CVaR constraint: CVaR_beta(x) <= kappa."""
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Input x must be a numpy array, got {type(x)}")
    if x.ndim != 1:
        raise ValueError(f"Input x must be a 1D array, got shape {x.shape}")
    if not 0 < beta < 1:
        raise ValueError(f"beta must be between 0 and 1 (exclusive), got {beta}")
    if kappa < 0:
        raise ValueError(f"kappa must be non-negative, got {kappa}")

    n_scenarios = x.shape[0]
    k = int((1 - beta) * n_scenarios)  # Number of tail scenarios
    if k == 0:
        return x.copy()  # Constraint is vacuous

    # Convert CVaR constraint to sum-of-k-largest
    alpha = kappa * k
    return proj_sum_largest(x, k, alpha)
