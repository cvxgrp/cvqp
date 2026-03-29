"""Projection onto sum-of-k-largest and CVaR constraints."""

import numpy as np
from ._cpp import proj_sum_largest as _proj_sum_largest_cpp


def proj_sum_largest(x: np.ndarray, k: int, alpha: float) -> np.ndarray:
    """Project onto {z : sum of k largest elements of z <= alpha}.

    Args:
        x: Input vector.
        k: Number of largest elements.
        alpha: Upper bound on the sum.

    Returns:
        Projected vector.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be a numpy array, got {type(x)}")
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {x.shape}")
    if not 0 < k <= len(x):
        raise ValueError(f"k must be in (0, len(x)], got {k}")

    sorted_idx = np.argsort(-x)
    x_sorted = x[sorted_idx].copy()

    # Early return if constraint already satisfied
    if np.sum(x_sorted[:k]) <= alpha:
        return x.copy()

    # C++ projection operates on sorted array in-place
    x_sorted, *_ = _proj_sum_largest_cpp(x_sorted, k, alpha, k, 0)

    result = np.empty_like(x)
    result[sorted_idx] = x_sorted
    return result


def proj_cvar(x: np.ndarray, beta: float, kappa: float) -> np.ndarray:
    """Project onto {z : CVaR_beta(z) <= kappa}.

    Args:
        x: Input vector (m scenarios).
        beta: Quantile level in [0, 1).
        kappa: CVaR upper bound.

    Returns:
        Projected vector.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be a numpy array, got {type(x)}")
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {x.shape}")
    if not 0 <= beta < 1:
        raise ValueError(f"beta must be in [0, 1), got {beta}")

    k = int((1 - beta) * len(x))
    alpha = kappa * k
    return proj_sum_largest(x, k, alpha)
