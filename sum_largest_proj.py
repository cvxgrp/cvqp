"""
Python wrapper for the C++ implementation of our sum-k-largest projection.
"""

import numpy as np
import mybindings


def proj_sum_largest(z: np.ndarray, k: int, alpha: float) -> np.ndarray:
    """
    Project a vector onto the set where the sum of its k largest elements is at most alpha.

    This function first sorts the input vector in descending order, applies the projection
    using C++ implementation, and then restores the original ordering of elements.

    Args:
        z: numpy array to project
        k: number of largest elements to consider in the sum constraint
        alpha: upper bound on the sum of k largest elements

    Returns:
        numpy array of same shape as z, containing the projected vector
    """
    # Sort in descending order and keep track of indices
    sorted_inds = np.argsort(z)[::-1]
    z_sorted = z[sorted_inds]

    # Apply projection (mybindings.sum_largest_proj returns multiple values, we only need first)
    z_projected, *_ = mybindings.sum_largest_proj(
        z_sorted, k, alpha, k, 0, len(z), False
    )

    # Restore original ordering
    x = np.empty_like(z)
    x[sorted_inds] = z_projected

    return x