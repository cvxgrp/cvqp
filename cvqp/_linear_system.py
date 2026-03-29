"""Factorization and solve for the ADMM linear system."""

import numpy as np
import scipy as sp


class LinearSystem:
    """Cache a Cholesky factorization of M = P + rho*(AtA + BtB).

    M is always dense because AtA = A^T A is dense (A is the scenario
    matrix, which has no sparsity). Uses LAPACK Cholesky (dpotrf/dpotrs)
    with LU fallback for robustness.
    """

    def __init__(
        self,
        P: np.ndarray | sp.sparse.spmatrix | None,
        AtA: np.ndarray,
        BtB: np.ndarray | sp.sparse.spmatrix,
        rho: float,
    ):
        self._P = P
        self._AtA = AtA
        self._BtB = BtB
        self._solve = None
        self.factorize(rho)

    def factorize(self, rho: float) -> None:
        """Build M = P + rho*(AtA + BtB) and compute Cholesky factorization."""
        penalty = rho * (self._AtA + self._BtB)
        M = penalty if self._P is None else self._P + penalty
        M = np.asarray(M)

        try:
            factor = sp.linalg.cho_factor(M)
            self._solve = lambda rhs: sp.linalg.cho_solve(factor, rhs)
        except np.linalg.LinAlgError:
            factor = sp.linalg.lu_factor(M)
            self._solve = lambda rhs: sp.linalg.lu_solve(factor, rhs)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve Mx = rhs using cached factorization."""
        return self._solve(rhs)
