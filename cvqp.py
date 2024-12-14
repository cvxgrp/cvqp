"""
CVQP: A solver for Conditional Value-at-Risk (CVaR) constrained quadratic programs.

This module implements an ADMM-based solver for quadratic optimization problems with 
CVaR constraints. It uses adaptive penalty updates and over-relaxation for improved 
convergence properties.
"""

from dataclasses import dataclass
import logging
import time
import cvxpy as cp
import numpy as np
import scipy as sp

from sum_largest_proj import proj_sum_largest
from cvqp_problems import CVQPParams

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: %(message)s", datefmt="%b %d %H:%M:%S"
)


@dataclass
class CVQPConfig:
    """
    Configuration parameters for the CVQP solver.

    Args:
        max_iter: Maximum number of iterations before termination
        alpha: Step size parameter for gradient updates
        rho: Initial penalty parameter for augmented Lagrangian
        abstol: Absolute tolerance for primal and dual residuals
        reltol: Relative tolerance for primal and dual residuals
        alpha_over: Over-relaxation parameter for improved convergence (typically in [1.5, 1.8])
        print_freq: Frequency of iteration status updates
        mu: Threshold parameter for adaptive rho updates
        rho_incr: Multiplicative factor for increasing rho
        rho_decr: Multiplicative factor for decreasing rho
        verbose: If True, prints detailed convergence information
        time_limit: Maximum time in seconds before termination (default: 3600s = 1h)
        dynamic_rho: If True, adaptively updates the penalty parameter rho during optimization
    """

    max_iter: int = int(1e5)
    alpha: float = 0.5
    rho: float = 0.1
    abstol: float = 1e-3
    reltol: float = 1e-3
    alpha_over: float = 1.7
    print_freq: int = 100
    mu: float = 10
    rho_incr: float = 2.0
    rho_decr: float = 2.0
    verbose: bool = False
    time_limit: float = 3600
    dynamic_rho: bool = True


@dataclass
class CVQPResults:
    """
    Results from the CVQP solver.

    Attributes:
        x: Optimal solution vector
        iter_count: Number of iterations performed
        solve_time: Total solve time in seconds
        objval: List of objective values at each iteration
        r_norm: List of primal residual norms
        s_norm: List of dual residual norms
        eps_pri: List of primal feasibility tolerances
        eps_dual: List of dual feasibility tolerances
        rho: List of penalty parameter values
        problem_status: Final status of the solve ("optimal", "unknown", etc.)
    """

    x: np.ndarray
    iter_count: int
    solve_time: float
    objval: list[float]
    r_norm: list[float]
    s_norm: list[float]
    eps_pri: list[float]
    eps_dual: list[float]
    rho: list[float]
    problem_status: str = "unknown"


class CVQP:
    """
    CVQP solver using ADMM with adaptive penalty parameter updates and over-relaxation.

    This solver handles quadratic programs with CVaR constraints using the alternating
    direction method of multipliers (ADMM). It includes features for automatic penalty
    parameter adaptation and over-relaxation to improve convergence speed.
    """

    def __init__(self, params: CVQPParams, options: CVQPConfig = CVQPConfig()):
        """
        Initialize the CVQP solver.

        Args:
            params: Problem parameters including objective and constraint matrices
            options: Solver configuration options
        """
        self.params = params
        self.options = options
        self.initialize_problem()

    def initialize_problem(self):
        """
        Initialize problem by scaling data and precomputing frequently used matrices.
        Improves numerical stability and computation efficiency.
        """
        self.scale_problem()
        self.setup_cvar_params()
        self.precompute_matrices()

    def scale_problem(self):
        """Scale problem data to improve numerical conditioning."""
        self.scale = max(-self.params.A.min(), self.params.A.max())
        self.params.A /= self.scale
        self.params.q /= self.scale
        self.params.P /= self.scale

    def setup_cvar_params(self):
        """Initialize CVaR-specific parameters based on problem dimensions."""
        self.m = self.params.A.shape[0]
        self.k = int((1 - self.params.beta) * self.m)
        self.alpha = self.params.kappa * self.k / self.scale

    def precompute_matrices(self):
        """Precompute and cache frequently used matrix products."""
        self.AtA = self.params.A.T @ self.params.A
        self.BtB = self.params.B.T @ self.params.B
        self.update_M_factor(self.options.rho)

    def update_M_factor(self, rho: float):
        """
        Update and factorize the matrix M used in the linear system solve.

        Args:
            rho: Current penalty parameter value
        """
        self.M = self.params.P + rho * (self.AtA + self.BtB)
        self.factor = sp.linalg.lu_factor(self.M)

    def initialize_variables(self, warm_start: np.ndarray | None) -> tuple:
        """
        Initialize optimization variables and results structure.

        Args:
            warm_start: Initial guess for x, if provided

        Returns:
            Tuple of (z, u, z_tilde, u_tilde, results) containing initial values
        """
        if warm_start is None:
            x = np.zeros(self.params.P.shape[0])
            z = np.zeros(self.m)
            z_tilde = np.zeros(self.params.B.shape[0])
        else:
            x = warm_start.copy()
            z = self.params.A @ warm_start
            z_tilde = self.params.B @ warm_start

        u = np.zeros(self.m)
        u_tilde = np.zeros(self.params.B.shape[0])

        results = CVQPResults(
            x=x,
            iter_count=0,
            solve_time=0,
            objval=[],
            r_norm=[],
            s_norm=[],
            eps_pri=[],
            eps_dual=[],
            rho=[],
        )

        return z, u, z_tilde, u_tilde, results

    def x_update(
        self,
        z: np.ndarray,
        u: np.ndarray,
        z_tilde: np.ndarray,
        u_tilde: np.ndarray,
        rho: float,
    ) -> np.ndarray:
        """
        Perform x-minimization step of ADMM.

        Args:
            z: First auxiliary variable
            u: First dual variable
            z_tilde: Second auxiliary variable
            u_tilde: Second dual variable
            rho: Current penalty parameter

        Returns:
            Updated x variable
        """
        rhs = (
            -self.params.q
            + rho * (self.params.A.T @ (z - u))
            + rho * (self.params.B.T @ (z_tilde - u_tilde))
        )
        return sp.linalg.lu_solve(self.factor, rhs)

    def z_update(
        self, x: np.ndarray, z: np.ndarray, u: np.ndarray, alpha_over: float
    ) -> np.ndarray:
        """
        Perform z-minimization step of ADMM with over-relaxation.

        Args:
            x: Current primal variable
            z: Current z variable
            u: Current dual variable
            alpha_over: Over-relaxation parameter

        Returns:
            Updated z variable after projection
        """
        z_hat = alpha_over * (self.params.A @ x) + (1 - alpha_over) * z + u
        return proj_sum_largest(z_hat, self.k, self.alpha)

    def z_tilde_update(
        self, x: np.ndarray, z_tilde: np.ndarray, u_tilde: np.ndarray, alpha_over: float
    ) -> np.ndarray:
        """
        Perform z_tilde-minimization step of ADMM with over-relaxation.

        Args:
            x: Current primal variable
            z_tilde: Current z_tilde variable
            u_tilde: Current dual variable
            alpha_over: Over-relaxation parameter

        Returns:
            Updated z_tilde variable after projection
        """
        z_hat_tilde = (
            alpha_over * self.params.B @ x + (1 - alpha_over) * z_tilde + u_tilde
        )
        return np.clip(z_hat_tilde, self.params.l, self.params.u)

    def compute_residuals(
        self,
        x: np.ndarray,
        z: np.ndarray,
        z_tilde: np.ndarray,
        z_old: np.ndarray,
        z_tilde_old: np.ndarray,
        rho: float,
    ) -> tuple:
        """
        Compute primal and dual residuals for convergence checking.

        Args:
            x: Current primal variable
            z, z_tilde: Current auxiliary variables
            z_old, z_tilde_old: Previous auxiliary variables
            rho: Current penalty parameter

        Returns:
            Tuple of (r_norm, s_norm, Ax, At_z) containing residual norms and intermediate products
        """
        Ax = self.params.A @ x
        r = np.concatenate([Ax - z, self.params.B @ x - z_tilde])
        r_norm = np.linalg.norm(r)

        z_diff = z - z_old
        z_tilde_diff = z_tilde - z_tilde_old
        At_z = self.params.A.T @ z_diff + self.params.B.T @ z_tilde_diff
        s_norm = np.linalg.norm(rho * At_z)

        return r_norm, s_norm, Ax, At_z

    def compute_tolerances(
        self,
        Ax: np.ndarray,
        z: np.ndarray,
        z_tilde: np.ndarray,
        At_z: np.ndarray,
        rho: float,
    ) -> tuple:
        """
        Compute primal and dual feasibility tolerances.

        Args:
            Ax: Product of A and x
            z, z_tilde: Current auxiliary variables
            At_z: Transposed product
            rho: Current penalty parameter

        Returns:
            Tuple of (eps_pri, eps_dual) containing primal and dual tolerances
        """
        d0 = self.params.A.shape[0] + self.params.B.shape[0]
        d1 = self.params.A.shape[1]

        eps_pri = (d0**0.5) * self.options.abstol + self.options.reltol * max(
            np.linalg.norm(Ax), np.linalg.norm(np.concatenate([z, z_tilde]))
        )
        eps_dual = (
            d1**0.5
        ) * self.options.abstol + self.options.reltol * np.linalg.norm(rho * At_z)

        return eps_pri, eps_dual

    def check_convergence(
        self, r_norm: float, s_norm: float, eps_pri: float, eps_dual: float
    ) -> bool:
        """
        Check if convergence criteria are satisfied.

        Args:
            r_norm: Primal residual norm
            s_norm: Dual residual norm
            eps_pri: Primal feasibility tolerance
            eps_dual: Dual feasibility tolerance

        Returns:
            True if both primal and dual residuals are within tolerances
        """
        return r_norm <= eps_pri and s_norm <= eps_dual

    def update_rho(
        self,
        rho: float,
        r_norm: float,
        s_norm: float,
        u: np.ndarray,
        u_tilde: np.ndarray,
    ) -> tuple:
        """
        Update penalty parameter using adaptive scheme.

        Args:
            rho: Current penalty parameter
            r_norm: Primal residual norm
            s_norm: Dual residual norm
            u, u_tilde: Current dual variables

        Returns:
            Tuple of (rho, u, u_tilde) containing updated values
        """
        if r_norm > self.options.mu * s_norm:
            rho *= self.options.rho_incr
            u /= self.options.rho_incr
            u_tilde /= self.options.rho_incr
            self.update_M_factor(rho)
        elif s_norm > self.options.mu * r_norm:
            rho /= self.options.rho_decr
            u *= self.options.rho_decr
            u_tilde *= self.options.rho_decr
            self.update_M_factor(rho)
        return rho, u, u_tilde

    def setup_progress_display(self):
        """Initialize progress display formatting and headers."""
        self.header_titles = [
            "iter",
            "r_norm",
            "eps_pri",
            "s_norm",
            "eps_dual",
            "rho",
            "obj_val",
        ]
        self.header_format = "{:<6} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}"
        self.row_format = (
            "{:<6} {:<12.3e} {:<12.3e} {:<12.3e} {:<12.3e} {:<12.2e} {:<12.3e}"
        )
        self.separator = "=" * 83

        logging.info(self.separator)
        title = "CVQP solver"
        logging.info(title.center(len(self.separator)))
        logging.info(self.separator)
        logging.info(self.header_format.format(*self.header_titles))
        logging.info("-" * 83)

    def print_iteration(
        self,
        iteration: int,
        r_norm: float,
        eps_pri: float,
        s_norm: float,
        eps_dual: float,
        rho: float,
        objval: float,
    ):
        """
        Print iteration results in formatted output.

        Args:
            iteration: Current iteration number
            r_norm: Primal residual norm
            eps_pri: Primal feasibility tolerance
            s_norm: Dual residual norm
            eps_dual: Dual feasibility tolerance
            rho: Current penalty parameter
            objval: Current objective value
        """
        logging.info(
            self.row_format.format(
                iteration, r_norm, eps_pri, s_norm, eps_dual, rho, objval
            )
        )

    def print_final_results(self, results: CVQPResults):
        """
        Print final optimization results summary.

        Args:
            results: Optimization results containing final values and statistics
        """
        logging.info(self.separator)
        logging.info(f"Optimal value: {results.objval[-1]:.3e}.")
        logging.info(f"Solver took {results.solve_time:.2f} seconds.")
        logging.info(f"Problem status: {results.problem_status}.")

    def record_iteration(
        self,
        results: CVQPResults,
        x: np.ndarray,
        r_norm: float,
        s_norm: float,
        eps_pri: float,
        eps_dual: float,
        rho: float,
    ):
        """
        Record the results of the current iteration for convergence analysis.

        Args:
            results: Results object to store iteration data
            x: Current primal variable
            r_norm: Primal residual norm
            s_norm: Dual residual norm
            eps_pri: Primal feasibility tolerance
            eps_dual: Dual feasibility tolerance
            rho: Current penalty parameter
        """
        objval = (0.5 * np.dot(x, self.params.P @ x) + self.params.q @ x) * self.scale
        results.objval.append(objval)
        results.r_norm.append(r_norm)
        results.s_norm.append(s_norm)
        results.eps_pri.append(eps_pri)
        results.eps_dual.append(eps_dual)
        results.rho.append(rho)

    def unscale_problem(self):
        """Restore original problem scaling for final results."""
        self.params.A *= self.scale
        self.params.q *= self.scale
        self.params.P *= self.scale

    def solve(self, warm_start: np.ndarray | None = None) -> CVQPResults:
        """
        Solve the optimization problem using ADMM algorithm.

        Args:
            warm_start: Optional initial guess for x variable

        Returns:
            CVQPResults object containing optimal solution and convergence information
        """
        start_time = time.time()

        # Initialize variables and results
        z, u, z_tilde, u_tilde, results = self.initialize_variables(warm_start)
        rho = self.options.rho

        # Setup progress display if verbose
        if self.options.verbose:
            self.setup_progress_display()

        # Main iteration loop
        for i in range(self.options.max_iter):
            # Store previous values
            z_old, z_tilde_old = z.copy(), z_tilde.copy()

            # Update primal and dual variables
            x = self.x_update(z, u, z_tilde, u_tilde, rho)
            z = self.z_update(x, z, u, self.options.alpha_over)
            z_tilde = self.z_tilde_update(x, z_tilde, u_tilde, self.options.alpha_over)

            # Update dual variables
            u += (
                self.options.alpha_over * (self.params.A @ x)
                + (1 - self.options.alpha_over) * z_old
                - z
            )
            u_tilde += (
                self.options.alpha_over * (self.params.B @ x)
                + (1 - self.options.alpha_over) * z_tilde_old
                - z_tilde
            )

            # Check convergence periodically
            if i % self.options.print_freq == 0:
                r_norm, s_norm, Ax, At_z = self.compute_residuals(
                    x, z, z_tilde, z_old, z_tilde_old, rho
                )
                eps_pri, eps_dual = self.compute_tolerances(Ax, z, z_tilde, At_z, rho)

                # Compute objective value
                objval = (
                    0.5 * np.dot(x, self.params.P @ x) + self.params.q @ x
                ) * self.scale

                # Record iteration
                self.record_iteration(
                    results, x, r_norm, s_norm, eps_pri, eps_dual, rho
                )

                # Print progress if verbose
                if self.options.verbose:
                    self.print_iteration(
                        i, r_norm, eps_pri, s_norm, eps_dual, rho, objval
                    )

                # Check time limit
                if time.time() - start_time > self.options.time_limit:
                    results.problem_status = "timeout"
                    break

                # Check convergence
                if self.check_convergence(r_norm, s_norm, eps_pri, eps_dual):
                    results.problem_status = "optimal"
                    break

                # Update penalty parameter
                if self.options.dynamic_rho:
                    rho, u, u_tilde = self.update_rho(rho, r_norm, s_norm, u, u_tilde)

        # Finalize results
        self.unscale_problem()
        results.x = x
        results.iter_count = i + 1
        results.solve_time = time.time() - start_time

        if self.options.verbose:
            self.print_final_results(results)

        return results


def test():
    """
    Test the CVQP implementation against CVXPY solver.

    Creates a random test problem and compares the solution obtained by CVQP
    against the solution from CVXPY using the MOSEK solver. Prints comparison
    metrics including objective values and CVaR constraint satisfaction.
    """
    m, d = int(1e4), int(1e2)
    np.random.seed(0)

    # Generate test problem
    params = CVQPParams(
        P=np.eye(d),
        q=np.ones(d) * -0.1 + np.random.randn(d) * 0.05,
        A=np.random.randn(m, d) * 0.2 + 0.1,
        B=np.eye(d),
        l=-np.ones(d),
        u=np.ones(d),
        beta=0.9,
        kappa=0.1,
    )

    # Solve with CVXPY for reference
    k = int((1 - params.beta) * m)
    alpha = params.kappa * k
    x_cvxpy = cp.Variable(d)
    objective = cp.Minimize(0.5 * cp.quad_form(x_cvxpy, params.P) + params.q @ x_cvxpy)
    constraints = [
        cp.sum_largest(params.A @ x_cvxpy, k) <= alpha,
        -1 <= x_cvxpy,
        x_cvxpy <= 1,
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=True)

    # Solve with CVQP solver
    solver = CVQP(params, CVQPConfig(verbose=True))
    results = solver.solve()

    # Compare results
    print("\nResults comparison:")
    print(f"CVXPY objective: {prob.value:.6f}")
    print(f"CVQP objective: {results.objval[-1]:.6f}")
    print(
        f"Relative objective error: "
        f"{abs(results.objval[-1] - prob.value) / abs(prob.value):.6f}"
    )

    print(f"\nCVaR constraint:")
    print(
        f"CVXPY CVaR: {cp.cvar(params.A @ x_cvxpy, params.beta).value:.6f} (limit: {params.kappa:.6f})"
    )
    print(
        f"CVQP CVaR: {cp.cvar(params.A @ results.x, params.beta).value:.6f} (limit: {params.kappa:.6f})"
    )


if __name__ == "__main__":
    test()
