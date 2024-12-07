"""
Script to benchmark CVQP solver againts MOSEK and Clarabel on a series of 
porblems from different domains: portfolio optimization, network traffic, and
supply chain contract selection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import pickle
import time
import cvxpy as cp
import numpy as np

from cvqp import CVQP, CVQPConfig, CVQPResults
from cvqp_utils import CVQPParams

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: %(message)s", datefmt="%b %d %H:%M:%S"
)


@dataclass
class BenchmarkResults:
    """
    Store benchmark results for a specific problem configuration.

    Args:
        problem: Name of the problem being solved.
        solver: Name of the solver used.
        n_vars: Number of variables in the problem.
        n_scenarios: Number of scenarios in the problem.
        times: List of solve times for each instance. None indicates a failed solve.
        status: Status returned by solver for each solve attempt ('optimal', 'infeasible', etc.).
    """

    problem: str
    solver: str
    n_vars: int
    n_scenarios: int
    times: list[float | None]
    status: list[str]
    cvqp_results: list[CVQPResults | None] = None

    @property
    def success_rate(self) -> float:
        """Return fraction of successful solves."""
        return sum(1 for t in self.times if t is not None) / len(self.times)

    @property
    def avg_time(self) -> float | None:
        """
        Average time of successful solves.

        Returns:
            Mean solve time of successful solves, or None if all solves failed.
        """
        valid_times = [t for t in self.times if t is not None]
        return np.mean(valid_times) if valid_times else None

    @property
    def std_time(self) -> float | None:
        """
        Standard deviation of successful solve times.

        Returns:
            Standard deviation of successful solve times, or None if all solves failed.
        """
        valid_times = [t for t in self.times if t is not None]
        return np.std(valid_times) if valid_times else None


class CVQProblem(ABC):
    """Abstract base class for CVQP problems."""

    @abstractmethod
    def generate_instance(
        self, n_vars: int, n_scenarios: int, seed: int | None = None
    ) -> CVQPParams:
        """
        Generate a problem instance with given dimensions.

        Args:
            n_vars: Number of decision variables.
            n_scenarios: Number of scenarios.
            seed: Random seed for reproducibility.

        Returns:
            CVQPParams instance containing the generated problem parameters.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Problem name for results identification."""
        pass


class PortfolioOptimization(CVQProblem):
    """
    Portfolio optimization problem.

    This class implements a portfolio optimization problem where returns are modeled
    using a two-component Gaussian mixture to capture both normal and stress market
    conditions. The objective balances expected return against variance risk, with
    additional CVaR constraints on worst-case losses.

    Args:
        alpha: Probability of normal market conditions.
        gamma: Risk aversion parameter for variance term.
        nu: Mean return in normal market conditions.
        sigma: Volatility scaling factor for stress periods.
        beta: List containing two CVaR probability levels.
        kappa: List containing two CVaR thresholds.
    """

    def __init__(
        self,
        alpha: float = 0.9,
        gamma: float = 0.05,
        nu: float = 0.1,
        sigma: float = 0.2,
        beta: float = 0.95,
        kappa: float = 0.15,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.nu = nu
        self.sigma = sigma
        self.beta = beta
        self.kappa = kappa

    def generate_return_matrix(
        self, n_vars: int, n_scenarios: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Generate returns using a two-component Gaussian mixture model.

        Args:
            n_vars: Number of assets.
            n_scenarios: Number of return scenarios.
            rng: Random number generator.

        Returns:
            Matrix of asset returns across scenarios.
        """
        # Normal market conditions
        normal_prob = rng.binomial(1, self.alpha, n_scenarios)
        R = np.zeros((n_scenarios, n_vars))

        # Generate returns for normal conditions
        normal_idx = normal_prob == 1
        R[normal_idx] = rng.normal(self.nu, 1.0, size=(normal_prob.sum(), n_vars))

        # Generate returns for stress conditions
        stress_idx = normal_prob == 0
        R[stress_idx] = rng.normal(
            -self.nu, self.sigma, size=(stress_idx.sum(), n_vars)
        )

        return R

    def generate_instance(
        self, n_vars: int, n_scenarios: int, seed: int | None = None
    ) -> CVQPParams:
        """
        Generate a problem instance.

        Args:
            n_vars: Number of assets.
            n_scenarios: Number of return scenarios.
            seed: Random seed for reproducibility.

        Returns:
            CVQPParams instance containing the generated problem parameters.
        """
        # Generate return matrix
        rng = np.random.default_rng(seed)
        R = self.generate_return_matrix(n_vars, n_scenarios, rng)

        # Compute mean returns
        mu = R.mean(axis=0)

        # Compute covariance matrix
        R_centered = R - mu
        Sigma = (R_centered.T @ R_centered) / n_scenarios

        # CVQP parameters
        P = self.gamma * Sigma
        q = -mu
        A = -R

        # Linear constraints: sum(x) = 1, x >= 0
        B = np.vstack([np.ones(n_vars), np.eye(n_vars)])
        l = np.array([1.0] + [0.0] * n_vars)
        u = np.array([1.0] + [float("inf")] * n_vars)

        return CVQPParams(
            P=P, q=q, A=A, B=B, l=l, u=u, beta=self.beta, kappa=self.kappa
        )

    @property
    def name(self) -> str:
        """Return the name identifier for this problem."""
        return "portfolio"


class NetworkTraffic(CVQProblem):
    """
    Network traffic engineering problem.

    This class implements a traffic engineering problem where path delays are modeled
    with both path-specific and scenario-wide variations. The objective balances
    nominal routing cost against traffic concentration, with a CVaR constraint on
    worst-case delays.

    Args:
        gamma: Regularization parameter for traffic concentration.
        beta: CVaR probability level.
        K: Number of source-destination pairs.
        sigma1: Standard deviation for path-specific variations.
        sigma2: Standard deviation for scenario-wide variations.
    """

    def __init__(
        self,
        gamma: float = 0.1,
        beta: float = 0.95,
        K: int = 10,
        sigma1: float = 0.05,
        sigma2: float = 0.1,
    ):
        self.gamma = gamma
        self.beta = beta
        self.K = K
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def generate_path_structure(
        self, n_vars: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """
        Generate paths and their mapping to source-destination pairs.

        Args:
            n_vars: Total number of paths.
            rng: Random number generator.

        Returns:
            Tuple containing:
                - demands array
                - list of constraint matrices
                - base hop counts array
        """
        # Generate demands
        demands = 1 + 0.5 * rng.normal(0, 1, self.K)

        # Generate base hop counts
        base_hops = 3 + np.floor(rng.exponential(scale=1 / 0.3, size=self.K)) + 1

        # Create constraint matrices
        B_matrices = [np.eye(n_vars)]  # For x >= 0 constraints

        # Allocate paths to pairs (roughly equal distribution)
        paths_per_pair = n_vars // self.K
        remainder = n_vars % self.K
        current_path = 0

        # Add flow conservation constraints
        for k in range(self.K):
            n_paths = paths_per_pair + (1 if k < remainder else 0)
            B_k = np.zeros(n_vars)
            B_k[current_path : current_path + n_paths] = 1
            B_matrices.append(B_k)
            current_path += n_paths

        return demands, B_matrices, base_hops

    def generate_instance(
        self, n_vars: int, n_scenarios: int, seed: int | None = None
    ) -> CVQPParams:
        """
        Generate a problem instance.

        Args:
            n_vars: Number of paths.
            n_scenarios: Number of delay scenarios.
            seed: Random seed for reproducibility.

        Returns:
            CVQPParams instance containing the generated problem parameters.
        """
        rng = np.random.default_rng(seed)
        demands, B_matrices, base_hops = self.generate_path_structure(n_vars, rng)

        # Generate hop counts for each path
        current_path = 0
        hop_counts = np.zeros(n_vars)
        paths_per_pair = n_vars // self.K
        remainder = n_vars % self.K

        for k in range(self.K):
            n_paths = paths_per_pair + (1 if k < remainder else 0)
            # Use exp() to ensure variations are positive
            path_variation = np.exp(0.2 * rng.normal(0, 1, n_paths))
            hop_counts[current_path : current_path + n_paths] = np.ceil(
                base_hops[k] * path_variation
            )
            current_path += n_paths

        # Generate nominal delays based on hop counts
        # Use exp() to ensure delay multipliers are positive
        tau = hop_counts * np.exp(0.1 * rng.normal(0, 1, n_vars))

        # Generate delay matrix
        D = np.zeros((n_scenarios, n_vars))
        for i in range(n_scenarios):
            eta = rng.normal(0, self.sigma2)
            epsilon = rng.normal(0, self.sigma1, n_vars)
            # Use exp() for perturbations to ensure positive delays
            D[i] = tau * np.exp(epsilon + eta)

        # CVQP parameters
        P = self.gamma * np.eye(n_vars)
        q = tau.copy()  # Use nominal delays as costs
        A = D
        B = np.vstack(B_matrices)
        l = np.concatenate([np.zeros(n_vars), demands])  # x >= 0  # flow conservation
        u = np.concatenate(
            [
                np.inf * np.ones(n_vars),  # x <= inf (non-negativity)
                np.inf * np.ones(self.K),  # Σx_j <= inf (flow conservation)
            ]
        )
        kappa = 4.0 * np.max(tau)  # kappa = 4 * max nominal delay

        return CVQPParams(P=P, q=q, A=A, B=B, l=l, u=u, beta=self.beta, kappa=kappa)

    @property
    def name(self) -> str:
        """Return the name identifier for this problem."""
        return "network"


class SupplyChain(CVQProblem):
    """
    Supply chain contract selection problem.

    This class implements a supply chain problem where supplier costs are modeled
    with both regional and supplier-specific disruptions. The objective balances
    expected cost against cost variance, with constraints on regional allocations
    and a CVaR constraint on worst-case costs.

    Args:
        lambda_: Risk aversion parameter for cost variance.
        K: Number of geographical regions.
        l: Minimum allocation per supplier.
        u: Maximum allocation per supplier.
        sigma_r: Standard deviation for regional disruptions.
        sigma_s: Standard deviation for supplier-specific disruptions.
        beta: CVaR probability level.
        kappa: CVaR threshold.
        v: Maximum allocation per region.
        theta: Regional sensitivity to disruptions.
    """

    def __init__(
        self,
        lambda_: float = 0.1,
        K: int = 10,
        l: float = 0.0,
        u: float = 0.3,
        sigma_r: float = 0.3,
        sigma_s: float = 0.1,
        beta: float = 0.95,
        kappa: float = 1.5,
        v: float = 0.4,
        theta: float = 1.0,
    ):
        self.lambda_ = lambda_
        self.K = K
        self.l = l
        self.u = u
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.beta = beta
        self.kappa = kappa
        self.v = v
        self.theta = theta

    def generate_instance(
        self, n_vars: int, n_scenarios: int, seed: int | None = None
    ) -> CVQPParams:
        """
        Generate a problem instance.

        Args:
            n_vars: Number of suppliers.
            n_scenarios: Number of cost scenarios.
            seed: Random seed for reproducibility.

        Returns:
            CVQPParams instance containing the generated problem parameters.
        """
        # Assign suppliers to regions (roughly equal distribution)
        rng = np.random.default_rng(seed)
        suppliers_per_region = n_vars // self.K
        remainder = n_vars % self.K

        # Generate nominal costs
        c = 1 + 0.2 * rng.normal(0, 1, n_vars)

        # Generate cost matrix
        C = np.zeros((n_scenarios, n_vars))
        current_supplier = 0

        for k in range(self.K):
            n_suppliers = suppliers_per_region + (1 if k < remainder else 0)
            # Regional disruptions
            r_k = rng.normal(0, self.sigma_r, n_scenarios)

            for i in range(n_scenarios):
                # Supplier-specific disruptions
                zeta = rng.normal(0, self.sigma_s, n_suppliers)
                suppliers = slice(current_supplier, current_supplier + n_suppliers)
                C[i, suppliers] = c[suppliers] * (1 + self.theta * r_k[i] + zeta)

            current_supplier += n_suppliers

        # Compute covariance matrix
        C_centered = C - C.mean(axis=0)
        Sigma = (C_centered.T @ C_centered) / n_scenarios

        # CVQP parameters
        P = self.lambda_ * Sigma
        q = c
        A = C

        # Create B matrix for all constraints
        B_blocks = [
            np.ones((1, n_vars)),  # sum(x) = 1
            np.eye(n_vars),  # x >= l
            -np.eye(n_vars),  # x <= u
        ]

        # Add regional constraints
        current_supplier = 0
        for k in range(self.K):
            n_suppliers = suppliers_per_region + (1 if k < remainder else 0)
            B_k = np.zeros(n_vars)
            B_k[current_supplier : current_supplier + n_suppliers] = 1
            B_blocks.append(B_k.reshape(1, -1))
            current_supplier += n_suppliers

        B = np.vstack(B_blocks)

        # Create bounds vectors
        l_bounds = [1.0]  # sum(x) = 1
        l_bounds.extend([self.l] * n_vars)  # x >= l
        l_bounds.extend([-self.u] * n_vars)  # -x >= -u
        l_bounds.extend([-float("inf")] * self.K)  # regional constraints

        u_bounds = [1.0]  # sum(x) = 1
        u_bounds.extend([self.u] * n_vars)  # x <= u
        u_bounds.extend([-self.l] * n_vars)  # -x <= -l
        u_bounds.extend([self.v] * self.K)  # regional constraints

        return CVQPParams(
            P=P,
            q=q,
            A=A,
            B=B,
            l=np.array(l_bounds),
            u=np.array(u_bounds),
            beta=self.beta,
            kappa=self.kappa,
        )

    @property
    def name(self) -> str:
        """Return the name identifier for this problem."""
        return "supply_chain"


class ExperimentRunner:
    """
    Runner class for CVQP benchmark experiments.

    This class manages the execution of benchmark experiments across multiple
    problem types, problem sizes, and solvers. It handles instance generation,
    solution, and result collection.

    Args:
        problems: List of problem instances to benchmark.
        n_instances: Number of random instances to generate for each configuration.
        n_vars_list: List of problem sizes (number of variables) to test.
        n_scenarios_list: List of scenario counts to test.
        solvers: List of solvers to benchmark.
        base_seed: Base random seed for reproducibility.
    """

    def __init__(
        self,
        problems: list[CVQProblem],
        n_instances: int = 10,
        n_vars_list: list[int] = [100, 1000, 4000, 10000],
        n_scenarios_list: list[int] = [1000, 10000, 100000, 1000000],
        solvers: list[str] = ["mosek", "clarabel", "cvqp"],
        base_seed: int = 42,
    ):
        self.problems = problems
        self.n_instances = n_instances
        self.n_vars_list = n_vars_list
        self.n_scenarios_list = n_scenarios_list
        self.solvers = solvers
        self.base_seed = base_seed
        self.results: list[BenchmarkResults] = []

    def get_instance_seed(
        self,
        problem_name: str,
        solver: str,
        n_vars: int,
        n_scenarios: int,
        instance_idx: int,
    ) -> int:
        """
        Generate reproducible seed for a specific problem instance.

        Args:
            problem_name: Name of the problem type.
            solver: Name of the solver.
            n_vars: Number of variables.
            n_scenarios: Number of scenarios.
            instance_idx: Index of the instance.

        Returns:
            Seed for random number generation.
        """
        instance_str = f"{problem_name}_{solver}_{n_vars}_{n_scenarios}_{instance_idx}"
        return self.base_seed + hash(instance_str) % (2**32)

    def solve_instance(
        self, params: CVQPParams, solver: str
    ) -> tuple[float | None, str] | tuple[float | None, str, CVQPResults | None]:
        """
        Solve a CVQP instance with specified solver.

        Args:
            params: Problem parameters.
            solver: Name of solver to use.

        Returns:
            For CVQP: Tuple of (solve_time, status, results). If solver fails, time and results will be None.
            For others: Tuple of (solve_time, status). If solver fails, time will be None.
        """
        if solver in ["mosek", "clarabel"]:
            return self.solve_cvxpy(params, solver)
        else:
            return self.solve_cvqp(params)

    def solve_cvxpy(
        self, params: CVQPParams, solver: str, verbose: bool = False
    ) -> tuple[float | None, str]:
        """
        Solve using CVXPY with specified solver.

        Args:
            params: Problem parameters.
            solver: Either 'mosek' or 'clarabel'.
            verbose: Whether to print solver output.

        Returns:
            Tuple of (solve_time, status). If solver fails, time will be None.
        """
        # Map solver string to CVXPY solver constant
        solver_map = {"mosek": cp.MOSEK, "clarabel": cp.CLARABEL}
        if solver.lower() not in solver_map:
            raise ValueError(
                f"Unsupported solver: {solver}. Must be one of {list(solver_map.keys())}"
            )

        # Get CVXPY solver constant
        solver = solver_map[solver.lower()]

        # Define variable
        n = params.q.shape[0]
        x = cp.Variable(n)

        # Objective function
        obj = 0.5 * cp.quad_form(x, params.P, assume_PSD=True) + params.q @ x

        # Basic linear constraints (l <= Bx <= u)
        finite_lb = np.isfinite(params.l)
        finite_ub = np.isfinite(params.u)
        constraints = []
        if np.any(finite_lb):
            constraints.append(params.B[finite_lb] @ x >= params.l[finite_lb])
        if np.any(finite_ub):
            constraints.append(params.B[finite_ub] @ x <= params.u[finite_ub])

        # CVaR constraints
        constraints.append(cp.cvar(params.A @ x, params.beta) <= params.kappa)

        # Create and solve problem
        prob = cp.Problem(cp.Minimize(obj), constraints)
        try:
            prob.solve(solver=solver, verbose=verbose)
            solve_time = prob._solve_time
            status = prob.status

            if status != "optimal":
                return None, status

            return solve_time, status

        except Exception as e:
            logging.warning(f"Solver failed with error: {str(e)}")
            return None, "error"

    def solve_cvqp(
        self, params: CVQPParams
    ) -> tuple[float | None, str, CVQPResults | None]:
        """
        Solve using custom CVQP implementation.

        Args:
            params: Problem parameters.

        Returns:
            Tuple of (solve_time, status). If solver fails, time will be None.
        """
        try:
            solver = CVQP(params, CVQPConfig())
            results = solver.solve()
            return results.solve_time, results.problem_status, results
        except Exception as e:
            logging.warning(f"CVQP solver failed with error: {str(e)}")
            return None, "error", None

    def run_experiments(self):
        """Run all experiments and store results."""
        start_time = time.time()
        logging.info(
            f"Benchmarking solvers {', '.join(self.solvers)} on {len(self.problems)} CVQP problems, "
            f"{self.n_instances} runs per test"
        )

        for problem in self.problems:
            for solver in self.solvers:
                for n_vars in self.n_vars_list:
                    for n_scenarios in self.n_scenarios_list:
                        solve_times = []
                        statuses = []
                        cvqp_results = [] if solver == "cvqp" else None

                        for i in range(self.n_instances):
                            seed = self.get_instance_seed(
                                problem.name, solver, n_vars, n_scenarios, i
                            )
                            params = problem.generate_instance(
                                n_vars, n_scenarios, seed=seed
                            )
                            if solver == "cvqp":
                                solve_time, status, result = self.solve_instance(
                                    params, solver
                                )
                                cvqp_results.append(result)
                            else:
                                solve_time, status = self.solve_instance(params, solver)
                            solve_times.append(solve_time)
                            statuses.append(status)

                        # Only compute statistics for successful solves
                        valid_times = [t for t in solve_times if t is not None]
                        if valid_times:
                            avg_time = np.mean(valid_times)
                            std_time = np.std(valid_times)
                            logging.info(
                                f"problem={problem.name}, solver={solver}, n_vars={n_vars}, "
                                f"n_scenarios={n_scenarios}, solve_time={avg_time:.3f}s (±{std_time:.3f}s) "
                                f"[{len(valid_times)}/{len(solve_times)} succeeded]"
                            )
                        else:
                            logging.warning(
                                f"problem={problem.name}, solver={solver}, n_vars={n_vars}, "
                                f"n_scenarios={n_scenarios}, all solves failed"
                            )

                        self.results.append(
                            BenchmarkResults(
                                problem=problem.name,
                                solver=solver,
                                n_vars=n_vars,
                                n_scenarios=n_scenarios,
                                times=solve_times,
                                status=statuses,
                                cvqp_results=cvqp_results,
                            )
                        )
        total_time_minutes = (time.time() - start_time) / 60
        logging.info(f"Completed all experiments in {total_time_minutes:.1f} minutes")

    def save_results(self, filename: str):
        """
        Save experiment results to a pickle file.

        Args:
            filename: Path where to save the results.
        """
        results_dict = {
            "base_seed": self.base_seed,
            "n_instances": self.n_instances,
            "n_vars_list": self.n_vars_list,
            "n_scenarios_list": self.n_scenarios_list,
            "results": self.results,
        }
        with open(filename, "wb") as f:
            pickle.dump(results_dict, f)

    def load_results(self, filename: str):
        """
        Load experiment results from a pickle file.

        Args:
            filename: Path to the results file.
        """
        with open(filename, "rb") as f:
            results_dict = pickle.load(f)
            self.base_seed = results_dict["base_seed"]
            self.n_instances = results_dict["n_instances"]
            self.n_vars_list = results_dict["n_vars_list"]
            self.n_scenarios_list = results_dict["n_scenarios_list"]
            self.results = results_dict["results"]


def main():
    """Run CVQP benchmark experiments."""
    # # Create problem instances
    # problems = [
    #     PortfolioOptimization(),
    #     NetworkTraffic(),
    #     SupplyChain()
    # ]

    # # Create and run experiments
    # runner = ExperimentRunner(problems)
    # runner.run_experiments()
    # runner.save_results("cvqp_results.pkl")

    # Create portfolio problem
    portfolio = PortfolioOptimization()

    # Create experiment runner with a grid of sizes
    runner = ExperimentRunner(
        problems=[portfolio],
        n_instances=1,  # Small number for testing
        n_vars_list=[100],
        n_scenarios_list=[1_000, 3_000],
        # solvers=["clarabel", "mosek", "cvqp"],
        solvers=["cvqp"],
    )

    # Run and save
    runner.run_experiments()
    runner.save_results("data/cvqp_results.pkl")


if __name__ == "__main__":
    main()
