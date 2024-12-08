"""
Benchmark script for evaluating performance of sum-k-largest projection algorithm.

This script compares different solvers' performance on the sum-k-largest projection
problem, including our custom implementation against standard solvers like MOSEK and 
CLARABEL.
"""

import numpy as np
import cvxpy as cp
import time
import pickle
from dataclasses import dataclass
from pathlib import Path
import logging
import warnings

from sum_largest_proj import proj_sum_largest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: %(message)s", datefmt="%b %d %H:%M:%S"
)

warnings.filterwarnings("ignore", module="cvxpy")

TIME_LIMIT = 3600
SOLVER_CONFIGS = {
    "MOSEK": {"mosek_params": {"MSK_DPAR_OPTIMIZER_MAX_TIME": TIME_LIMIT}},
    "CLARABEL": {"time_limit": TIME_LIMIT},
    "Ours": None,
}


@dataclass
class Projection:
    """
    Single instance of a sum-k-largest projection problem.

    The problem involves projecting a vector onto the set where the sum of its
    k largest elements is bounded by d.

    Args:
        v: Input vector to project
        k: Number of largest elements to consider in sum constraint
        d: Upper bound on sum of k largest elements
        tau: Hardness parameter in (0,1) controlling problem difficulty
        m: Length of vector v (number of scenarios)
        seed: Random seed used to generate this instance for reproducibility
    """

    v: np.ndarray
    k: int
    d: float
    tau: float
    m: int
    seed: int


@dataclass
class ProjectionResults:
    """
    Results from benchmarking a solver on multiple problem instances.

    Stores solve times and success information for a specific solver on
    multiple instances of a problem with fixed size (m) and hardness (tau).

    Args:
        solver: Name of the solver used
        m: Number of scenarios in the tested problems
        tau: Hardness parameter used in problem generation
        times: List of solve times (np.nan indicates failed solve)
        status: List of solver status strings for each attempt
    """

    solver: str
    m: int
    tau: float
    times: list[float]
    status: list[str]

    @property
    def success_rate(self) -> float:
        """Return the fraction of successful solves out of all attempts."""
        return np.sum(~np.isnan(self.times)) / len(self.times)

    @property
    def avg_time(self) -> float:
        """
        Calculate the average time of successful solves.

        Returns:
            Mean solve time of successful solves, or np.nan if all solves failed
        """
        return np.nanmean(self.times)

    @property
    def std_time(self) -> float:
        """
        Calculate the standard deviation of successful solve times.

        Returns:
            Standard deviation of successful solve times, or np.nan if all solves failed
        """
        return np.nanstd(self.times)

    @property
    def num_success(self) -> int:
        """Return the total number of successful solves."""
        return int(np.sum(~np.isnan(self.times)))

    @property
    def num_total(self) -> int:
        """Return the total number of solve attempts."""
        return len(self.times)


class ProjectionBenchmark:
    """
    Benchmark manager for sum-k-largest projection problems.
    
    Handles generation of test instances, running solvers, collecting results,
    and saving/loading benchmark data.
    """

    def solve_instance_cvxpy(
        self, instance: Projection, solver: str
    ) -> tuple[float, str]:
        """
        Solve a projection instance using a CVXPY-supported solver.

        Formulates the problem using CVXPY's interface and solves it with
        the specified solver, measuring the solution time.

        Args:
            instance: Problem instance to solve
            solver: Name of CVXPY-supported solver to use

        Returns:
            Tuple of (solve_time, status). solve_time is np.nan if solve failed
        """
        try:
            x = cp.Variable(instance.v.shape)
            objective = cp.Minimize(cp.sum_squares(x - instance.v))

            k = int(instance.k)
            if k <= 0:
                raise ValueError(f"k must be positive, got {k}")

            constraints = [cp.sum_largest(x, k) <= instance.d]
            prob = cp.Problem(objective, constraints)

            solver_opts = SOLVER_CONFIGS[solver]
            prob.solve(solver=solver, verbose=False, **(solver_opts or {}))

            if prob.status in ["optimal", "optimal_inaccurate"]:
                return prob._solve_time, prob.status
            logging.error(
                f"Solver {solver} failed to find optimal solution. Status: {prob.status}"
            )
            return np.nan, prob.status

        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return np.nan, "failed"

    def solve_instance_custom(self, instance: Projection) -> tuple[float, str]:
        """
        Solve a projection instance using our custom C++ implementation.

        Args:
            instance: Problem instance to solve

        Returns:
            Tuple of (solve_time, status). solve_time is np.nan if solve failed
        """
        try:
            start_time = time.time()
            _ = proj_sum_largest(instance.v, int(instance.k), instance.d)
            solve_time = time.time() - start_time
            return solve_time, "optimal"
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return np.nan, "failed"

    def generate_instance(self, m: int, tau: float, seed: int) -> Projection:
        """
        Generate a random sum-k-largest projection problem instance.

        Creates a problem instance with uniformly distributed vector entries
        and a CVaR-style constraint on the sum of largest elements.

        Args:
            m: Number of scenarios (vector length)
            tau: Hardness parameter in (0,1)
            seed: Random seed for reproducibility

        Returns:
            Generated problem instance with specified parameters
        """
        rng = np.random.RandomState(seed)
        beta = 0.95

        k = int(np.ceil((1 - beta) * m))
        v = rng.uniform(0, 1, m)
        d = tau * cp.sum_largest(v, k).value

        return Projection(v=v, k=k, d=d, tau=tau, m=m, seed=seed)

    def get_reproducible_seed(self, m: int, tau: float, instance: int) -> int:
        """
        Generate a reproducible seed for a specific problem configuration.

        Args:
            m: Number of scenarios
            tau: Hardness parameter
            instance: Instance number

        Returns:
            Deterministic seed value within valid range
        """
        param_str = f"{m}_{tau}_{instance}"
        return abs(hash(param_str)) % (2**32 - 1)

    @staticmethod
    def format_time_s(t: float) -> str:
        """
        Format a time value in seconds using scientific notation.

        Args:
            t: Time value in seconds

        Returns:
            Formatted string with time in scientific notation
        """
        return f"{t:.2e}s"

    def run_benchmark(
        self,
        m_list: list[int],
        tau_list: list[float],
        n_instances: int,
        solvers: list[str],
    ) -> list[ProjectionResults]:
        """
        Run complete benchmark experiment across all configurations.

        Tests each solver on multiple random instances for each combination
        of problem size (m) and hardness parameter (tau).

        Args:
            m_list: List of scenario counts to test
            tau_list: List of hardness parameters to test
            n_instances: Number of random instances per configuration
            solvers: List of solvers to benchmark

        Returns:
            List of ProjectionResults objects containing benchmark results
        """
        results = []

        for m in m_list:
            logging.info(f"\nBenchmarking problems with m={m: .0e}")
            for tau in tau_list:
                logging.info(f"\n  τ={tau}:")
                for solver in solvers:
                    times = []
                    statuses = []

                    for i in range(n_instances):
                        seed = self.get_reproducible_seed(m, tau, i)
                        instance = self.generate_instance(m, tau, seed)

                        if solver == "Ours":
                            solve_time, status = self.solve_instance_custom(instance)
                        else:
                            solve_time, status = self.solve_instance_cvxpy(
                                instance, solver
                            )

                        times.append(solve_time)
                        statuses.append(status)

                    result = ProjectionResults(
                        solver=solver, m=m, tau=tau, times=times, status=statuses
                    )

                    if result.num_success > 0:
                        logging.info(
                            f"    {solver:8s}: {self.format_time_s(result.avg_time):>10s} ± "
                            f"{self.format_time_s(result.std_time):>9s} "
                            f"[{result.num_success}/{result.num_total} succeeded]"
                        )
                    else:
                        logging.info(
                            f"    {solver:8s}: all {result.num_total} attempts failed"
                        )

                    results.append(result)

        return results

    @staticmethod
    def save_results(results: list[ProjectionResults], filename: str):
        """
        Save benchmark results to a pickle file.

        Args:
            results: List of benchmark results to save
            filename: Path where results should be saved
        """
        Path("data").mkdir(exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(results, f)

    @staticmethod
    def load_results(filename: str) -> list[ProjectionResults]:
        """
        Load benchmark results from a pickle file.

        Args:
            filename: Path to the pickle file containing results

        Returns:
            List of loaded ProjectionResults objects
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def run_experiment(
        self,
        m_list: list[int],
        tau_list: list[float],
        n_instances: int,
        solvers: list[str],
        output_file: str,
    ):
        """
        Run a complete benchmark experiment and save results.

        Orchestrates the entire benchmarking process, including logging 
        configuration details, running benchmarks, and saving results.

        Args:
            m_list: List of scenario counts to test
            tau_list: List of hardness parameters to test
            n_instances: Number of random instances per configuration
            solvers: List of solvers to benchmark
            output_file: Path to save the results
        """
        logging.info("Starting sum-k-largest projection benchmark")
        logging.info(f"Testing m values: {m_list}")
        logging.info(f"Testing τ values: {tau_list}")
        logging.info(f"Testing solvers: {solvers}")
        logging.info(f"Running {n_instances} instances per configuration")

        results = self.run_benchmark(m_list, tau_list, n_instances, solvers)

        logging.info("All experiments completed!")
        self.save_results(results, output_file)


if __name__ == "__main__":
    benchmark = ProjectionBenchmark()

    tau_list = [0.1, 0.9]
    m_list = [int(x) for x in [1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7]]
    n_instances = 50
    solvers = ["Ours", "MOSEK", "CLARABEL"]

    benchmark.run_experiment(
        m_list=m_list,
        tau_list=tau_list,
        n_instances=n_instances,
        solvers=solvers,
        output_file="data/sum_largest_proj.pkl",
    )
