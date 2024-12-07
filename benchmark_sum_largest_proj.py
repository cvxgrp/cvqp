"""
Script to benchmark our algorithm for projecting a vector onto the sum-k-largest
constraint set (where the sum of the k largest elements must not exceed alpha).
The performance is compared against the commercial solver MOSEK and the
open-source solver Clarabel.
"""

import numpy as np
import cvxpy as cp
import time
import pickle
from dataclasses import dataclass
from pathlib import Path
import logging
import warnings
import mybindings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: %(message)s", datefmt="%b %d %H:%M:%S"
)

# Suppress CVXPY warnings
warnings.filterwarnings("ignore", module="cvxpy")

# Define available solvers and their time limits
TIME_LIMIT = 3600
SOLVER_CONFIGS = {
    "MOSEK": {"mosek_params": {"MSK_DPAR_OPTIMIZER_MAX_TIME": TIME_LIMIT}},
    "CLARABEL": {"time_limit": TIME_LIMIT},
    "Ours": None,
}


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


@dataclass
class Projection:
    """
    Single instance of a sum-k-largest projection problem.

    Args:
        v: Input vector to project.
        k: Number of largest elements to consider.
        d: Upper bound on sum of k largest elements.
        tau: Scaling factor for the bound d.
        m: Number of scenarios (length of vector v).
        seed: Random seed used to generate this instance.
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
    Store benchmark results for a specific problem configuration.

    Args:
        solver: Name of the solver used.
        m: Number of scenarios in the problem.
        tau: Scaling factor used in problem generation.
        times: List of solve times for each instance. np.nan indicates a failed solve.
        status: Status returned by solver for each solve attempt.
    """

    solver: str
    m: int
    tau: float
    times: list[float]
    status: list[str]

    @property
    def success_rate(self) -> float:
        """Return fraction of successful solves."""
        return np.sum(~np.isnan(self.times)) / len(self.times)

    @property
    def avg_time(self) -> float:
        """
        Average time of successful solves.

        Returns:
            Mean solve time of successful solves, or np.nan if all solves failed.
        """
        return np.nanmean(self.times)

    @property
    def std_time(self) -> float:
        """
        Standard deviation of successful solve times.

        Returns:
            Standard deviation of successful solve times, or np.nan if all solves failed.
        """
        return np.nanstd(self.times)

    @property
    def num_success(self) -> int:
        """Number of successful solves."""
        return int(np.sum(~np.isnan(self.times)))

    @property
    def num_total(self) -> int:
        """Total number of solve attempts."""
        return len(self.times)


class ProjectionBenchmark:
    """Class to manage benchmarking experiments for sum-k-largest projection problems."""

    def __init__(self):
        """Initialize the benchmark class."""
        pass

    def solve_instance_cvxpy(
        self, instance: Projection, solver: str
    ) -> tuple[float, str]:
        """
        Solve sum-k-largest projection problem using CVXPY.

        Args:
            instance: Projection instance containing problem parameters
            solver: Name of the CVXPY solver to use

        Returns:
            Tuple containing (solve_time, status). solve_time is np.nan if the solve failed
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
        Solve sum-k-largest projection problem using custom C++ implementation.

        Args:
            instance: Projection instance containing problem parameters

        Returns:
            Tuple containing (solve_time, status). solve_time is np.nan if the solve failed
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

        Args:
            m: Number of scenarios
            tau: Scaling factor for the bound d
            seed: Random seed for reproducibility

        Returns:
            Projection instance containing the generated problem parameters
        """
        rng = np.random.RandomState(seed)
        beta = 0.95

        k = int(np.ceil((1 - beta) * m))
        v = rng.uniform(0, 1, m)
        d = tau * cp.sum_largest(v, k).value

        return Projection(v=v, k=k, d=d, tau=tau, m=m, seed=seed)

    def get_reproducible_seed(self, m: int, tau: float, instance: int) -> int:
        """Generate a reproducible seed within valid range."""
        param_str = f"{m}_{tau}_{instance}"
        return abs(hash(param_str)) % (2**32 - 1)

    @staticmethod
    def format_time_s(t: float) -> str:
        """Format time in seconds using scientific notation."""
        return f"{t:.2e}s"

    def run_benchmark(
        self,
        m_list: list[int],
        tau_list: list[float],
        n_instances: int,
        solvers: list[str],
    ) -> list[ProjectionResults]:
        """
        Run complete benchmark experiment.

        Args:
            m_list: List of scenario counts to test
            tau_list: List of tau values to test
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
                    solver_failed = False

                    for i in range(n_instances):
                        if solver_failed:
                            break

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

                        if np.isnan(solve_time):
                            solver_failed = True
                            logging.warning(
                                f"    {solver} failed on instance {i}. Skipping remaining instances."
                            )

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
        """Save benchmark results to pickle file."""
        Path("data").mkdir(exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(results, f)

    @staticmethod
    def load_results(filename: str) -> list[ProjectionResults]:
        """Load benchmark results from pickle file."""
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

        Args:
            m_list: List of scenario counts to test
            tau_list: List of tau values to test
            n_instances: Number of random instances per configuration
            solvers: List of solvers to benchmark
            output_file: Path to save the results
        """
        logging.info("Starting sum-k-largest projection benchmark")
        logging.info(f"Testing m values: {m_list}")
        logging.info(f"Testing τ values: {tau_list}")
        logging.info(f"Testing solvers: {solvers}")
        logging.info(f"Running {n_instances} instances per configuration")

        start_time = time.time()
        results = self.run_benchmark(m_list, tau_list, n_instances, solvers)
        total_time = (time.time() - start_time) / 60

        logging.info(f"\nCompleted all experiments in {total_time:.1f} minutes")
        self.save_results(results, output_file)


if __name__ == "__main__":
    benchmark = ProjectionBenchmark()

    # Set up experiment
    tau_list = [0.1, 0.9]
    m_list = [int(x) for x in [1e3, 3e3, 1e4]]  # [int(x) for x in [1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7]]
    n_instances = 50
    solvers = ["Ours", "MOSEK", "CLARABEL"]

    benchmark.run_experiment(
        m_list=m_list,
        tau_list=tau_list,
        n_instances=n_instances,
        solvers=solvers,
        output_file="data/sum_largest_proj.pkl",
    )
