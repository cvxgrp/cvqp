"""
Script to benchmark our CVQP solver against MOSEK and Clarabel on a set of problems.
"""

from dataclasses import dataclass
import logging
import pickle
from pathlib import Path
import cvxpy as cp
import numpy as np
import warnings

from cvqp import CVQP, CVQPConfig, CVQPResults
from cvqp_problems import *

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%b %d %H:%M:%S"
)

warnings.filterwarnings("ignore", module="cvxpy")

TIME_LIMIT = 3600
SOLVER_CONFIGS = {
    "mosek": {"mosek_params": {"MSK_DPAR_OPTIMIZER_MAX_TIME": TIME_LIMIT}},
    "clarabel": {"time_limit": TIME_LIMIT},
}


@dataclass
class BenchmarkResults:
    """
    Store benchmark results for a specific problem configuration.

    Args:
        problem: Name of the problem being solved
        solver: Name of the solver used
        n_vars: Number of variables in the problem
        n_scenarios: Number of scenarios in the problem
        times: List of solve times for each instance. None indicates a failed solve
        status: Status returned by solver for each solve attempt ('optimal', 'infeasible', etc.)
        cvqp_results: List of detailed results from CVQP solver, if used
    """

    problem: str
    solver: str
    n_vars: int
    n_scenarios: int
    times: list[float]
    status: list[str]
    cvqp_results: list[CVQPResults | None] = None

    @property
    def success_rate(self) -> float:
        """Return fraction of successful solves."""
        return np.sum(~np.isnan(self.times)) / len(self.times)

    @property
    def avg_time(self) -> float | None:
        """Average time of successful solves."""
        return np.nanmean(self.times)

    @property
    def std_time(self) -> float | None:
        """Standard deviation of successful solve times."""
        return np.nanstd(self.times)

    @property
    def num_success(self) -> int:
        """Return the total number of successful solves."""
        return int(np.sum(~np.isnan(self.times)))

    @property
    def num_total(self) -> int:
        """Return the total number of solve attempts."""
        return len(self.times)


class CVQPBenchmark:
    """
    Runner class for CVQP benchmark experiments.

    Args:
        problems: List of problem instances to benchmark
        n_instances: Number of random instances to generate for each configuration
        n_vars_list: List of problem sizes (number of variables) to test
        n_scenarios_list: List of scenario counts to test
        solvers: List of solvers to benchmark
        base_seed: Base random seed for reproducibility
        n_consecutive_failures: Number of consecutive failures before stopping (None to run all)
    """

    def __init__(
        self,
        problems: list[CVQProblem],
        n_instances: int = 5,
        n_vars_list: list[int] = [100, 1000],
        n_scenarios_list: list[int] = [int(x) for x in [1e4, 3e4, 1e5, 3e5, 1e6, 3e6]],
        solvers: list[str] = ["cvqp", "mosek", "clarabel"],
        base_seed: int = 42,
        n_consecutive_failures: int | None = None,
    ):
        self.problems = problems
        self.n_instances = n_instances
        self.n_vars_list = n_vars_list
        self.n_scenarios_list = n_scenarios_list
        self.solvers = solvers
        self.base_seed = base_seed
        self.n_consecutive_failures = n_consecutive_failures
        self.results: dict[str, list[BenchmarkResults]] = {p.name: [] for p in problems}

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
            problem_name: Name of the problem type
            solver: Name of the solver
            n_vars: Number of variables
            n_scenarios: Number of scenarios
            instance_idx: Index of the instance

        Returns:
            Seed for random number generation
        """
        instance_str = f"{problem_name}_{solver}_{n_vars}_{n_scenarios}_{instance_idx}"
        return self.base_seed + hash(instance_str) % (2**32)

    @staticmethod
    def format_time_s(t: float) -> str:
        """Format time in scientific notation."""
        return f"{t:.2e}s"

    def solve_instance(
        self, params: CVQPParams, solver: str
    ) -> tuple[float | None, str] | tuple[float | None, str, CVQPResults | None]:
        """
        Solve a CVQP instance with specified solver.

        Args:
            params: Problem parameters
            solver: Name of solver to use

        Returns:
            Tuple of (solve_time, status) or (solve_time, status, results) for CVQP solver
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
            params: Problem parameters
            solver: Name of solver to use
            verbose: Whether to print solver output

        Returns:
            Tuple of (solve_time, status)
        """
        solver_map = {"mosek": cp.MOSEK, "clarabel": cp.CLARABEL}
        if solver.lower() not in solver_map:
            raise ValueError(f"Unsupported solver: {solver}")

        solver = solver_map[solver.lower()]
        solver_opts = SOLVER_CONFIGS[solver.lower()]

        n = params.q.shape[0]
        x = cp.Variable(n)
        obj = 0.5 * cp.quad_form(x, params.P, assume_PSD=True) + params.q @ x

        finite_lb = np.isfinite(params.l)
        finite_ub = np.isfinite(params.u)
        constraints = []
        if np.any(finite_lb):
            constraints.append(params.B[finite_lb] @ x >= params.l[finite_lb])
        if np.any(finite_ub):
            constraints.append(params.B[finite_ub] @ x <= params.u[finite_ub])
        constraints.append(cp.cvar(params.A @ x, params.beta) <= params.kappa)

        prob = cp.Problem(cp.Minimize(obj), constraints)
        try:
            prob.solve(solver=solver, verbose=verbose, **(solver_opts or {}))
            solve_time = prob._solve_time
            status = prob.status

            if status != "optimal":
                return np.nan, status

            return solve_time, status

        except Exception as e:
            logging.warning(f"Solver failed with error: {str(e)}")
            return np.nan, "error"

    def solve_cvqp(
        self, params: CVQPParams
    ) -> tuple[float | None, str, CVQPResults | None]:
        """
        Solve using custom CVQP implementation.

        Args:
            params: Problem parameters

        Returns:
            Tuple of (solve_time, status, results)
        """
        try:
            solver = CVQP(params, CVQPConfig())
            results = solver.solve()
            return results.solve_time, results.problem_status, results
        except Exception as e:
            logging.warning(f"CVQP solver failed with error: {str(e)}")
            return np.nan, "error", None

    def run_experiments(self):
        """Run all experiments and store results."""
        Path("data").mkdir(exist_ok=True)

        logging.info("Starting CVQP benchmark")
        logging.info(f"Testing n_vars values: {self.n_vars_list}")
        logging.info(f"Testing n_scenarios values: {self.n_scenarios_list}")
        logging.info(f"Testing solvers: {[s.upper() for s in self.solvers]}")
        logging.info(f"Running {self.n_instances} instances per configuration")
        if self.n_consecutive_failures:
            logging.info(
                f"Will stop after {self.n_consecutive_failures} consecutive failures"
            )

        for problem in self.problems:
            logging.info(f"Benchmarking problem: {problem.name}")

            for n_vars in self.n_vars_list:
                logging.info(f"n_vars={n_vars:.0e}:")

                for n_scenarios in self.n_scenarios_list:
                    logging.info(f"  n_scenarios={n_scenarios:.0e}:")

                    for solver in self.solvers:
                        solve_times = []
                        statuses = []
                        cvqp_results = [] if solver == "cvqp" else None
                        consecutive_failures = 0

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

                            if np.isnan(solve_time):
                                consecutive_failures += 1
                            else:
                                consecutive_failures = 0

                            if (
                                self.n_consecutive_failures is not None
                                and consecutive_failures >= self.n_consecutive_failures
                            ):
                                logging.info(
                                    f"    {solver.upper():<8s} : stopping after "
                                    f"{consecutive_failures} consecutive failures"
                                )
                                break

                        result = BenchmarkResults(
                            problem=problem.name,
                            solver=solver,
                            n_vars=n_vars,
                            n_scenarios=n_scenarios,
                            times=solve_times,
                            status=statuses,
                            cvqp_results=cvqp_results,
                        )

                        if result.num_success > 0:
                            logging.info(
                                f"    {solver.upper():<8s} : "
                                f"{self.format_time_s(result.avg_time)} ± "
                                f"{self.format_time_s(result.std_time)} "
                                f"[{result.num_success}/{result.num_total} succeeded]"
                            )
                        else:
                            logging.info(
                                f"    {solver.upper():<8s} : all {result.num_total} attempts failed"
                            )

                        self.results[problem.name].append(result)
                        self.save_problem_results(problem.name)

        logging.info("All experiments completed!")

    def save_problem_results(self, problem_name: str):
        """
        Save results for a specific problem to its own pickle file.

        Args:
            problem_name: Name of the problem whose results should be saved
        """
        results_dict = {
            "base_seed": self.base_seed,
            "n_instances": self.n_instances,
            "n_vars_list": self.n_vars_list,
            "n_scenarios_list": self.n_scenarios_list,
            "results": self.results[problem_name],
        }
        filename = f"data/{problem_name.lower()}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(results_dict, f)

    def load_problem_results(self, problem_name: str):
        """
        Load results for a specific problem from its pickle file.

        Args:
            problem_name: Name of the problem whose results should be loaded
        """
        filename = f"data/{problem_name.lower()}.pkl"
        with open(filename, "rb") as f:
            results_dict = pickle.load(f)
            self.base_seed = results_dict["base_seed"]
            self.n_instances = results_dict["n_instances"]
            self.n_vars_list = results_dict["n_vars_list"]
            self.n_scenarios_list = results_dict["n_scenarios_list"]
            self.results[problem_name] = results_dict["results"]


def main():
    """Run CVQP benchmark experiments."""
    # Create problem instances
    problems = [
        PortfolioOptimization(),
        # NetworkTraffic(),
        # SupplyChain()
    ]

    # Create experiment runner
    runner = CVQPBenchmark(
        problems=problems,
        n_instances=5,
        n_vars_list=[10, 100],
        n_scenarios_list=[100, 1000],
        solvers=["cvqp", "mosek", "clarabel"],
        n_consecutive_failures=2,  # Stop after 2 consecutive failures
    )

    # Run experiments
    runner.run_experiments()


if __name__ == "__main__":
    main()
