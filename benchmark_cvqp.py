"""
Script to benchmark CVQP solver. 
"""

from dataclasses import dataclass
import logging
import pickle
import time
import cvxpy as cp
import numpy as np

from cvqp import CVQP, CVQPConfig, CVQPResults
from cvqp_problems import *

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
