"""Run benchmark experiments for the CVQP paper.

Usage:
    python run.py portfolio
    python run.py quantile_regression
    python run.py projection
"""

import argparse
import csv
import logging
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import warnings

import cvqp
from problems import make_portfolio, make_quantile_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%b %d %H:%M:%S")
warnings.filterwarnings("ignore", module="cvxpy")

RESULTS_DIR = Path(__file__).parent / "results"
MAX_TIME = 7200
CVXPY_OPTS = {
    "MOSEK": {"mosek_params": {"MSK_DPAR_OPTIMIZER_MAX_TIME": MAX_TIME}},
    "CLARABEL": {"time_limit": MAX_TIME},
}


def _seed(label: str, size: int, idx: int, base: int = 42) -> int:
    return base + hash(f"{label}_{size}_{idx}") % (2**32)


def _solve_cvqp(data: dict) -> tuple[float, str]:
    try:
        result = cvqp.solve(**data)
        return result.time, result.status
    except Exception:
        return float("nan"), "error"


def _solve_cvxpy(prob: cp.Problem, solver: str) -> tuple[float, str]:
    try:
        solver_map = {"MOSEK": cp.MOSEK, "CLARABEL": cp.CLARABEL}
        prob.solve(solver=solver_map[solver], verbose=False,
                   **CVXPY_OPTS.get(solver, {}), canon_backend=cp.SCIPY_CANON_BACKEND)
        if prob.status == "optimal":
            return prob._solve_time, prob.status
        return float("nan"), prob.status
    except Exception:
        return float("nan"), "error"


# ---------------------------------------------------------------------------
# CVQP benchmarks (portfolio, quantile regression)
# ---------------------------------------------------------------------------

def run_cvqp(
    name: str,
    make_problem,
    n_vars_list: list[int],
    n_scenarios_list: list[int],
    n_instances: int = 3,
    solvers: list[str] = ["CVQP", "MOSEK", "CLARABEL"],
):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.csv"

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["solver", "n_vars", "n_scenarios", "instance", "time", "status"])

        for n_vars in n_vars_list:
            for n_scenarios in n_scenarios_list:
                logging.info(f"n_vars={n_vars}, n_scenarios={n_scenarios:.0e}")

                for solver in solvers:
                    times = []
                    for i in range(n_instances):
                        seed = _seed(f"{name}_{n_vars}", n_scenarios, i)
                        data, cvxpy_prob = make_problem(n_vars, n_scenarios, seed=seed)

                        if solver == "CVQP":
                            t, s = _solve_cvqp(data)
                        else:
                            t, s = _solve_cvxpy(cvxpy_prob, solver)

                        w.writerow([solver, n_vars, n_scenarios, i,
                                    f"{t:.6f}" if not np.isnan(t) else "", s])
                        times.append(t)

                    avg = np.nanmean(times)
                    ok = sum(1 for t in times if not np.isnan(t))
                    logging.info(f"  {solver:<8s}: {avg:.2e}s ({ok}/{n_instances} OK)")

    logging.info(f"Saved {path}")


# ---------------------------------------------------------------------------
# Projection benchmark
# ---------------------------------------------------------------------------

def _make_projection_instance(m: int, tau: float, seed: int):
    rng = np.random.RandomState(seed)
    k = int(np.ceil(0.05 * m))  # beta=0.95
    v = rng.uniform(0, 1, m)
    d = tau * float(cp.sum_largest(v, k).value)
    return v, k, d


def run_projection(
    vector_sizes: list[int],
    tau: float = 0.5,
    n_instances: int = 10,
    solvers: list[str] = ["Ours", "MOSEK", "CLARABEL"],
):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "projection.csv"

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["solver", "m", "tau", "instance", "time", "status"])

        for m in vector_sizes:
            logging.info(f"m={m:.0e}")

            for solver in solvers:
                times = []
                for i in range(n_instances):
                    seed = _seed(f"proj_{m}", int(tau * 1000), i)
                    v, k, d = _make_projection_instance(m, tau, seed)

                    if solver == "Ours":
                        try:
                            t0 = time.time()
                            cvqp.proj_sum_largest(v, k, d)
                            t, s = time.time() - t0, "optimal"
                        except Exception:
                            t, s = float("nan"), "error"
                    else:
                        x = cp.Variable(v.shape)
                        prob = cp.Problem(
                            cp.Minimize(cp.sum_squares(x - v)),
                            [cp.sum_largest(x, k) <= d],
                        )
                        t, s = _solve_cvxpy(prob, solver)

                    w.writerow([solver, m, tau, i,
                                f"{t:.6f}" if not np.isnan(t) else "", s])
                    times.append(t)

                avg = np.nanmean(times)
                ok = sum(1 for t in times if not np.isnan(t))
                logging.info(f"  {solver:<8s}: {avg:.2e}s ({ok}/{n_instances} OK)")

    logging.info(f"Saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run CVQP paper experiments")
    parser.add_argument("experiment", choices=["portfolio", "quantile_regression", "projection"])
    args = parser.parse_args()

    if args.experiment == "portfolio":
        run_cvqp(
            "portfolio", make_portfolio,
            n_vars_list=[2000],
            n_scenarios_list=[int(x) for x in [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]],
        )
    elif args.experiment == "quantile_regression":
        run_cvqp(
            "quantile_regression", make_quantile_regression,
            n_vars_list=[500],
            n_scenarios_list=[int(x) for x in [1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7]],
        )
    elif args.experiment == "projection":
        run_projection(
            vector_sizes=[int(x) for x in [1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7]],
        )


if __name__ == "__main__":
    main()
