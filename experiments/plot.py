"""Generate paper figures from CSV benchmark results.

Usage:
    python plot.py              # generate all figures
    python plot.py --show       # also display interactively
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
FIGS_DIR = Path(__file__).parent / "figs"
COLORS = {"CVQP": "#2E86AB", "Ours": "#2E86AB", "MOSEK": "#A23B72", "CLARABEL": "#F18F01"}


def setup_style():
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.figsize": (6, 4.5),
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "backend": "ps",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{gensymb} \usepackage{amsmath}",
    })


def _read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _avg_times(rows, solver_col: str, size_col: str, solver: str) -> tuple[list, list]:
    """Group rows by size, compute mean time for a solver."""
    groups = defaultdict(list)
    for r in rows:
        if r[solver_col] == solver and r["time"]:
            groups[int(r[size_col])].append(float(r["time"]))
    sizes = sorted(groups)
    means = [np.mean(groups[s]) for s in sizes]
    return sizes, means


def plot_cvqp(csv_path: Path, name: str, save: bool = False, show: bool = False):
    """Plot solver comparison for a CVQP benchmark."""
    setup_style()
    rows = _read_csv(csv_path)
    fig, ax = plt.subplots()

    for solver in ["CLARABEL", "MOSEK", "CVQP"]:
        xs, ys = _avg_times(rows, "solver", "n_scenarios", solver)
        if xs:
            ax.plot(xs, ys, label=solver, marker="o", color=COLORS.get(solver, "gray"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of scenarios")
    ax.set_ylabel("Solve time (seconds)")
    ax.legend()
    plt.tight_layout()

    if save:
        FIGS_DIR.mkdir(exist_ok=True)
        fig.savefig(FIGS_DIR / f"{name}.pdf")
    if show:
        plt.show()
    return fig


def plot_projection(csv_path: Path, save: bool = False, show: bool = False):
    """Plot solver comparison for projection benchmark."""
    setup_style()
    rows = _read_csv(csv_path)
    fig, ax = plt.subplots()

    solvers = sorted(set(r["solver"] for r in rows))
    for solver in solvers:
        xs, ys = _avg_times(rows, "solver", "m", solver)
        if xs:
            label = solver if solver == "Ours" else solver.upper()
            ax.plot(xs, ys, label=label, marker="o", color=COLORS.get(solver, "gray"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of scenarios")
    ax.set_ylabel("Solve time (seconds)")
    ax.legend()
    plt.tight_layout()

    if save:
        FIGS_DIR.mkdir(exist_ok=True)
        fig.savefig(FIGS_DIR / "proj.pdf")
    if show:
        plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    for name in ["portfolio", "quantile_regression"]:
        path = RESULTS_DIR / f"{name}.csv"
        if path.exists():
            plot_cvqp(path, name, save=True, show=args.show)
            print(f"Generated {name}.pdf")
        else:
            print(f"Skipping {name} (no results)")

    path = RESULTS_DIR / "projection.csv"
    if path.exists():
        plot_projection(path, save=True, show=args.show)
        print("Generated proj.pdf")
    else:
        print("Skipping projection (no results)")


if __name__ == "__main__":
    main()
