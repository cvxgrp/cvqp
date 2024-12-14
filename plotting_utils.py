"""
Utility functions for plotting benchmark results and generating result tables.
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable
from IPython.display import display

from benchmark_proj import *
from benchmark_cvqp import *

# ========= Plotting Style Configuration =========


def setup_plotting_style():
    """Configure matplotlib plotting style for publication quality figures."""
    plt.rcParams.update(
        {
            # Font sizes
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            # Figure properties
            "figure.figsize": (8, 6),
            "lines.linewidth": 2,
            "lines.markersize": 8,
            # Grid properties
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": ":",
        }
    )


# ========= Data Loading Functions =========


def load_proj_results(data_dir="data"):
    """
    Load and combine projection benchmark results from all pickle files in the data directory.

    Args:
        data_dir: Directory containing the pickle files

    Returns:
        List of ProjectionResults objects
    """
    all_results = []
    data_path = Path(data_dir)

    # Load each pickle file
    for pkl_file in data_path.glob("proj_m=*.pkl"):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            all_results.extend(data["results"])

    return all_results


def load_cvqp_results(problem_name: str, data_dir: str = "data"):
    """
    Load CVQP benchmark results for a specific problem and number of variables.

    Args:
        problem_name: Name of the problem to load
        data_dir: Directory containing the pickle files

    Returns:
        List of BenchmarkResults objects for the specified problem
    """
    data_path = Path(data_dir) / f"{problem_name.lower()}.pkl"
    if not data_path.exists():
        raise FileNotFoundError(f"No results file found for {problem_name}")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    return data["results"]


# ========= Plotting Functions =========


def create_benchmark_plot(
    results: list[Any],
    get_groups: Callable[[list[Any]], list[Any]],
    filter_group: Callable[[list[Any], Any], list[Any]],
    get_solvers: Callable[[list[Any]], list[str]],
    filter_solver: Callable[[list[Any], str], list[Any]],
    get_x_values: Callable[[list[Any]], list[float]],
    get_times: Callable[[list[Any]], list[float]],
    get_title: Callable[[Any], str],
    markers: dict[str, str],
    save_path: Callable[[Path, Any], Path] | None = None,
    save_figures: bool = False,
) -> dict[Any, plt.Figure]:
    """
    Create benchmark plots with common styling and configuration.

    Args:
        results: List of benchmark results
        get_groups: Function to get list of group values (e.g., tau values)
        filter_group: Function to filter results for a specific group
        get_solvers: Function to get list of solvers from results
        filter_solver: Function to filter results for a specific solver
        get_x_values: Function to get x-axis values from results
        get_times: Function to get solve times from results
        get_title: Function to generate plot title from group
        markers: Dictionary mapping solvers to marker styles
        save_path: Function to generate save path for a group
        save_figures: Whether to save figures to disk

    Returns:
        Dictionary mapping groups to figure objects
    """
    if save_figures:
        Path("figures").mkdir(exist_ok=True)

    figures = {}
    groups = get_groups(results)

    for group in groups:
        setup_plotting_style()
        fig, ax = plt.subplots()

        # Filter results for this group
        group_results = filter_group(results, group)
        solvers = get_solvers(group_results)

        # Plot solution times
        for solver in solvers:
            solver_results = filter_solver(group_results, solver)

            x_values = get_x_values(solver_results)
            times = get_times(solver_results)

            ax.plot(
                x_values,
                times,
                label=solver if solver == "Ours" else solver.upper(),
                marker=markers[solver],
                markersize=8,
                linestyle="-",
                markeredgewidth=1,
            )

        # Configure plot
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of scenarios")
        ax.set_ylabel("Solution time (seconds)")
        ax.set_title(get_title(group))

        ax.grid(True, which="major", linestyle="-", alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", alpha=0.2)
        ax.legend()

        plt.tight_layout()
        figures[group] = fig

        # Save figure if requested
        if save_figures and save_path:
            fig.savefig(save_path(Path("figures"), group))
            plt.show()
        else:
            plt.show()

    return figures


def plot_proj_benchmarks(results, save_figures=False):
    """Create plots comparing solver performance for projection benchmarks."""
    markers = {"Ours": "o", "MOSEK": "s", "CLARABEL": "^"}

    return create_benchmark_plot(
        results=results,
        get_groups=lambda r: sorted(set(x.tau for x in r)),
        filter_group=lambda r, tau: [x for x in r if x.tau == tau],
        get_solvers=lambda r: sorted(set(x.solver for x in r)),
        filter_solver=lambda r, s: sorted(
            [x for x in r if x.solver == s], key=lambda x: x.m
        ),
        get_x_values=lambda r: [x.m for x in r],
        get_times=lambda r: [x.avg_time for x in r],
        get_title=lambda tau: f"τ={tau}",
        markers=markers,
        save_path=lambda p, tau: p / f"proj_tau={tau}.pdf",
        save_figures=save_figures,
    )


def plot_cvqp_benchmarks(results, save_figures=False):
    """Create plot comparing solver performance for CVQP benchmarks."""
    markers = {"cvqp": "o", "mosek": "s", "clarabel": "^"}

    problem_name = results[0].problem.lower()

    return create_benchmark_plot(
        results=results,
        get_groups=lambda r: sorted(
            set(x.n_vars for x in r)
        ),  # Get all unique n_vars values
        filter_group=lambda r, n: [x for x in r if x.n_vars == n],  # Filter by n_vars
        get_solvers=lambda r: sorted(set(x.solver for x in r)),
        filter_solver=lambda r, s: sorted(
            [x for x in r if x.solver == s], key=lambda x: x.n_scenarios
        ),
        get_x_values=lambda r: [x.n_scenarios for x in r],
        get_times=lambda r: [x.avg_time for x in r],
        get_title=lambda n_vars: f"n_vars = {n_vars}",
        markers=markers,
        save_path=lambda p, n_vars: p / f"{problem_name}_n={n_vars}.pdf",
        save_figures=save_figures,
    )


# ========= Table Generation Data Classes =========


@dataclass
class TableConfig:
    """Configuration for table generation."""

    group_param: str  # Parameter to group by (e.g., 'tau' or 'n_vars')
    scenario_param: str  # Parameter for scenarios (e.g., 'm' or 'n_scenarios')
    solvers: list[str]  # List of solvers to include
    title_format: str  # Format string for table title
    get_group_value: Callable  # Function to get group value from result
    get_scenario_value: Callable  # Function to get scenario value from result
    get_solver_name: Callable = lambda x: x  # Function to format solver names


# ========= Table Formatting Functions =========


def format_time_with_std(time: float, std: float) -> str:
    """Format time and standard deviation with consistent spacing."""
    if isinstance(time, str) or np.isnan(time):
        return "N/A"
    return f"{time:.2e} ± {std:.2e}"


def format_success_rate(successes: int, total: int) -> str:
    """Format success rate with consistent spacing."""
    if isinstance(successes, str):
        return "N/A"
    return f"{successes}/{total}"


# ========= Table Generation Functions =========


def create_results_dataframe(
    results: list, group_value: float | int, config: TableConfig
) -> pd.DataFrame:
    """Convert results for a specific group value into a pandas DataFrame."""
    # Filter results for this group
    group_results = [r for r in results if config.get_group_value(r) == group_value]
    scenario_values = sorted(set(config.get_scenario_value(r) for r in group_results))

    data = []
    for scenario in scenario_values:
        row = {"scenarios": f"{scenario:.0e}"}

        # Collect data for each solver
        for solver in config.solvers:
            result = next(
                (
                    r
                    for r in group_results
                    if r.solver == solver and config.get_scenario_value(r) == scenario
                ),
                None,
            )
            if result:
                row[f"time_{solver}"] = format_time_with_std(
                    result.avg_time, result.std_time
                )
                row[f"success_{solver}"] = format_success_rate(
                    result.num_success, result.num_total
                )
            else:
                row[f"time_{solver}"] = "N/A"
                row[f"success_{solver}"] = "N/A"

        data.append(row)

    # Create DataFrame with scenario column
    df_data = {"Scenarios": [row["scenarios"] for row in data]}

    # Add solver times
    for solver in config.solvers:
        display_name = config.get_solver_name(solver)
        df_data[display_name] = [row[f"time_{solver}"] for row in data]

    # Add success rates
    for solver in config.solvers:
        display_name = config.get_solver_name(solver)
        df_data[f"success_{display_name}"] = [row[f"success_{solver}"] for row in data]

    # Create DataFrame
    df = pd.DataFrame(df_data)

    # Create MultiIndex for columns to group by category
    columns = pd.MultiIndex.from_tuples(
        [("Scenarios", "")]
        + [
            (f"Solve Times (seconds)", config.get_solver_name(solver))
            for solver in config.solvers
        ]
        + [
            ("Success Rate", config.get_solver_name(solver))
            for solver in config.solvers
        ]
    )

    # Reorganize DataFrame with proper column structure
    df.columns = columns

    return df


def display_benchmark_tables(results: list, config: TableConfig) -> None:
    """Display formatted tables using pandas with minimal styling."""
    group_values = sorted(set(config.get_group_value(r) for r in results))

    for group_value in group_values:
        print(config.title_format.format(group_value))

        # Create and style DataFrame
        df = create_results_dataframe(results, group_value, config)

        # Apply minimal styling
        styled_df = (
            df.style.set_properties(
                **{"text-align": "center", "padding": "8px", "border": "1px solid #666"}
            )
            .set_table_styles(
                [
                    # Headers
                    {
                        "selector": "th",
                        "props": [
                            ("text-align", "center"),
                            ("padding", "8px"),
                            ("border", "1px solid #666"),
                        ],
                    },
                    # Table borders
                    {"selector": "td", "props": [("border", "1px solid #666")]},
                ]
            )
            .hide(axis=0)
        )  # Hide index

        # Display the styled DataFrame
        display(styled_df)
        print()  # Add space between tables


# ========= Default Configurations =========

proj_config = TableConfig(
    group_param="tau",
    scenario_param="m",
    solvers=["Ours", "MOSEK", "CLARABEL"],
    title_format="Benchmark Results (τ = {:.1f})",
    get_group_value=lambda r: r.tau,
    get_scenario_value=lambda r: r.m,
)

cvqp_config = TableConfig(
    group_param="n_vars",
    scenario_param="n_scenarios",
    solvers=["cvqp", "mosek", "clarabel"],
    title_format="Benchmark Results (n_vars = {})",
    get_group_value=lambda r: r.n_vars,
    get_scenario_value=lambda r: r.n_scenarios,
    get_solver_name=str.upper,  # Convert solver names to uppercase
)
