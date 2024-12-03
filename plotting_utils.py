"""
Utility functions for plotting benchmark results.
"""
import os
import matplotlib.pyplot as plt

def setup_plotting_style():
    """Configure matplotlib plotting style for publication quality figures."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (10, 6),
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

def plot_solver_comparison(results, problem_name, n_vars):
    """
    Create a log-log plot comparing solver performance for a specific problem and variable count.
    """
    # Filter results for the specific problem and n_vars
    relevant_results = [r for r in results['results'] 
                       if r.problem == problem_name and r.n_vars == n_vars]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot for each solver
    colors = {'mosek': '#1f77b4', 'clarabel': '#d62728', 'admm': '#2ca02c'}
    markers = {'mosek': 'o', 'clarabel': 's', 'admm': '^'}
    
    for solver in ['mosek', 'clarabel', 'admm']:
        solver_results = [r for r in relevant_results if r.solver == solver]
        
        # Extract data points
        scenarios = [r.n_scenarios for r in solver_results]
        times = [r.avg_time for r in solver_results]
        std_times = [r.std_time for r in solver_results]
        
        # Plot with error bars
        ax.errorbar(scenarios, times, yerr=std_times, 
                   label=solver.upper(),
                   color=colors[solver],
                   marker=markers[solver],
                   markersize=8,
                   capsize=5,
                   linestyle='-')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of scenarios')
    ax.set_ylabel('Solution time (seconds)')
    ax.set_title(f'{problem_name.title()} example (n={n_vars})')
    ax.grid(True, which='major', linestyle='-', alpha=0.2)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    return fig

def get_specific_admm_run(results, problem_name, n_vars, n_scenarios, run_index=0):
    """
    Get ADMM results for a specific problem configuration.
    """
    relevant_results = [r for r in results['results'] 
                       if r.problem == problem_name 
                       and r.solver == 'admm'
                       and r.n_vars == n_vars 
                       and r.n_scenarios == n_scenarios]
    
    if not relevant_results:
        raise ValueError("No matching results found")
        
    return relevant_results[0].admm_results[run_index]

def plot_admm_convergence(admm_results):
    """
    Create a 4-panel plot showing ADMM convergence metrics.
    """
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))
    
    # Extract convergence history
    iters = [i * 100 for i in range(len(admm_results.objval))]
    
    # Objective value
    axs[0].plot(iters, admm_results.objval, 'k', linewidth=2)
    axs[0].set_ylabel("Objective")
    axs[0].set_xlabel('Iteration')
    axs[0].grid(True, which='both', linestyle='--', alpha=0.3)
    
    # Primal residual
    axs[1].semilogy(iters, admm_results.r_norm, 'k', linewidth=2, 
                    label="Primal residual norm")
    axs[1].semilogy(iters, admm_results.eps_pri, 'k--', 
                    linewidth=2, label="Primal tolerance")
    axs[1].set_ylabel(r'Primal residual norm')
    axs[1].set_xlabel('Iteration')
    axs[1].grid(True, which='both', linestyle='--', alpha=0.3)
    axs[1].legend(frameon=True, fancybox=True)
    
    # Dual residual
    axs[2].semilogy(iters, admm_results.s_norm, 'k', linewidth=2,
                    label="Dual residual norm")
    axs[2].semilogy(iters, admm_results.eps_dual, 'k--',
                    linewidth=2, label="Dual tolerance")
    axs[2].set_ylabel(r'Dual residual norm')
    axs[2].set_xlabel('Iteration')
    axs[2].grid(True, which='both', linestyle='--', alpha=0.3)
    axs[2].legend(frameon=True, fancybox=True)
    
    # Penalty parameter
    axs[3].semilogy(iters, admm_results.rho, 'k', linewidth=2)
    axs[3].set_ylabel(r'$\rho$')
    axs[3].set_xlabel('Iteration')
    axs[3].grid(True, which='both', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig

def save_fig(fig, name, folder='figures'):
    """
    Save a figure in publication-ready quality.
    
    Args:
        fig: matplotlib figure object
        name: name of the file without extension
        folder: output directory (default: 'figures')
    """
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f'{name}.pdf')
    fig.savefig(filepath,
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.1)