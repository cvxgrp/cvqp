"""
Utility functions for plotting benchmark results from multiple pickle files.
"""
import os
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

def setup_plotting_style():
    """Configure matplotlib plotting style for publication quality figures."""
    plt.rcParams.update({
        # Font sizes
        'font.size': 14,          # Slightly larger base font
        'axes.labelsize': 14,     # Clear axis labels
        'axes.titlesize': 16,     # Prominent titles
        'xtick.labelsize': 12,    # Readable tick labels
        'ytick.labelsize': 12,
        'legend.fontsize': 12,    # Clear legend text
        
        # Figure properties
        'figure.figsize': (8, 6),
        'lines.linewidth': 2,
        'lines.markersize': 8,
        
        # Grid properties
        'axes.grid': True,
        'grid.alpha': 0.35,       # Slightly more visible major grid
        'grid.linestyle': ':',    # Dotted style
    })

def load_all_results(data_dir='data'):
    """
    Load and combine results from all pickle files in the data directory.
    
    Args:
        data_dir: Directory containing the pickle files
        
    Returns:
        List of ProjectionResults objects
    """
    all_results = []
    data_path = Path(data_dir)
    
    # Load each pickle file
    for pkl_file in data_path.glob('proj_m=*.pkl'):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            all_results.extend(data['results'])
            
    return all_results

def plot_proj_benchmarks(results, save_figures=False):
    """
    Create plots comparing solver performance for sum-k-largest projection benchmarks.
    
    Args:
        results: List of ProjectionResults objects
        save_figures: If True, saves figures as PDFs (default: False)
        
    Returns:
        Dictionary mapping tau values to figure objects
    """
    markers = {
        'Ours': 'o',           
        'MOSEK': 's',     
        'CLARABEL': '^'     
    }
    
    # Create figures directory if saving is requested
    if save_figures:
        Path('figures').mkdir(exist_ok=True)
    
    figures = {}
    tau_values = sorted(set(r.tau for r in results))
    
    for tau in tau_values:
        setup_plotting_style()  # Ensure fresh style for each plot
        fig, ax = plt.subplots()
        
        # Filter results for this tau value
        tau_results = [r for r in results if r.tau == tau]
        solvers = sorted(set(r.solver for r in tau_results))
        
        # Plot solution times
        for solver in solvers:
            solver_results = [r for r in tau_results if r.solver == solver]
            solver_results.sort(key=lambda x: x.m)
            
            ms = [r.m for r in solver_results]
            times = [r.avg_time for r in solver_results]
            
            ax.plot(ms, times,
                   label=solver,
                   marker=markers[solver],
                   markersize=8,
                   linestyle='-',
                   markeredgewidth=1)
        
        # Configure plot
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of scenarios')
        ax.set_ylabel('Solution time (seconds)')
        ax.set_title(f'τ={tau}')
        
        # Enhanced grid
        ax.grid(True, which='major', linestyle='-', alpha=0.35)
        ax.grid(True, which='minor', linestyle=':', alpha=0.2)
        
        # Enhanced legend
        ax.legend()
        
        plt.tight_layout()
        figures[tau] = fig
        
        # Save figure if requested
        if save_figures:
            save_path = Path('figures') / f'proj_benchmark_tau_{tau}.pdf'
            fig.savefig(save_path)
            plt.show()  # Display in notebook even when saving
        else:
            plt.show()  # Always display in notebook
    
    return figures

"""
Functions for displaying benchmark results showing all solvers side by side.
"""
import pandas as pd
import numpy as np
from typing import List
from IPython.display import display, HTML

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

def create_results_dataframe(results: List, tau: float) -> pd.DataFrame:
    """Convert results for a specific tau value into a pandas DataFrame."""
    tau_results = [r for r in results if r.tau == tau]
    m_values = sorted(set(r.m for r in tau_results))
    solvers = ['Ours', 'MOSEK', 'CLARABEL']
    
    data = []
    for m in m_values:
        row = {'m': f"{m:.0e}"}
        
        # Collect data for each solver
        for solver in solvers:
            result = next((r for r in tau_results if r.solver == solver and r.m == m), None)
            if result:
                row[f'time_{solver}'] = format_time_with_std(result.avg_time, result.std_time)
                row[f'success_{solver}'] = format_success_rate(result.num_success, result.num_total)
            else:
                row[f'time_{solver}'] = "N/A"
                row[f'success_{solver}'] = "N/A"
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create MultiIndex for columns to group by category then solver
    columns = pd.MultiIndex.from_tuples([
        ('Scenarios', 'm'),
        ('Solve Times (seconds)', 'Ours'),
        ('Solve Times (seconds)', 'MOSEK'),
        ('Solve Times (seconds)', 'CLARABEL'),
        ('Success Rate', 'Ours'),
        ('Success Rate', 'MOSEK'),
        ('Success Rate', 'CLARABEL'),
    ])
    
    # Reorganize DataFrame with proper column structure
    df_display = pd.DataFrame({
        ('Scenarios', 'm'): df['m'],
        ('Solve Times (seconds)', 'Ours'): df['time_Ours'],
        ('Solve Times (seconds)', 'MOSEK'): df['time_MOSEK'],
        ('Solve Times (seconds)', 'CLARABEL'): df['time_CLARABEL'],
        ('Success Rate', 'Ours'): df['success_Ours'],
        ('Success Rate', 'MOSEK'): df['success_MOSEK'],
        ('Success Rate', 'CLARABEL'): df['success_CLARABEL'],
    })
    
    return df_display

def display_benchmark_tables(results: List) -> None:
    """Display formatted tables using pandas with minimal styling."""
    tau_values = sorted(set(r.tau for r in results))
    
    for tau in tau_values:
        print(f"Benchmark Results (τ = {tau:.1f})")
        
        # Create and style DataFrame
        df = create_results_dataframe(results, tau)
        
        # Apply minimal styling
        styled_df = df.style\
            .set_properties(**{
                'text-align': 'center',
                'padding': '8px',
                'border': '1px solid #666'
            })\
            .set_table_styles([
                # Headers
                {'selector': 'th', 'props': [
                    ('text-align', 'center'),
                    ('padding', '8px'),
                    ('border', '1px solid #666')
                ]},
                # Table borders
                {'selector': 'td', 'props': [
                    ('border', '1px solid #666')
                ]},
            ])\
            .hide(axis=0)  # Hide index
        
        # Display the styled DataFrame
        display(styled_df)
        print()  # Add space between tables