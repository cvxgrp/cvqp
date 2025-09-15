"""
CVQP: A Python-embedded solver for CVaR-constrained quadratic programs.
"""

__version__ = "0.2.1"

from .types import CVQPParams, CVQPConfig, CVQPResults
from .admm import CVQP, solve_cvqp
from .projection import proj_sum_largest, proj_cvar

__all__ = ["CVQP", "CVQPParams", "CVQPResults", "CVQPConfig", "solve_cvqp", "proj_sum_largest", "proj_cvar"]
