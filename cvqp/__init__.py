"""CVQP: solver for CVaR-constrained quadratic programs."""

from ._types import Result, Settings, History
from ._solver import solve
from ._projection import proj_sum_largest, proj_cvar

__all__ = ["solve", "proj_cvar", "proj_sum_largest", "Result", "Settings", "History"]
