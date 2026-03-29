"""Type definitions for the CVQP solver."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Settings:
    """ADMM solver settings."""

    max_iter: int = 100_000
    abstol: float = 1e-4
    reltol: float = 1e-3
    time_limit: float = 7200.0
    rho: float = 1e-2
    adaptive_rho: bool = True
    # ADMM internals (rarely changed)
    alpha_relax: float = 1.7
    mu: float = 10.0
    tau: float = 2.0
    check_interval: int = 50


@dataclass
class History:
    """Convergence history, recorded every check_interval iterations."""

    r_norm: list[float] = field(default_factory=list)
    s_norm: list[float] = field(default_factory=list)
    eps_pri: list[float] = field(default_factory=list)
    eps_dual: list[float] = field(default_factory=list)
    objective: list[float] = field(default_factory=list)
    rho: list[float] = field(default_factory=list)


@dataclass
class Result:
    """Solver output."""

    x: np.ndarray
    value: float
    status: str
    iterations: int
    time: float
    history: History
