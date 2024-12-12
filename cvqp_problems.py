"""
Classes to define CVQP problems and generate examples.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class CVQPParams:
    """
    Parameters defining a CVQP instance.

    Args:
        P: Quadratic cost matrix in objective.
        q: Linear cost vector in objective.
        A: Matrix for CVaR constraints.
        B: Linear constraint matrix.
        l: Lower bounds for Bx.
        u: Upper bounds for Bx.
        beta: List of probability levels for CVaR constraints.
        kappa: List of CVaR limits for CVaR constraints.
    """

    P: np.ndarray
    q: np.ndarray
    A: np.ndarray
    B: np.ndarray
    l: np.ndarray
    u: np.ndarray
    beta: float
    kappa: float


class CVQProblem(ABC):
    """Abstract base class for CVQP problems."""

    @abstractmethod
    def generate_instance(
        self, n_vars: int, n_scenarios: int, seed: int | None = None
    ) -> CVQPParams:
        """
        Generate a problem instance with given dimensions.

        Args:
            n_vars: Number of decision variables.
            n_scenarios: Number of scenarios.
            seed: Random seed for reproducibility.

        Returns:
            CVQPParams instance containing the generated problem parameters.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Problem name."""
        pass


class PortfolioOptimization(CVQProblem):
    """
    Portfolio optimization problem.

    This class implements a portfolio optimization problem where returns are modeled
    using a two-component Gaussian mixture to capture both normal and stress market
    conditions. The objective balances expected return against variance risk, with
    additional CVaR constraints on worst-case losses.

    Args:
        alpha: Probability of normal market conditions.
        gamma: Risk aversion parameter for variance term.
        nu: Mean return in normal market conditions.
        sigma: Volatility scaling factor for stress periods.
        beta: List containing two CVaR probability levels.
        kappa: List containing two CVaR thresholds.
    """

    def __init__(
        self,
        alpha: float = 0.9,
        gamma: float = 0.05,
        nu: float = 0.1,
        sigma: float = 0.2,
        beta: float = 0.95,
        kappa: float = 0.5,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.nu = nu
        self.sigma = sigma
        self.beta = beta
        self.kappa = kappa

    def generate_return_matrix(
        self, n_vars: int, n_scenarios: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Generate returns using a two-component Gaussian mixture model.

        Args:
            n_vars: Number of assets.
            n_scenarios: Number of return scenarios.
            rng: Random number generator.

        Returns:
            Matrix of asset returns across scenarios.
        """
        # Normal market conditions
        normal_prob = rng.binomial(1, self.alpha, n_scenarios)
        R = np.zeros((n_scenarios, n_vars))

        # Generate returns for normal conditions
        normal_idx = normal_prob == 1
        R[normal_idx] = rng.normal(self.nu, 1.0, size=(normal_prob.sum(), n_vars))

        # Generate returns for stress conditions
        stress_idx = normal_prob == 0
        R[stress_idx] = rng.normal(
            -self.nu, self.sigma, size=(stress_idx.sum(), n_vars)
        )

        return R

    def generate_instance(
        self, n_vars: int, n_scenarios: int, seed: int | None = None
    ) -> CVQPParams:
        """
        Generate a problem instance.

        Args:
            n_vars: Number of assets.
            n_scenarios: Number of return scenarios.
            seed: Random seed for reproducibility.

        Returns:
            CVQPParams instance containing the generated problem parameters.
        """
        # Generate return matrix
        rng = np.random.default_rng(seed)
        R = self.generate_return_matrix(n_vars, n_scenarios, rng)

        # Compute mean returns
        mu = R.mean(axis=0)

        # Compute covariance matrix
        R_centered = R - mu
        Sigma = (R_centered.T @ R_centered) / n_scenarios

        # CVQP parameters
        P = self.gamma * Sigma
        q = -mu
        A = -R

        # Linear constraints: sum(x) = 1, x >= 0
        B = np.vstack([np.ones(n_vars), np.eye(n_vars)])
        l = np.array([1.0] + [0.0] * n_vars)
        u = np.array([1.0] + [float("inf")] * n_vars)

        return CVQPParams(
            P=P, q=q, A=A, B=B, l=l, u=u, beta=self.beta, kappa=self.kappa
        )

    @property
    def name(self) -> str:
        """Return the name identifier for this problem."""
        return "portfolio"


class PortfolioOptimization2(CVQProblem):
    """
    Portfolio optimization problem with factor model structure.

    This class implements a portfolio optimization problem where returns are modeled
    using a factor structure, with k common factors driving returns across assets plus
    asset-specific risk. The objective balances expected return against variance risk,
    with additional CVaR constraints on worst-case losses.

    The number of factors is fixed independent of the number of assets, reflecting that
    the same underlying market factors drive returns regardless of portfolio size.

    Args:
        gamma: Risk aversion parameter for variance term.
        beta: CVaR probability level.
        kappa: CVaR threshold.
        k_factors: Number of factors in the return model.
        alpha: Probability of normal market conditions.
        nu: Factor risk premium.
        sigma: Volatility scaling in stress periods.
        mu_scale: Scale of expected returns.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        beta: float = 0.95,
        kappa: float = 0.2,
        k_factors: int = 15,
        alpha: float = 0.9,
        nu: float = 0.02,        
        sigma: float = 4.0,    
        mu_scale: float = 0.02 
    ):
        self.gamma = gamma
        self.beta = beta
        self.kappa = kappa
        self.k_factors = k_factors
        self.alpha = alpha
        self.nu = nu
        self.sigma = sigma
        self.mu_scale = mu_scale

    def generate_factor_model(
        self, n_vars: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate factor model parameters.

        Args:
            n_vars: Number of assets.
            rng: Random number generator.

        Returns:
            Tuple containing:
                - Factor loading matrix F
                - Diagonal asset-specific risk matrix D
                - Expected returns vector mu
        """
        # Generate factor loadings
        F = rng.normal(0, 1, size=(n_vars, self.k_factors))
        
        # Generate asset-specific risks
        D = np.diag(rng.uniform(0, np.sqrt(self.k_factors), size=n_vars))
        
        # Generate expected returns with positive risk premium
        mu = rng.normal(self.mu_scale, 1, size=n_vars)
        
        return F, D, mu

    def generate_return_matrix(
        self,
        n_scenarios: int,
        F: np.ndarray,
        D: np.ndarray,
        mu: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate return scenarios using the factor model structure.

        Args:
            n_scenarios: Number of scenarios.
            F: Factor loading matrix.
            D: Diagonal asset-specific risk matrix.
            mu: Expected returns vector.
            rng: Random number generator.

        Returns:
            Matrix of asset returns across scenarios.
        """
        # Generate market regime indicators
        normal_regime = rng.binomial(1, self.alpha, n_scenarios)
        
        # Initialize factor returns
        Z = np.zeros((n_scenarios, self.k_factors))
        
        # Generate factor returns for normal regime
        normal_idx = normal_regime == 1
        Z[normal_idx] = rng.normal(
            self.nu, 1.0, size=(normal_regime.sum(), self.k_factors)
        )
        
        # Generate factor returns for stress regime
        stress_idx = normal_regime == 0
        Z[stress_idx] = rng.normal(
            -10*self.nu,  # More negative mean in stress
            self.sigma,  # Higher volatility remains
            size=(stress_idx.sum(), self.k_factors)
        )
        
        # Generate asset-specific return scenarios
        E = rng.normal(0, 1, size=(n_scenarios, F.shape[0])) * np.sqrt(np.diag(D))
        
        # Combine systematic and idiosyncratic returns
        R = Z @ F.T + E + np.ones(n_scenarios)[:, None] @ mu[None, :]
        
        return R

    def generate_instance(
        self, n_vars: int, n_scenarios: int, seed: int | None = None
    ) -> CVQPParams:
        """
        Generate a problem instance.

        Args:
            n_vars: Number of assets.
            n_scenarios: Number of scenarios.
            seed: Random seed for reproducibility.

        Returns:
            CVQPParams instance containing the generated problem parameters.
        """
        rng = np.random.default_rng(seed)
        
        # Generate factor model parameters
        F, D, mu = self.generate_factor_model(n_vars, rng)
        
        # Generate return scenarios
        R = self.generate_return_matrix(n_scenarios, F, D, mu, rng)
        
        # Compute covariance matrix
        Sigma = F @ F.T + D
        
        # CVQP parameters
        P = self.gamma * Sigma
        q = -mu
        A = -R
        
        # Linear constraints: sum(x) = 1, x >= 0
        B = np.vstack([np.ones(n_vars), np.eye(n_vars)])
        l = np.array([1.0] + [0.0] * n_vars)
        u = np.array([1.0] + [float("inf")] * n_vars)
        
        return CVQPParams(
            P=P, q=q, A=A, B=B, l=l, u=u, beta=self.beta, kappa=self.kappa
        )

    @property
    def name(self) -> str:
        """Return the name identifier for this problem."""
        return "portfolio2"


class NetworkTraffic(CVQProblem):
    """
    Network traffic engineering problem.

    This class implements a traffic engineering problem where path delays are modeled
    with both path-specific and scenario-wide variations. The objective balances
    nominal routing cost against traffic concentration, with a CVaR constraint on
    worst-case delays.

    Args:
        gamma: Regularization parameter for traffic concentration.
        beta: CVaR probability level.
        K: Number of source-destination pairs.
        sigma1: Standard deviation for path-specific variations.
        sigma2: Standard deviation for scenario-wide variations.
    """

    def __init__(
        self,
        gamma: float = 0.1,
        beta: float = 0.95,
        K: int = 10,
        sigma1: float = 0.05,
        sigma2: float = 0.1,
    ):
        self.gamma = gamma
        self.beta = beta
        self.K = K
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def generate_path_structure(
        self, n_vars: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """
        Generate paths and their mapping to source-destination pairs.

        Args:
            n_vars: Total number of paths.
            rng: Random number generator.

        Returns:
            Tuple containing:
                - demands array
                - list of constraint matrices
                - base hop counts array
        """
        # Generate demands
        demands = 1 + 0.5 * rng.normal(0, 1, self.K)

        # Generate base hop counts
        base_hops = 3 + np.floor(rng.exponential(scale=1 / 0.3, size=self.K)) + 1

        # Create constraint matrices
        B_matrices = [np.eye(n_vars)]  # For x >= 0 constraints

        # Allocate paths to pairs (roughly equal distribution)
        paths_per_pair = n_vars // self.K
        remainder = n_vars % self.K
        current_path = 0

        # Add flow conservation constraints
        for k in range(self.K):
            n_paths = paths_per_pair + (1 if k < remainder else 0)
            B_k = np.zeros(n_vars)
            B_k[current_path : current_path + n_paths] = 1
            B_matrices.append(B_k)
            current_path += n_paths

        return demands, B_matrices, base_hops

    def generate_instance(
        self, n_vars: int, n_scenarios: int, seed: int | None = None
    ) -> CVQPParams:
        """
        Generate a problem instance.

        Args:
            n_vars: Number of paths.
            n_scenarios: Number of delay scenarios.
            seed: Random seed for reproducibility.

        Returns:
            CVQPParams instance containing the generated problem parameters.
        """
        rng = np.random.default_rng(seed)
        demands, B_matrices, base_hops = self.generate_path_structure(n_vars, rng)

        # Generate hop counts for each path
        current_path = 0
        hop_counts = np.zeros(n_vars)
        paths_per_pair = n_vars // self.K
        remainder = n_vars % self.K

        for k in range(self.K):
            n_paths = paths_per_pair + (1 if k < remainder else 0)
            # Use exp() to ensure variations are positive
            path_variation = np.exp(0.2 * rng.normal(0, 1, n_paths))
            hop_counts[current_path : current_path + n_paths] = np.ceil(
                base_hops[k] * path_variation
            )
            current_path += n_paths

        # Generate nominal delays based on hop counts
        # Use exp() to ensure delay multipliers are positive
        tau = hop_counts * np.exp(0.1 * rng.normal(0, 1, n_vars))

        # Generate delay matrix
        D = np.zeros((n_scenarios, n_vars))
        for i in range(n_scenarios):
            eta = rng.normal(0, self.sigma2)
            epsilon = rng.normal(0, self.sigma1, n_vars)
            # Use exp() for perturbations to ensure positive delays
            D[i] = tau * np.exp(epsilon + eta)

        # CVQP parameters
        P = self.gamma * np.eye(n_vars)
        q = tau.copy()  # Use nominal delays as costs
        A = D
        B = np.vstack(B_matrices)
        l = np.concatenate([np.zeros(n_vars), demands])  # x >= 0  # flow conservation
        u = np.concatenate(
            [
                np.inf * np.ones(n_vars),  # x <= inf (non-negativity)
                np.inf * np.ones(self.K),  # Σx_j <= inf (flow conservation)
            ]
        )
        kappa = 4.0 * np.max(tau)  # kappa = 4 * max nominal delay

        return CVQPParams(P=P, q=q, A=A, B=B, l=l, u=u, beta=self.beta, kappa=kappa)

    @property
    def name(self) -> str:
        """Return the name identifier for this problem."""
        return "network"


class SupplyChain(CVQProblem):
    """
    Supply chain contract selection problem.

    This class implements a supply chain problem where supplier costs are modeled
    with both regional and supplier-specific disruptions. The objective balances
    expected cost against cost variance, with constraints on regional allocations
    and a CVaR constraint on worst-case costs.

    Args:
        lambda_: Risk aversion parameter for cost variance.
        K: Number of geographical regions.
        l: Minimum allocation per supplier.
        u: Maximum allocation per supplier.
        sigma_r: Standard deviation for regional disruptions.
        sigma_s: Standard deviation for supplier-specific disruptions.
        beta: CVaR probability level.
        kappa: CVaR threshold.
        v: Maximum allocation per region.
        theta: Regional sensitivity to disruptions.
    """

    def __init__(
        self,
        lambda_: float = 0.1,
        K: int = 10,
        l: float = 0.0,
        u: float = 0.3,
        sigma_r: float = 0.3,
        sigma_s: float = 0.1,
        beta: float = 0.95,
        kappa: float = 1.5,
        v: float = 0.4,
        theta: float = 1.0,
    ):
        self.lambda_ = lambda_
        self.K = K
        self.l = l
        self.u = u
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.beta = beta
        self.kappa = kappa
        self.v = v
        self.theta = theta

    def generate_instance(
        self, n_vars: int, n_scenarios: int, seed: int | None = None
    ) -> CVQPParams:
        """
        Generate a problem instance.

        Args:
            n_vars: Number of suppliers.
            n_scenarios: Number of cost scenarios.
            seed: Random seed for reproducibility.

        Returns:
            CVQPParams instance containing the generated problem parameters.
        """
        # Assign suppliers to regions (roughly equal distribution)
        rng = np.random.default_rng(seed)
        suppliers_per_region = n_vars // self.K
        remainder = n_vars % self.K

        # Generate nominal costs
        c = 1 + 0.2 * rng.normal(0, 1, n_vars)

        # Generate cost matrix
        C = np.zeros((n_scenarios, n_vars))
        current_supplier = 0

        for k in range(self.K):
            n_suppliers = suppliers_per_region + (1 if k < remainder else 0)
            # Regional disruptions
            r_k = rng.normal(0, self.sigma_r, n_scenarios)

            for i in range(n_scenarios):
                # Supplier-specific disruptions
                zeta = rng.normal(0, self.sigma_s, n_suppliers)
                suppliers = slice(current_supplier, current_supplier + n_suppliers)
                C[i, suppliers] = c[suppliers] * (1 + self.theta * r_k[i] + zeta)

            current_supplier += n_suppliers

        # Compute covariance matrix
        C_centered = C - C.mean(axis=0)
        Sigma = (C_centered.T @ C_centered) / n_scenarios

        # CVQP parameters
        P = self.lambda_ * Sigma
        q = c
        A = C

        # Create B matrix for all constraints
        B_blocks = [
            np.ones((1, n_vars)),  # sum(x) = 1
            np.eye(n_vars),  # x >= l
            -np.eye(n_vars),  # x <= u
        ]

        # Add regional constraints
        current_supplier = 0
        for k in range(self.K):
            n_suppliers = suppliers_per_region + (1 if k < remainder else 0)
            B_k = np.zeros(n_vars)
            B_k[current_supplier : current_supplier + n_suppliers] = 1
            B_blocks.append(B_k.reshape(1, -1))
            current_supplier += n_suppliers

        B = np.vstack(B_blocks)

        # Create bounds vectors
        l_bounds = [1.0]  # sum(x) = 1
        l_bounds.extend([self.l] * n_vars)  # x >= l
        l_bounds.extend([-self.u] * n_vars)  # -x >= -u
        l_bounds.extend([-float("inf")] * self.K)  # regional constraints

        u_bounds = [1.0]  # sum(x) = 1
        u_bounds.extend([self.u] * n_vars)  # x <= u
        u_bounds.extend([-self.l] * n_vars)  # -x <= -l
        u_bounds.extend([self.v] * self.K)  # regional constraints

        return CVQPParams(
            P=P,
            q=q,
            A=A,
            B=B,
            l=np.array(l_bounds),
            u=np.array(u_bounds),
            beta=self.beta,
            kappa=self.kappa,
        )

    @property
    def name(self) -> str:
        """Return the name identifier for this problem."""
        return "supply_chain"
