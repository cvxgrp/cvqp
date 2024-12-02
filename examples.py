"""
Script to benchmark solvers on CVaR-constrained Quadratic Programs (CVQP).

This module implements three examples:
- Portfolio optimization
- Network traffic engineering
- Supply chain optimization

Each problem is formulated as a CVQP and solved using a custom ADMM solver
and state-of-the-art commercial (MOSEK) and open-source (Clarabel) solvers 
to benchmark performance.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pickle
import cvxpy as cp
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s PM: %(message)s',
    datefmt='%b %d %H:%M:%S'
)

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
    beta: list[float]   
    kappa: list[float]
    
    
@dataclass
class BenchmarkResults:
    """
    Store benchmark results for a specific problem configuration.
    
    Args:
        problem: Name of the problem being solved.
        solver: Name of the solver used.
        n_vars: Number of variables in the problem.
        n_scenarios: Number of scenarios in the problem.
        times: List of solve times for each instance.
    """
    problem: str
    solver: str
    n_vars: int
    n_scenarios: int
    times: list[float]
    
    @property
    def avg_time(self) -> float:
        return np.mean(self.times)
    
    @property
    def std_time(self) -> float:
        return np.std(self.times)
    
    @property
    def min_time(self) -> float:
        return np.min(self.times)
    
    @property
    def max_time(self) -> float:
        return np.max(self.times)


class CVQProblem(ABC):
    """Abstract base class for CVQP problems."""
    @abstractmethod
    def generate_instance(self, n_vars: int, n_scenarios: int, seed: int | None = None) -> CVQPParams:
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
        """Problem name for results identification."""
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
        beta1: float = 0.95,    
        beta2: float = 0.99,    
        kappa1: float = 0.15,    
        kappa2: float = 0.20   
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.nu = nu
        self.sigma = sigma
        self.beta = [beta1, beta2]
        self.kappa = [kappa1, kappa2]
        
    def generate_return_matrix(self, n_vars: int, n_scenarios: int, rng: np.random.Generator) -> np.ndarray:
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
        R[normal_idx] = rng.normal(
            self.nu, 1.0, 
            size=(normal_prob.sum(), n_vars)
        )
        
        # Generate returns for stress conditions
        stress_idx = normal_prob == 0
        R[stress_idx] = rng.normal(
            -self.nu, self.sigma, 
            size=(stress_idx.sum(), n_vars)
        )
        
        return R
    
    def generate_instance(self, n_vars: int, n_scenarios: int, seed: int | None = None) -> CVQPParams:
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
        u = np.array([1.0] + [float('inf')] * n_vars)
        
        return CVQPParams(
            P=P, q=q, A=A, B=B, l=l, u=u,
            beta=self.beta, kappa=self.kappa
        )
        
    @property
    def name(self) -> str:
        """Return the name identifier for this problem."""
        return "portfolio"


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
        K: int = 50,            
        sigma1: float = 0.1,    
        sigma2: float = 0.2   
    ):
        self.gamma = gamma
        self.beta = [beta]
        self.K = K
        self.sigma1 = sigma1
        self.sigma2 = sigma2
    
    def generate_path_structure(self, n_vars: int, rng: np.random.Generator) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Generate paths and their mapping to source-destination pairs.
        
        Args:
            n_vars: Total number of paths.
            rng: Random number generator.
            
        Returns:
            Tuple of demands array and list of constraint matrices.
        """
        # Generate demands
        demands = 1 + 0.5 * rng.normal(0, 1, self.K)
        
        # Generate base hop counts
        base_hops = 3 + np.floor(rng.exponential(scale=1/0.3, size=self.K)) + 1
        
        # Allocate paths to pairs (roughly equal distribution)
        paths_per_pair = n_vars // self.K
        remainder = n_vars % self.K
        
        # Create constraint matrices for each pair
        B_matrices = []
        current_path = 0
        
        for k in range(self.K):
            n_paths = paths_per_pair + (1 if k < remainder else 0)
            B_k = np.zeros(n_vars)
            B_k[current_path:current_path + n_paths] = 1
            B_matrices.append(B_k)
            current_path += n_paths
            
        return demands, B_matrices
    
    def generate_instance(self, n_vars: int, n_scenarios: int, seed: int | None = None) -> CVQPParams:
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
        demands, B_matrices = self.generate_path_structure(n_vars, rng)
        
        # Generate nominal delays
        base_delays = rng.normal(0, 1, n_vars)
        tau = base_delays * (1 + 0.1 * rng.normal(0, 1, n_vars))
        
        # Generate delay matrix
        D = np.zeros((n_scenarios, n_vars))
        for i in range(n_scenarios):
            eta = rng.normal(0, self.sigma2)
            epsilon = rng.normal(0, self.sigma1, n_vars)
            D[i] = tau * (1 + epsilon + eta)
        
        # CVQP parameters
        P = self.gamma * np.eye(n_vars)
        q = tau.copy()  # Use nominal delays as costs
        A = D
        B = np.vstack(B_matrices)
        l = demands
        u = demands
        kappa = [1.5 * np.max(tau)]  # kappa = 1.5 * max nominal delay
        
        return CVQPParams(
            P=P, q=q, A=A, B=B, l=l, u=u,
            beta=self.beta, kappa=kappa
        )
    
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
        theta: float = 1.0      
    ):
        self.lambda_ = lambda_
        self.K = K
        self.l = l
        self.u = u
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.beta = [beta]
        self.kappa = [kappa]
        self.v = v
        self.theta = theta
    
    def generate_instance(self, n_vars: int, n_scenarios: int, seed: int | None = None) -> CVQPParams:
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
            np.ones((1, n_vars)),              # sum(x) = 1
            np.eye(n_vars),                    # x >= l
            -np.eye(n_vars),                   # x <= u
        ]
        
        # Add regional constraints
        current_supplier = 0
        for k in range(self.K):
            n_suppliers = suppliers_per_region + (1 if k < remainder else 0)
            B_k = np.zeros(n_vars)
            B_k[current_supplier:current_supplier + n_suppliers] = 1
            B_blocks.append(B_k.reshape(1, -1))
            current_supplier += n_suppliers
        
        B = np.vstack(B_blocks)
        
        # Create bounds vectors
        l_bounds = [1.0]                         # sum(x) = 1
        l_bounds.extend([self.l] * n_vars)       # x >= l
        l_bounds.extend([-self.u] * n_vars)      # -x >= -u
        l_bounds.extend([-float('inf')] * self.K)  # regional constraints
        
        u_bounds = [1.0]                         # sum(x) = 1
        u_bounds.extend([self.u] * n_vars)       # x <= u
        u_bounds.extend([-self.l] * n_vars)      # -x <= -l
        u_bounds.extend([self.v] * self.K)       # regional constraints
        
        return CVQPParams(
            P=P, q=q, A=A, B=B,
            l=np.array(l_bounds),
            u=np.array(u_bounds),
            beta=self.beta,
            kappa=self.kappa
        )
    
    @property
    def name(self) -> str:
        """Return the name identifier for this problem."""
        return "supply_chain"


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
        solvers: list[str] = ["mosek", "clarabel", "admm"],
        base_seed: int = 42
    ):
        self.problems = problems
        self.n_instances = n_instances
        self.n_vars_list = n_vars_list
        self.n_scenarios_list = n_scenarios_list
        self.solvers = solvers
        self.base_seed = base_seed
        self.results: list[BenchmarkResults] = []

    def get_instance_seed(self, problem_name: str, solver: str, n_vars: int, n_scenarios: int, instance_idx: int) -> int:
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
    
    def solve_instance(self, params: CVQPParams, solver: str) -> float:
        """
        Solve a CVQP instance with specified solver.
        
        Args:
            params: Problem parameters.
            solver: Name of solver to use.
            
        Returns:
            Solution time in seconds.
        """
        if solver in ["mosek", "clarabel"]:
            return self.solve_cvxpy(params, solver)
        else:  # "admm"
            return self.solve_admm(params)
    
    def solve_cvxpy(self, params: CVQPParams, solver: str, verbose: bool = False) -> float:
        """
        Solve using CVXPY with specified solver.
        
        Args:
            params: Problem parameters.
            solver: Either 'mosek' or 'clarabel'.
            verbose: Whether to print solver output.
            
        Returns:
            Solution time in seconds.
        """
        # Map solver string to CVXPY solver constant
        solver_map = {
            'mosek': cp.MOSEK,
            'clarabel': cp.CLARABEL
        }
        if solver.lower() not in solver_map:
            raise ValueError(f"Unsupported solver: {solver}. Must be one of {list(solver_map.keys())}")
        
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
        for i in range(len(params.beta)):
            constraints.append(cp.cvar(params.A @ x, params.beta[i]) <= params.kappa[i])
        
        # Create and solve problem
        prob = cp.Problem(cp.Minimize(obj), constraints)
        
        # Time the solve
        prob.solve(solver=solver, verbose=verbose)
        solve_time = prob._solve_time
        
        if prob.status != 'optimal':
            raise RuntimeError(f"Problem failed to solve optimally. Status: {prob.status}")
        
        return solve_time
    
    def solve_admm(self, params: CVQPParams) -> float:
        """
        Solve using custom ADMM implementation.
        
        Args:
            params: Problem parameters.
            
        Returns:
            Solution time in seconds.
        """
        # Implementation placeholder
        pass
    
    def run_experiments(self):
        """Run all experiments and store results."""
        logging.info(f"Benchmarking solvers ({', '.join(self.solvers)}) on {len(self.problems)} CVQP problems")
        for problem in self.problems:
            for solver in self.solvers:
                for n_vars in self.n_vars_list:
                    for n_scenarios in self.n_scenarios_list:
                        solve_times = []
                        
                        for i in range(self.n_instances):
                            seed = self.get_instance_seed(
                                problem.name, solver, n_vars, n_scenarios, i
                            )
                            params = problem.generate_instance(n_vars, n_scenarios, seed=seed)
                            solve_time = self.solve_instance(params, solver)
                            solve_times.append(solve_time)
                        
                        avg_time = np.mean(solve_times)
                        std_time = np.std(solve_times)
                        logging.info(
                        f"problem={problem.name}, solver={solver}, n_vars={n_vars}, "
                        f"n_scenarios={n_scenarios}, solve_time={avg_time:.3f}s (±{std_time:.3f}s)"
)
                        self.results.append(BenchmarkResults(
                            problem=problem.name,
                            solver=solver,
                            n_vars=n_vars,
                            n_scenarios=n_scenarios,
                            times=solve_times
                        ))
        logging.info("Completed all experiments")
    
    def save_results(self, filename: str):
        """
        Save experiment results to a pickle file.
        
        Args:
            filename: Path where to save the results.
        """
        results_dict = {
            'base_seed': self.base_seed,
            'n_instances': self.n_instances,
            'n_vars_list': self.n_vars_list,
            'n_scenarios_list': self.n_scenarios_list,
            'results': self.results
        }
        with open(filename, 'wb') as f:
            pickle.dump(results_dict, f)
    
    def load_results(self, filename: str):
        """
        Load experiment results from a pickle file.
        
        Args:
            filename: Path to the results file.
        """
        with open(filename, 'rb') as f:
            results_dict = pickle.load(f)
            self.base_seed = results_dict['base_seed']
            self.n_instances = results_dict['n_instances']
            self.n_vars_list = results_dict['n_vars_list']
            self.n_scenarios_list = results_dict['n_scenarios_list']
            self.results = results_dict['results']

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
        n_instances=3,  # Small number for testing
        n_vars_list=[100, 1000],
        n_scenarios_list=[1000],
        solvers=["clarabel", "mosek"]
    )

    # Run and save
    runner.run_experiments()
    runner.save_results("data/portfolio_results.pkl")

if __name__ == "__main__":
    main()