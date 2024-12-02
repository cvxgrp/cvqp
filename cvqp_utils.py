"""
Common data structures for formulating CVQP problems. 
"""

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