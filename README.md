# CVQP

[![PyPI version](https://img.shields.io/pypi/v/cvqp.svg)](https://pypi.org/project/cvqp/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

CVQP solves large-scale CVaR-constrained quadratic programs with millions of scenarios, where standard solvers fail or are prohibitively slow. It also provides an efficient algorithm for Euclidean projection onto CVaR constraints (equivalently, sum-of-k-largest constraints). For details, see our [paper](https://web.stanford.edu/~boyd/papers/cvar_qp.html).

## Installation

From PyPI:

```bash
pip install cvqp
```

From source:

```bash
git clone https://github.com/cvxgrp/cvqp.git
cd cvqp
poetry install
poetry run pip install -e .
```

## Usage

### Solve a CVaR-constrained quadratic program

CVQP solves problems of the form

$$
\begin{array}{ll}
\text{minimize} & \frac{1}{2}x^TPx + q^Tx \\
\text{subject to} & \phi_\beta(Ax) \leq \kappa \\
                  & l \leq Bx \leq u,
\end{array}
$$

where $\phi_\beta$ is the Conditional Value-at-Risk (CVaR) at confidence level $\beta \in [0,1)$ of the losses $Ax \in \mathbf{R}^m$. CVaR is the expected value of the worst $(1-\beta)$ fraction of losses. For example, with $\beta = 0.9$ and $m = 1000$ scenarios, CVaR is the average of the 100 largest losses.

Below we show how to formulate and solve a CVQP arising from a portfolio construction problem:

```python
import numpy as np
from cvqp import solve_cvqp

# Generate data
np.random.seed(1)
n_assets = 100
n_scenarios = 10000
R = np.random.randn(n_scenarios, n_assets) * 0.1 + 0.01
mu = np.mean(R, axis=0)
Sigma = np.cov(R.T)

# Build problem matrices
P = Sigma
q = -mu
A = -R
B = np.vstack([np.ones((1, n_assets)), np.eye(n_assets)])
l = np.concatenate([[1.0], np.zeros(n_assets)])
u = np.concatenate([[1.0], np.ones(n_assets)])

# Solve portfolio optimization with CVaR constraint
beta = 0.95
kappa = 0.025
results = solve_cvqp(P=P, q=q, A=A, B=B, l=l, u=u,
                     beta=beta, kappa=kappa, verbose=True)

# Check CVaR at solution
losses = A @ results.x
k = int((1 - beta) * n_scenarios)
cvar = np.mean(np.sort(losses)[-k:])

print(f"CVaR at β={beta}: {cvar:.6f} (limit: {kappa})")
```

### Project onto CVaR / sum-of-k-largest constraints

Projection onto a CVaR constraint finds the closest vector to a given vector $v$ that satisfies the CVaR limit. This problem has the form

$$
\begin{array}{ll}
\text{minimize} & \|v - x\|_2^2 \\
\text{subject to} & \phi_\beta(x) \leq \kappa.
\end{array}
$$

This is equivalent to projection onto a sum-of-k-largest constraint:

$$
\begin{array}{ll}
\text{minimize} & \|v - x\|_2^2 \\
\text{subject to} & f_k(x) \leq d,
\end{array}
$$

where $f_k(x) = \sum_{i=1}^k x_{[i]}$ (sum of k largest components), $k = \lceil(1-\beta)m\rceil$, and $d = \kappa k$.

```python
from cvqp import proj_cvar, proj_sum_largest
import numpy as np

# Project a vector onto CVaR constraint
v = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
beta = 0.75
kappa = 5.0

# Both formulations give the same result
x_cvar = proj_cvar(v, beta, kappa)

k = int((1 - beta) * len(v))
d = kappa * k
x_sum = proj_sum_largest(v, k, d)

# Verify constraints are satisfied
cvar_original = np.mean(np.sort(v)[-k:])
cvar_projected = np.mean(np.sort(x_cvar)[-k:])

print(f"Original:         {v}")
print(f"Projected:        {x_cvar}")
print(f"CVaR: {cvar_original:.2f} → {cvar_projected:.2f} (limit: {kappa})")
print(f"Equivalent to sum-k-largest: {np.allclose(x_cvar, x_sum)}")
```

## Benchmarks

See `benchmarks/` for benchmark results against MOSEK and Clarabel on a suite of large-scale problems.

## Citation

If you use CVQP in your research, please cite:

```bibtex
@misc{cvqp2025,
  title={An Operator Splitting Method for Large-Scale {CVaR}-Constrained Quadratic Programs},
  author={Luxenberg, Eric and P\'erez-Pi\~neiro, David and Diamond, Steven and Boyd, Stephen},
  year={2025},
  eprint={2504.10814},
  archivePrefix={arXiv},
  primaryClass={math.OC},
  url={https://arxiv.org/abs/2504.10814}
}
```