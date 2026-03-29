# CVQP

[![PyPI version](https://img.shields.io/pypi/v/cvqp.svg)](https://pypi.org/project/cvqp/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

CVQP is a Python solver for CVaR-constrained quadratic programs. It also provides an efficient projection onto CVaR constraints. Both scale to millions of scenarios. For details, see our [paper](https://web.stanford.edu/~boyd/papers/cvar_qp.html).

## Installation

```bash
pip install cvqp
```

## Background

The sample Conditional Value-at-Risk (CVaR) at level $\beta \in (0,1)$ of a vector $z \in \mathbf{R}^m$ is

$$
\phi_\beta(z) = \frac{1}{k}\sum_{i=1}^k z_{[i]}, \qquad k = (1-\beta)m,
$$

where $z_{[1]} \geq z_{[2]} \geq \cdots \geq z_{[m]}$ are the components of $z$ sorted in nonincreasing order. In other words, CVaR is the average of the $k$ largest entries. For example, with $\beta = 0.95$ and $m = 10{,}000$ scenarios, CVaR is the average of the 500 largest entries.

The CVaR constraint $\phi_\beta(z) \leq \kappa$ is equivalent to $f_k(z) \leq d$, where $f_k(z) = \sum_{i=1}^k z_{[i]}$ is the sum of the $k$ largest components and $d = \kappa k$.

## Usage

### Solving a CVQP

The CVQP solver handles problems of the form

$$
\begin{array}{ll}
\text{minimize} & (1/2)\, x^T P x + q^T x \\
\text{subject to} & \phi_\beta(Ax) \leq \kappa \\
                  & l \leq Bx \leq u,
\end{array}
$$

where $x \in \mathbf{R}^n$ is the decision variable, $P \in \mathbf{S}^n_+$ is positive semidefinite (or absent for linear objectives), and $A \in \mathbf{R}^{m \times n}$ maps $x$ to $m$ scenario losses. The additional constraints $l \leq Bx \leq u$ encode box constraints, equality constraints, or other polyhedral constraints on $x$. The following example solves a portfolio optimization problem with a CVaR constraint on losses.

```python
import numpy as np
import scipy.sparse as sp
import cvqp

# Portfolio: n assets, m historical return scenarios
n, m = 10, 5000
np.random.seed(0)
R = np.random.randn(m, n) * 0.05 + 0.01     # return scenarios
mu = R.mean(axis=0)                         # expected returns
Sigma = np.cov(R.T)                         # covariance

P = Sigma
q = -mu
A = -R                                      # losses = negative returns
B = sp.vstack([np.ones(n), sp.eye(n)])      # sum-to-one + long-only
l = np.concatenate([[1.0], np.zeros(n)])
u = np.concatenate([[1.0], np.full(n, np.inf)])

result = cvqp.solve(P, q, A, B, l, u, beta=0.95, kappa=0.1)
print(result.status, result.value)
```

### CVaR projection

The CVaR projection finds the closest point to a vector $v$ that satisfies a CVaR constraint,

$$
\begin{array}{ll}
\text{minimize} & \lVert v - z \rVert_2^2 \\
\text{subject to} & \phi_\beta(z) \leq \kappa.
\end{array}
$$

```python
import numpy as np
import cvqp

v = np.random.randn(100_000)
z = cvqp.proj_cvar(v, beta=0.95, kappa=1.0)
```

### Sum-of-k-largest projection

Since $\phi_\beta(z) \leq \kappa$ is equivalent to $f_k(z) \leq d$ with $k = \lceil(1-\beta)m\rceil$ and $d = \kappa k$, the projection can also be expressed as

$$
\begin{array}{ll}
\text{minimize} & \lVert v - z \rVert_2^2 \\
\text{subject to} & f_k(z) \leq d,
\end{array}
$$

and is available directly as `proj_sum_largest`.

```python
import numpy as np
import cvqp

v = np.random.randn(100_000)
k = int(0.05 * len(v))  # same as ceil((1 - 0.95) * m)
d = 1.0 * k             # same as kappa * k
z = cvqp.proj_sum_largest(v, k, d)
```

The two projection functions are equivalent: `proj_cvar(v, beta, kappa)` converts to `proj_sum_largest(v, k, d)` internally.

## Paper experiments

See `experiments/` to reproduce the numerical results from the paper.

```bash
python experiments/plot.py           # generate figures from existing results
python experiments/run.py portfolio  # re-run CVQP benchmarks
python experiments/run.py projection # re-run projection benchmarks
```

## Citation

```bibtex
@article{luxenberg2025operator,
  title={An operator splitting method for large-scale CVaR-constrained quadratic programs},
  author={Luxenberg, Eric and P{\'e}rez-Pi{\~n}eiro, David and Diamond, Steven and Boyd, Stephen},
  journal={arXiv preprint arXiv:2504.10814},
  year={2025}
}
```
