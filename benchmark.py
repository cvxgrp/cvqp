import numpy as np
import time
import matplotlib.pyplot as plt
import cvxpy as cp
import tqdm as tqdm
from admm import run_admm

import scipy.stats as sc
def generate_gmm(m, d, sigma):
    # X = np.random.randn(m, d)
    # y = np.random.binomial(1, .8, m)
    # X[y == 1] += .2
    # X[y == 0] -= .2
    # X[y == 0] *= sigma

    # m_0 = int(.8*m)
    # X_0 = sc.norm.rvs(loc=.2, size=(m_0, d))
    # X_1 = sc.norm.rvs(loc=-.2, scale = sigma, size=(m - m_0, d))
    # X = np.vstack((X_0, X_1))

    # Preallocate an empty array
    X = np.empty((m, d))

    # Calculate the number of samples for each group
    m_0 = int(.8*m)
    m_1 = m - m_0

    # Fill the array in-place
    X[:m_0] = sc.norm.rvs(loc=.2, size=(m_0, d))
    X[m_0:] = sc.norm.rvs(loc=-.2, scale=sigma, size=(m_1, d))

    # compute the empirical covariance matrix
    P = np.cov(X.T) if m < 1e6 else np.eye(d)
    # compute the mean of the data
    q = np.mean(X, axis=0)

    # multiply X by -1 in place
    X *= -1

    
    return X, P, -q

def main():

    np.random.seed(0)
    m = 3_000
    d = 3_000
    A, P, q = generate_gmm(m, d, 2)

    beta = .1
    kappa = .2

    k = int(beta * m)
    alpha = kappa * k

    gamma = .05

    A_box = np.eye(d)
    A_total = np.ones((1, d))

    def box_prox(z):
        return np.clip(z, 0, lim)

    def total_prox(z):
        return np.clip(z, 0, 1)

    proj_As = [A_box, A_total]
    proj_fns = [box_prox, total_prox]

    lim = 1
    def gen_constraints(x):
        return [
            0 <= cp.sum(x), cp.sum(x) <= 1,
            0 <= x, x <= lim,
        ]

    def gen_objective(x):
        return cp.Minimize(.5 * cp.quad_form(x, gamma*P) + q @ x)


    # solve the problem with cvxpy for reference
    x_cvxpy = cp.Variable(d)
    objective = gen_objective(x_cvxpy)
    constraints = gen_constraints(x_cvxpy)
    constraints += [
        cp.sum_largest(A@x_cvxpy, k) <= alpha,
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=True, canon_backend=cp.SCIPY_CANON_BACKEND)
    print("Optimal value: ", prob.value)
    print("CVaR and limit: ", np.sort(A @ x_cvxpy.value)[::-1][:k].sum() / m, kappa)
    print("Solve time: ", prob._solve_time)

if __name__ == "__main__":
    main()