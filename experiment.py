import numpy as np
import time
import matplotlib.pyplot as plt
import cvxpy as cp
import tqdm as tqdm
from admm import run_admm
import pickle

def generate_gmm(m, d, sigma):
    X = np.random.randn(m, d)
    y = np.random.binomial(1, .8, m)
    X[y == 1] += .2
    X[y == 0] -= .2
    X[y == 0] *= sigma

    # compute the empirical covariance matrix
    P = np.cov(X.T)
    # compute the mean of the data
    q = np.mean(X, axis=0)
    
    return -X, P, -q


beta = .1
kappa = .2
gamma = .05

def box_prox(z):
    return np.clip(z, 0, lim)

def total_prox(z):
    return np.clip(z, 0, 1)

proj_fns = [box_prox, total_prox]

lim = 1
def gen_constraints(x):
    return [
        0 <= cp.sum(x), cp.sum(x) <= 1,
        0 <= x, x <= lim,
    ]

def gen_objective(x):
    return cp.Minimize(.5 * cp.quad_form(x, gamma*P) + q @ x)

d = 2000
scenarios = [10_000] #, 50_000, 100_000]
solve_times_cvxpy = []
solve_times_admm = []

for m in scenarios:
    print(f"Running for {m} scenarios")
    # Data generation
    A, P, q = generate_gmm(m, d, 2)
    k = int(beta * m)
    alpha = kappa * k
    A_box = np.eye(d)
    A_total = np.ones((1, d))
    proj_As = [A_box, A_total]
    def gen_objective(x):
        return cp.Minimize(.5 * cp.quad_form(x, gamma*P) + q @ x)
    
    # Solution via CVXPY
    x_cvxpy = cp.Variable(d)
    objective = gen_objective(x_cvxpy)
    constraints = gen_constraints(x_cvxpy)
    constraints += [
        cp.sum_largest(A@x_cvxpy, k) <= alpha ,
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False, canon_backend=cp.SCIPY_CANON_BACKEND)
    solver_time = prob._solve_time
    solve_times_cvxpy.append(solver_time)
    print("\tCVXPY solve time: ", solver_time)
    
    # Solution via ADMM
    ts = time.time()
    x, history = run_admm(
     gamma*P, q, A, beta, kappa, proj_As, proj_fns, 
     max_iter=10000, 
     alpha_over=1.7, rho=.1, 
     warm=None, #x_cvxpy_.value, 
     verbose=False,
     # constraint_func=gen_constraints
)
    te = time.time()
    solve_times_admm.append(te - ts)
    print("\tADMM solve time: ", te - ts)

    # pickle the scenarios and solve times
data = {
    "scenarios": scenarios,
    "solve_times_cvxpy": solve_times_cvxpy,
    "solve_times_admm": solve_times_admm,
}
with open("solve_times_script.pkl", "wb") as f:
    pickle.dump(data, f)