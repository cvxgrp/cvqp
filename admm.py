from tqdm import tqdm
import time
import numpy as np
import scipy as sp
import cvxpy as cp
from cvar_proj import proj_sum_largest, proj_sum_largest_cvxpy, proj_sum_largest_cpp

def factor_solve(factor, rhs):
    return sp.linalg.lu_solve(factor, rhs)

def update_x(
    factor,
    A: np.ndarray,
    A_tilde: np.ndarray,
    q: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
    z_tilde: np.ndarray,
    u_tilde: np.ndarray,
    rho: float,
) -> np.ndarray:
    """
    The x-update in the ADMM algorithm consists of minimizing a quadratic function, which is equivalent to
    solving a linear system of equations where the coefficient matrix is positive definite. We first factorize the matrix using Cholesky,
    and then solve the linear system using the cached factorization. The linear system is of the form L'Lx = A'v - c/rho, where A' is the transpose of A and
    L is the lower triangular Cholesky factor of A'A.

    Args:
        L (np.ndarray): The lower triangular Cholesky factor of the positive-definite matrix A'A.
        prox_fns (list[ProxFn]): List of ProxFn objects that define the proximal operators for each constraint.
        v_list (list[np.ndarray]): List of tensors (z - u).
        c (np.ndarray): A vector of shape (num_assets * num_energy_segments) representing average losses.
        rho (float): The ADMM penalty parameter.

    Returns:
        np.ndarray: The solution to the linear system.
    """

    # rhs = -q + rho * A.T @ (z - u) + rho * A_tilde.T @ (z_tilde - u_tilde)
    rhs = -q + rho * y_1(A, z, u) + rho * y_2(A_tilde, z_tilde, u_tilde)
    
    # x = sp.linalg.lu_solve(factor, rhs)
    x = factor_solve(factor, rhs)

    return x

def y_1(A, z, u):
    return A.T @ (z - u)

def y_2(A_tilde, z_tilde, u_tilde):
    return A_tilde.T @ (z_tilde - u_tilde)

def run_admm(P, q, A, beta, kappa, proj_As, proj_fns, max_iter=10_000, alpha=.5, rho=1.0, abstol=1e-4, reltol=1e-2, alpha_over=1.7, print_freq=100, max_time_sec=1_200, warm=None):
        
    history = { 
        "iter": [],
        "objval": [],
        "r_norm": [],
        "s_norm": [],
        "eps_pri": [],
        "eps_dual": [],
        "rho": [],
    }
    start_time = time.time()

    m = A.shape[0]
    k = int(beta * m)
    alpha = kappa * k

    AtA = A.T @ A
    AtA_tilde = sum([A.T @ A for A in proj_As])
    A_tilde = np.vstack(proj_As)

    # form the matrix P + rho * A'A + rho A_tilde'A_tilde
    M = P + rho * AtA + rho * AtA_tilde
    factor = sp.linalg.lu_factor(M)

    z = np.zeros(A.shape[0]) if warm is None else A @ warm
    u = np.zeros(A.shape[0])
 
    z_tildes = [np.zeros(A.shape[0]) for A in proj_As] if warm is None else [A @ warm for A in proj_As]
    u_tildes = [np.zeros(A.shape[0]) for A in proj_As]

    d0 = sum(A.shape[0] for A in proj_As) + A.shape[0]
    d1 = A.shape[1]
    
    # for i in tqdm(range(max_iter)):
    for i in range(max_iter):
        z_old = z.copy()
        z_tilde_old = [z.copy() for z in z_tildes]
        
        # x-update: solve the linear system 
        z_tilde = np.concatenate(z_tildes)
        u_tilde = np.concatenate(u_tildes)
        x = update_x(factor=factor, A=A, A_tilde=A_tilde, q=q, z=z, u=u, z_tilde=z_tilde, u_tilde=u_tilde, rho=rho)

        # z-update: apply projection for cvar
        z_hat = alpha_over * A @ x + (1 - alpha_over) * z + u
        # z_hat =  A @ x + u
        
        # z = proj_sum_largest(z_hat, k, alpha)
        z = proj_sum_largest_cpp(z_hat, k, alpha)

        time.time()

        # z-update: apply projection operators for each constraint
        z_hats = [ alpha_over * A @ x + (1 - alpha_over) * z_ + u for A, u, z_ in zip(proj_As, u_tildes, z_tildes)]
        # z_hats = [ A @ x + u for A, u in zip(proj_As, u_tildes)]

        for j in range(len(z_tildes)):
            z_tildes[j] = proj_fns[j](z_hats[j])

        # Update dual variables.
        u = z_hat - z
        u_tildes = [z_hat - z for z_hat, z in zip(z_hats, z_tildes)]

        # Compute norm of primal and dual residuals. We only compute this every rho_update_interval iterations.
        if i % print_freq == 0:
            # Ax = A_x(prox_fns, x)
            Ax = np.concatenate([A @ x]+[A @ x for A in proj_As])
            r = Ax - np.concatenate([z] + z_tildes)
            r_norm = np.linalg.norm(r)

            z_diff = z - z_old
            z_diffs = [z_ - z_old_ for z_, z_old_ in zip(z_tildes, z_tilde_old)]
            At_z = sum(
                [A.T @ z_diff] + 
                [A_.T @  z_diff for A_, z_diff in zip(proj_As, z_diffs)]
            )
            s_norm = np.linalg.norm(rho * At_z)


            # Diagnostics, reporting, termination checks
            # Compute and save convergence metrics
            history["iter"].append(i)
            objval = 0.5 * np.dot(x, P @ x) + q @ x
            history["objval"].append(objval)
            history["r_norm"].append(r_norm)
            history["s_norm"].append(s_norm)
            print(f"iter: {i}, objval: {objval}, r_norm: {r_norm}, s_norm: {s_norm}, u_norm: {np.linalg.norm(np.concatenate([u] + u_tildes))}, time: {time.time() - start_time}")

            eps_pri = (d0 ** .5) * abstol + reltol * max(np.linalg.norm(Ax), np.linalg.norm(np.concatenate([z] + z_tildes)))
            eps_dual = (d1 ** .5) * abstol + reltol * np.linalg.norm(rho * At_z)
            history["eps_pri"].append(eps_pri)
            history["eps_dual"].append(eps_dual)

            if (
                history["r_norm"][-1] < history["eps_pri"][-1]
                and history["s_norm"][-1] < history["eps_dual"][-1]
            ):
                break

            # Break loop if time limit is reached.
            if time.time() - start_time > max_time_sec:
                break
    
    print("ADMM terminated after ", i, " iterations")
    print("Time: ", time.time() - start_time)
    return x, history


def test():

    m = 10_000
    d = 500
    np.random.seed(0)
    A = np.random.randn(m, d) * .2 + .1

    P = np.eye(d) #* (.2**2)
    q = np.ones(d) * -.1 + np.random.randn(d) * .05

    beta = .1
    kappa = .1

    A_box = np.eye(d)
    proj_As = [A_box]
    
    def box_prox(z):
        return np.clip(z, -1, 1)
    proj_fns = [box_prox]

    # solve the problem with cvxpy for reference
    m = A.shape[0]
    k = int(beta * m)
    alpha = kappa * k
    x_cvxpy = cp.Variable(d)
    objective = cp.Minimize(0.5 * cp.quad_form(x_cvxpy, P) + q @ x_cvxpy)
    constraints = [cp.sum_largest(A@x_cvxpy, k) <= alpha , -1 <= x_cvxpy, x_cvxpy <= 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=True)
    print(prob.value)
    # print the sum of the largest k elements of A @ x_cvxpy.value
    print(np.sort(A @ x_cvxpy.value)[::-1][:k].sum())


    x, history = run_admm(
        P, q, A, beta, kappa, proj_As, proj_fns, max_iter=500, #warm=x_cvxpy.value
    )

    print(history["objval"][-1])
    print(np.sort(A @ x)[::-1][:k].sum())

    print("relative objective error: ", abs(history["objval"][-1] - prob.value) / abs(prob.value))

if __name__ == "__main__":
    test()