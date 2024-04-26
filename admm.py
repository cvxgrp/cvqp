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
    At: np.ndarray,
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
        factor (np.ndarray): The Cholesky factor of the coefficient matrix.
        A (np.ndarray): The matrix A.
        At (np.ndarray): The transpose of A.
        A_tilde (np.ndarray): The matrix A_tilde.
        q (np.ndarray): The vector q.
        z (np.ndarray): The vector z.
        u (np.ndarray): The vector u.
        z_tilde (np.ndarray): The vector z_tilde.
        u_tilde (np.ndarray): The vector u_tilde.
        rho (float): The penalty parameter rho.

    Returns:
        np.ndarray: The solution to the linear system.
    """

    # rhs = -q + rho * A.T @ (z - u) + rho * A_tilde.T @ (z_tilde - u_tilde)
    rhs = -q + rho * y_1(A, z, u, At) + rho * y_2(A_tilde, z_tilde, u_tilde)
    
    # x = sp.linalg.lu_solve(factor, rhs)
    x = factor_solve(factor, rhs)

    return x

def y_1(A, z, u, At):
    # return A.T @ (z - u)
    return At @ (z - u)

def y_2(A_tilde, z_tilde, u_tilde):
    return A_tilde.T @ (z_tilde - u_tilde)

def run_admm(P, q, A, beta, kappa, proj_As, proj_fns, max_iter=10_000, alpha=.5, rho=1.0, abstol=1e-4, reltol=1e-2, alpha_over=1.7, print_freq=100, max_time_sec=1_200, warm=None):
        
    mu = 10
    rho_incr = 1.5
    rho_decr = 1.5

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

    # scale problem data
    scale = max(-A.min(), A.max())
    # scale = 1
    A /= scale
    q /= scale
    P /= scale 
    alpha /= scale

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

    At = np.array(A.T, order='C')
    
    # for i in tqdm(range(max_iter)):
    for i in range(max_iter):
        z_old = z.copy()
        z_tilde_old = [z.copy() for z in z_tildes]
        
        # x-update: solve the linear system 
        z_tilde = np.concatenate(z_tildes)
        u_tilde = np.concatenate(u_tildes)
        x = update_x(factor=factor, A=A, At=At, A_tilde=A_tilde, q=q, z=z, u=u, z_tilde=z_tilde, u_tilde=u_tilde, rho=rho)

        # z-update: apply projection for cvar
        def over_relax_z_hat(x, z, alpha_over):  
            return alpha_over * (A @ x) + (1 - alpha_over) * z + u    
            # return A @ x + u

        z_hat = over_relax_z_hat(x, z, alpha_over)
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
            objval *= scale

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
            # if time.time() - start_time > max_time_sec:
            #     break
    
            # Update rho
            changed = True
            if r_norm > mu * s_norm:
                rho *= rho_incr
                u = u / rho_incr
                u_tildes = [u_ / rho_incr for u_ in u_tildes]
            elif s_norm > mu * r_norm:
                rho /= rho_decr
                u = u * rho_decr
                u_tildes = [u_ * rho_decr for u_ in u_tildes]
            else: 
                changed = False
            if changed: 
                M = P + rho * AtA + rho * AtA_tilde
                factor = sp.linalg.lu_factor(M)

    # unscale
    A *= scale
    q *= scale
    P *= scale
    alpha *= scale

    print("ADMM terminated after ", i, " iterations")
    print("Time: ", time.time() - start_time)
    return x, history


def test():

    m = 100_000
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
    print(np.sort(A @ x_cvxpy.value)[::-1][:k].sum(), alpha)


    x, history = run_admm(
        P, q, A, beta, kappa, proj_As, proj_fns, max_iter=6000, #warm=x_cvxpy.value / 2 
    )

    print(history["objval"][-1])
    print(np.sort(A @ x)[::-1][:k].sum(), alpha)

    print("relative objective error: ", abs(history["objval"][-1] - prob.value) / abs(prob.value))

if __name__ == "__main__":
    test()