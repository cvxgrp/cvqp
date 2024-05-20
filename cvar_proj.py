import numpy as np
import cvxpy as cp
import sys
sys.path.append('./build')
import mybindings

def proj_sum_largest_cvxpy(z: np.ndarray, k: int, alpha) -> np.ndarray:
    z = np.array(z)
    x = cp.Variable(z.shape)
    objective = cp.Minimize(cp.sum_squares(x - z))
    constraints = [cp.sum_largest(x, k) <= alpha]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK)
    return x.value

def _sort_z(z: np.ndarray,) -> tuple[np.ndarray, np.ndarray]:
    sorted_inds = np.argsort(z)[::-1]
    z = z[sorted_inds]
    return z, sorted_inds

def _unsort_z(z: np.ndarray, sorted_inds: np.ndarray) -> np.ndarray:
    x = np.empty_like(z)
    x[sorted_inds] = z
    return x

def form_delta(untied: int, tied: int, k: int) -> tuple[float, float]:
    """Calculates the scaling factors 'val_1' and 'val_2' used in the projection algorithm.
    These factors are determined based on the count of 'untied' and 'tied' elements relative
    to 'k', the number of elements to sum."""
    u = untied
    t = tied
    n = k - u

    untied_val = t / n if n > 0 else 1.0
    val_1 = untied_val if untied > 0 else 1.0
    val_2 = 1.0 if tied > 0 else 0.0

    if k - u > 0:
        a = u * t
        b = k - u
        t1 = a / b
        t2 = k - u
        normalization = t1 + t2
    else:
        normalization = k

    val_1 /= normalization
    val_2 /= normalization

    return val_1, val_2

def proj_sum_largest_sorted(
    z: np.ndarray, k: int, alpha: float,
) -> np.ndarray:
    """
    Projects the vector 'z' such that the sum of its largest 'k' elements is less than or equal to 'alpha'.
    This function assumes that 'z' is already sorted in descending order.
    """
    val = z[:k].sum()
    untied = k
    tied = 0
    iters = 0
    TOL = 1e-9
    n = z.shape[0]
    untied_decrease = 0.0  # decrease to untied elements
    tied_final = z[
        k
    ]  # the final value of tied elements, initialized to the value of the k+1st element
    last_untied_val = z[k - 1]  # the value of the last element in the untied block
    post_tied_val = z[k]  # the value of the first element after the tied block
    # inf in torch
    lim_inf = float("inf")
    MAX_ITERS = n
    final_untied = k
    final_tied = 0
    complete = False

    while (val > alpha + TOL) and (iters < MAX_ITERS):
        val_1, val_2 = form_delta(untied, tied, k)
        extra = val - alpha

        untied_val = val_1
        tied_val = val_2
        valid_s1 = untied > 0

        if valid_s1:
            # Find the decrease where the last untied element reaches the tied value
            # last_untied - s1 * untied_val = tied_final - s1 * tied_val
            s1 = (tied_final - last_untied_val) / (tied_val - untied_val)
        else:
            s1 = lim_inf

        valid_s2 = untied + tied < n

        if valid_s2:
            # the penultimate val is the rate at which the last element in the changing block changes
            if tied == 0:
                penultimate_val = untied_val
            else:
                penultimate_val = tied_val
            # v is the value of the last element in the changing block
            v = last_untied_val if tied == 0 else tied_final
            s2 = (post_tied_val - v) / (0.0 - penultimate_val)
        else:
            s2 = lim_inf

        s = min(s1, s2, extra)

        val -= s * untied * untied_val
        val -= s * (k - untied) * tied_val

        if tied > 0:
            tied_final -= s * tied_val

        untied_decrease += s * untied_val
        final_untied = untied
        final_tied = tied

        untied = max(untied - 1, 0) if s == s1 else untied
        if tied == 0:
            tied = 1
        tied = min(tied + 1, n)

        if untied > 0:
            last_untied_val = z[untied - 1] - untied_decrease
        if untied + tied < n:
            post_tied_val = z[untied + tied]

        iters += 1

    z[:final_untied] -= untied_decrease
    z[final_untied : final_untied + final_tied] = tied_final

    return z

def proj_sum_largest(
        z: np.ndarray, k: int, alpha: float
) -> np.ndarray:
    """
    Projects the vector 'z' such that the sum of its largest 'k' elements is less than or equal to 'alpha'.
    """
    z, sorted_inds = _sort_z(z)
    z = proj_sum_largest_sorted(z, k, alpha)
    z = _unsort_z(z, sorted_inds)
    return z


def proj_sum_largest_cpp(
        z: np.ndarray, k: int, alpha: float
) -> np.ndarray:
    """
    Projects the vector 'z' such that the sum of its largest 'k' elements is less than or equal to 'alpha'.
    """
    z, sorted_inds = _sort_z(z)
    _ = mybindings.sum_largest_proj(z, k, alpha, k, 0, len(z), False)
    z = _unsort_z(z, sorted_inds)
    return z