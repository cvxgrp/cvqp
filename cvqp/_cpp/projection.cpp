// O(m log m) Euclidean projection onto the sum-of-k-largest constraint.
//
// Algorithm: iteratively reduce the sum of k largest elements to <= alpha.
// Maintains three groups in the sorted (descending) array:
//   [0, n_untied)                        -- decrease independently
//   [n_untied, n_untied + n_tied)        -- all equal, decrease together
//   [n_untied + n_tied, n)               -- unchanged
//
// Each iteration computes a step direction (delta_u, delta_t) and three
// candidate step sizes:
//   step_merge:  step until the smallest untied element equals the tied value
//   step_absorb: step until the tied value equals the next inactive element
//   excess:      step that exactly satisfies the constraint
// The actual step is min(step_merge, step_absorb, excess). After the step,
// groups may merge (untied joins tied) or grow (inactive joins tied).
//
// Total complexity: O(m log m) dominated by sorting; the loop runs at most
// m iterations, each in O(1) time.

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>
#include "projection.h"

static double compute_initial_sum(const double* z, int k) {
    double s = 0.0;
    for (int i = 0; i < k; i++) s += z[i];
    return s;
}

static std::pair<double, double> compute_step_direction(
    int n_untied, int n_tied, int k
) {
    // Returns (delta_u, delta_t): per-unit decrease for untied and tied groups.
    int n_in_tied_block = k - n_untied;

    double delta_u = (n_in_tied_block > 0)
        ? static_cast<double>(n_tied) / n_in_tied_block
        : 1.0;
    if (n_untied == 0) delta_u = 1.0;

    double delta_t = (n_tied > 0) ? 1.0 : 0.0;

    // Normalize so that the total decrease per unit step equals 1
    double norm;
    if (n_in_tied_block > 0) {
        norm = static_cast<double>(n_untied) * n_tied / n_in_tied_block
             + n_in_tied_block;
    } else {
        norm = k;
    }
    delta_u /= norm;
    delta_t /= norm;

    return {delta_u, delta_t};
}

static std::pair<double, double> compute_step_sizes(
    int n_untied, int n_tied, int n,
    double delta_u, double delta_t,
    double tied_val, double last_untied_val, double next_inactive_val
) {
    const double inf = std::numeric_limits<double>::infinity();

    // step_merge: step until smallest untied element equals tied value
    double step_merge = inf;
    if (n_untied > 0) {
        step_merge = (tied_val - last_untied_val) / (delta_t - delta_u);
    }

    // step_absorb: step until tied value equals next inactive element
    double step_absorb = inf;
    if (n_untied + n_tied < n) {
        double tail_delta = (n_tied == 0) ? delta_u : delta_t;
        double tail_val = (n_tied == 0) ? last_untied_val : tied_val;
        step_absorb = (next_inactive_val - tail_val) / (0.0 - tail_delta);
    }

    return {step_merge, step_absorb};
}

static void apply_projection(
    double* z, int n_untied, int n_tied,
    double untied_decrease, double tied_val
) {
    for (int i = 0; i < n_untied; i++) {
        z[i] -= untied_decrease;
    }
    for (int i = n_untied; i < n_untied + n_tied; i++) {
        z[i] = tied_val;
    }
}

std::tuple<int, int, bool> proj_sum_largest(
    double* z, int n, int k, double alpha,
    int n_untied, int n_tied
) {
    const double TOL = 1e-9;

    double sum_k = compute_initial_sum(z, k);
    if (sum_k <= alpha + TOL) {
        return {n_untied, n_tied, true};
    }

    double untied_decrease = 0.0;
    double tied_val = z[k];
    double last_untied_val = z[k - 1];
    double next_inactive_val = z[k];

    int final_n_untied = n_untied;
    int final_n_tied = n_tied;

    for (int iter = 0; iter < n; iter++) {
        auto [delta_u, delta_t] = compute_step_direction(n_untied, n_tied, k);
        double excess = sum_k - alpha;

        auto [step_merge, step_absorb] = compute_step_sizes(
            n_untied, n_tied, n,
            delta_u, delta_t,
            tied_val, last_untied_val, next_inactive_val);

        double step = std::min({step_merge, step_absorb, excess});

        // Apply step
        sum_k -= step * n_untied * delta_u;
        sum_k -= step * static_cast<double>(k - n_untied) * delta_t;

        if (n_tied > 0) tied_val -= step * delta_t;
        untied_decrease += step * delta_u;

        final_n_untied = n_untied;
        final_n_tied = n_tied;

        // Update group boundaries
        if (step == step_merge) {
            n_untied = std::max(n_untied - 1, 0);
        }
        if (n_tied == 0) n_tied = 1;
        n_tied = std::min(n_tied + 1, n);

        if (n_untied > 0) {
            last_untied_val = z[n_untied - 1] - untied_decrease;
        }
        if (n_untied + n_tied < n) {
            next_inactive_val = z[n_untied + n_tied];
        }

        if (sum_k <= alpha + TOL) break;
    }

    apply_projection(z, final_n_untied, final_n_tied, untied_decrease, tied_val);

    bool converged = (sum_k <= alpha + TOL);
    return {final_n_untied, final_n_tied, converged};
}
