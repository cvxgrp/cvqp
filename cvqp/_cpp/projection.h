#ifndef CVQP_PROJECTION_H
#define CVQP_PROJECTION_H

#include <tuple>

// Project a sorted (descending) vector onto {z : sum of k largest <= alpha}.
// Returns (final_n_untied, final_n_tied, converged).
std::tuple<int, int, bool> proj_sum_largest(
    double* z, int n, int k, double alpha,
    int n_untied, int n_tied);

#endif
