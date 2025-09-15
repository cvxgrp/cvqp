// proj_sum_largest.h

#ifndef PROJ_SUM_LARGEST_H
#define PROJ_SUM_LARGEST_H

#include <vector>
#include <tuple>

std::tuple<int, int, bool> sum_largest_proj(double *z, int n, int k, double alpha, int untied, int tied, int cutoff, bool debug);

#endif  // PROJ_SUM_LARGEST_H