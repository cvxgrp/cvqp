// sum_largest_proj.h

#ifndef SUM_LARGEST_PROJ_H
#define SUM_LARGEST_PROJ_H

#include <vector>
#include <tuple>

std::tuple<int, int, bool> sum_largest_proj(double *z, int n, int k, double alpha, int untied, int tied, int cutoff, bool debug);

#endif  // SUM_LARGEST_PROJ_H