#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include "projection.h"

namespace py = pybind11;

py::tuple py_proj_sum_largest(
    py::array_t<double> z, int k, double alpha,
    int n_untied, int n_tied
) {
    py::buffer_info buf = z.request();
    double* ptr = static_cast<double*>(buf.ptr);
    int n = buf.size;

    auto [final_untied, final_tied, converged] =
        proj_sum_largest(ptr, n, k, alpha, n_untied, n_tied);

    return py::make_tuple(z, final_untied, final_tied, converged);
}

PYBIND11_MODULE(projection, m) {
    m.doc() = "C++ projection onto sum-of-k-largest constraint";
    m.def("proj_sum_largest", &py_proj_sum_largest,
          "Project onto {z : sum of k largest elements <= alpha}");
}
