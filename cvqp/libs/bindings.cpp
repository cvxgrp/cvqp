#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <tuple>
#include "proj_sum_largest.h" // Include the declarations

namespace py = pybind11;

py::tuple proj_sum_largest(py::array_t<double> z, int k, double alpha, int untied, int tied, int cutoff, bool debug) {
    // Extract the NumPy array data into a C++ vector
    py::buffer_info buf_info = z.request();
    
    double* ptr = static_cast<double*>(buf_info.ptr);
    int n = buf_info.size;

    auto r = sum_largest_proj(ptr, n, k, alpha, untied, tied, cutoff, debug);
    
    return py::make_tuple(z, std::get<0>(r), std::get<1>(r), std::get<2>(r));

}

PYBIND11_MODULE(proj_sum_largest, m) {
    m.doc() = "Python bindings for proj_sum_largest";
    m.def("proj_sum_largest", &proj_sum_largest, "Project onto sum-of-k-largest constraint");
}
