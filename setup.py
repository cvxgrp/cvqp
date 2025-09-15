from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ext_modules = [
    Pybind11Extension(
        "cvqp.libs.proj_sum_largest",
        ["cvqp/libs/bindings.cpp", "cvqp/libs/proj_sum_largest.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3"],
    ),
]


setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext}, zip_safe=False)
