from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ext_modules = [
    Pybind11Extension(
        "cvqp._cpp.projection",
        ["cvqp/_cpp/bindings.cpp", "cvqp/_cpp/projection.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3"],
    ),
]


setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext}, zip_safe=False)
