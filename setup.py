from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "mybindings",
        ["mybindings.cpp", "sum_largest_proj.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=['-std=c++11', '-O3'],
        language='c++'
    ),
]

setup(
    name="mybindings",
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.10.0'],
    python_requires=">=3.11"
)