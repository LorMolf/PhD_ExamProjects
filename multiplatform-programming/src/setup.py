from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# List all your C++ source files here
cpp_sources = [
    "../CompHom/comphom.cpp",
    "../CompHom/matrix.cpp",
    "../CompHom/simplex.cpp",
]

extensions = [
    Extension(
        "comphom_wrapper",
        sources=["comphom_wrapper.pyx"] + cpp_sources,  # Include C++ sources
        language="c++",
        include_dirs=[".", "../CompHom"],  # Adjust paths if headers are elsewhere
        extra_compile_args=["-std=c++11"],
    )
]

setup(
    name="comphom_wrapper",
    ext_modules=cythonize(extensions),
)
