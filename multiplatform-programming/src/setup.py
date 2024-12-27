from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# Source files
cpp_sources = [
    "../CompHom/comphom.cpp",
    "../CompHom/matrix.cpp",
    "../CompHom/simplex.cpp",
]

extensions = [
    Extension(
        "comphom_wrapper",
        sources=["comphom_wrapper.pyx"] + cpp_sources,
        language="c++",
        include_dirs=[".", "../CompHom"],
        extra_compile_args=["-std=c++11"],
    )
]

setup(
    name="comphom_wrapper",
    ext_modules=cythonize(extensions),
)
