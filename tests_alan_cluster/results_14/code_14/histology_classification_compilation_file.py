from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("histology_classification_cython_code.pyx")
)

