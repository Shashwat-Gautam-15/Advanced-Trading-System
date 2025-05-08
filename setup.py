from setuptools import setup, Extension
import pybind11

module = Extension(
    'order_execution',
    sources=['order_execution.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=['-std=c++11'],
)

setup(
    name='order_execution',
    version='1.0',
    description='Order execution engine',
    ext_modules=[module],
)