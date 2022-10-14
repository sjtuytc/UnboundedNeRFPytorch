import os
import pdb
from setuptools import setup, Extension
from torch.utils import cpp_extension


def setup_sources(name, sources):
      parent_dir = os.path.dirname(os.path.abspath(__file__))
      sources = [os.path.join(parent_dir, path) for path in sources]
      setup(name=name,
            ext_modules=[cpp_extension.CppExtension(name=name, sources=sources)],
            cmdclass={'build_ext': cpp_extension.BuildExtension})
      return

setup_sources(name='adam_upd_cuda', sources=['adam_upd.cpp', 'adam_upd_kernel.cu'])
setup_sources(name='ub360_utils_cuda', sources=['ub360_utils.cpp', 'ub360_utils_kernel.cu'])
setup_sources(name='render_utils_cuda', sources=['render_utils.cpp', 'render_utils_kernel.cu'])
setup_sources(name='total_variation_cuda', sources=['total_variation.cpp', 'total_variation_kernel.cu'])

# run python setup.py install