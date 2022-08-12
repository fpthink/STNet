import os
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))
_backend = load(name='_backend4',
                extra_cflags=['-O3', '-std=c++17'],
                sources=[os.path.join(_src_path,'src', f) for f in [
                    'voxelization/vox.cpp',
                    'voxelization/vox.cu',
                    'voxelize_float/voxf.cpp',
                    'voxelize_float/voxf.cu',
                    'bindings.cpp',
                ]]
                )

__all__ = ['_backend']
