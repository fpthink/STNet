#python3 setup.py install
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='voxelization',
    ext_modules=[
        CUDAExtension('voxelize_cuda', [
            'src/bindings.cpp',

            'src/voxelization/vox.cpp',
            'src/voxelization/vox.cu'
        ],)
    ],
    cmdclass={'build_ext': BuildExtension})
