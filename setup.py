from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gpu_ops',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=[
        CUDAExtension(
            name='gpu_ops._cuda',
            sources=[
                'src/cuda_ops/tensor_ops.cu',
                'src/bindings.cpp'
            ],
            include_dirs=['src/cuda_ops'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.7',
    install_requires=[
        'torch>=2.1.0',
        'numpy>=1.24.0'
    ]
) 