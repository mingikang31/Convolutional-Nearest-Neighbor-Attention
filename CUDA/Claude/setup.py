"""
setup.py for the flash_convnn_cuda extension.

Build with:
    pip install -e .          # editable install (recommended for development)
    # or
    python setup.py build_ext --inplace

Requires:
    torch >= 2.0, CUDA toolkit matching torch.version.cuda, a C++17 compiler.

Tested target archs: A100 (sm_80) and H100 (sm_90). Adjust TORCH_CUDA_ARCH_LIST
in your environment if needed, or edit the `arch_flags` list below.
"""
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Match A100 / H100. Add more as needed (e.g. '7.5' for T4, '8.6' for A40/3090).
arch_flags = []
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    # Default: build for A100 (8.0) and H100 (9.0) since those are Mingi's target hw
    arch_flags = [
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_90,code=sm_90",
    ]

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-lineinfo",
] + arch_flags

cxx_flags = ["-O3", "-std=c++17"]

setup(
    name="flash_convnn_cuda",
    version="0.1.0",
    description="FlashConvNN-Attention CUDA kernels (depthwise weighting).",
    ext_modules=[
        CUDAExtension(
            name="flash_convnn_cuda",
            sources=[
                "csrc/flash_convnn_attention.cpp",
                "csrc/flash_convnn_attention_kernel.cu",
            ],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    py_modules=["flash_convnn_attention"],
)
