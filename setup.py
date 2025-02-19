import sys

if sys.version_info < (3, 6):
    sys.stdout.write(
        "Minkowski Engine requires Python 3.6 or higher. Please use anaconda https://www.anaconda.com/distribution/ for an isolated python environment.\n"
    )
    sys.exit(1)

try:
    import torch
except ImportError:
    raise ImportError("Pytorch not found. Please install pytorch first.")

import codecs
import os
import re
import subprocess
from pathlib import Path
from sys import argv, platform

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

if platform == "win32":
    raise ImportError("Windows is currently not supported.")
if platform == "darwin":
    # Set the distutils to use clang instead of g++ for valid std
    if "CC" not in os.environ:
        os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def run_command(*args):
    subprocess.check_call(args)


run_command("pip", "uninstall", "MinkowskiEngine", "-y")


CPU_ONLY = False
FORCE_CUDA = True


# args with return value
MAX_COMPILATION_THREADS = 12

Extension = CUDAExtension
extra_link_args = []
include_dirs = []
libraries = []
CC_FLAGS = []
NVCC_FLAGS = []

if CPU_ONLY:
    print("--------------------------------")
    print("| WARNING: CPU_ONLY build set  |")
    print("--------------------------------")
    Extension = CppExtension
else:
    print("--------------------------------")
    print("| CUDA compilation set         |")
    print("--------------------------------")
    # system python installation
    libraries.append("cusparse")


if sys.platform == "win32":
    vc_version = os.getenv("VCToolsVersion", "")
    if vc_version.startswith("14.16."):
        CC_FLAGS += ["/sdl"]
    else:
        CC_FLAGS += ["/sdl", "/permissive-"]
else:
    CC_FLAGS += ["-fopenmp"]

if "darwin" in platform:
    CC_FLAGS += ["-stdlib=libc++", "-std=c++17"]

NVCC_FLAGS += ["--expt-relaxed-constexpr", "--expt-extended-lambda"]

# The Ninja cannot compile the files that have the same name with different
# extensions correctly and uses the nvcc/CC based on the extension. Import a
# .cpp file to the corresponding .cu file to force the nvcc compilation.
SOURCE_SETS = {
    "cpu": [
        CppExtension,
        [
            "math_functions_cpu.cpp",
            "coordinate_map_manager.cpp",
            "convolution_cpu.cpp",
            "convolution_transpose_cpu.cpp",
            "local_pooling_cpu.cpp",
            "local_pooling_transpose_cpu.cpp",
            "global_pooling_cpu.cpp",
            "broadcast_cpu.cpp",
            "pruning_cpu.cpp",
            "interpolation_cpu.cpp",
            "quantization.cpp",
            "direct_max_pool.cpp",
        ],
        ["pybind/minkowski.cpp"],
        ["-DCPU_ONLY"],
    ],
    "gpu": [
        CUDAExtension,
        [
            "math_functions_cpu.cpp",
            "math_functions_gpu.cu",
            "coordinate_map_manager.cu",
            "coordinate_map_gpu.cu",
            "convolution_kernel.cu",
            "convolution_gpu.cu",
            "depthwise_convolution_kernel.cu",
            "depthwise_convolution_gpu.cu",
            "convolution_transpose_gpu.cu",
            "pooling_avg_kernel.cu",
            "pooling_max_kernel.cu",
            "local_pooling_gpu.cu",
            "local_pooling_transpose_gpu.cu",
            "global_pooling_gpu.cu",
            "broadcast_kernel.cu",
            "broadcast_gpu.cu",
            "pruning_gpu.cu",
            "interpolation_gpu.cu",
            "spmm.cu",
            "gpu.cu",
            "quantization.cpp",
            "direct_max_pool.cpp",
        ],
        ["pybind/minkowski.cu"],
        [],
    ],
}


HERE = Path(os.path.dirname(__file__)).absolute()
SRC_PATH = HERE / "src"

if "CC" in os.environ or "CXX" in os.environ:
    # distutils only checks CC not CXX
    if "CXX" in os.environ:
        os.environ["CC"] = os.environ["CXX"]
        CC = os.environ["CXX"]
    else:
        CC = os.environ["CC"]
    print(f"Using {CC} for c++ compilation")
    if torch.__version__ < "1.7.0":
        NVCC_FLAGS += [f"-ccbin={CC}"]
else:
    print("Using the default compiler")


CC_FLAGS += ["-O3"]
NVCC_FLAGS += ["-O3", "-Xcompiler=-fno-gnu-unique"]

if "MAX_JOBS" not in os.environ and os.cpu_count() > MAX_COMPILATION_THREADS:
    # Clip the num compilation thread to 8
    os.environ["MAX_JOBS"] = str(MAX_COMPILATION_THREADS)

target = "cpu" if CPU_ONLY else "gpu"

Extension = SOURCE_SETS[target][0]
SRC_FILES = SOURCE_SETS[target][1]
BIND_FILES = SOURCE_SETS[target][2]
ARGS = SOURCE_SETS[target][3]
CC_FLAGS += ARGS
NVCC_FLAGS += ARGS

ext_modules = [
    Extension(
        name="MinkowskiEngineBackend._C",
        sources=[*[str(SRC_PATH / src_file) for src_file in SRC_FILES], *BIND_FILES],
        extra_compile_args={"cxx": CC_FLAGS, "nvcc": NVCC_FLAGS},
        libraries=libraries,
    ),
]

# Python interface
setup(
    name="MinkowskiEngine",
    version=find_version("MinkowskiEngine", "__init__.py"),
    install_requires=["torch", "numpy"],
    packages=["MinkowskiEngine", "MinkowskiEngine.utils", "MinkowskiEngine.modules"],
    package_dir={"MinkowskiEngine": "./MinkowskiEngine"},
    ext_modules=ext_modules,
    include_dirs=[str(SRC_PATH), str(SRC_PATH / "3rdparty"), *include_dirs],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    author="Christopher Choy",
    author_email="chrischoy@ai.stanford.edu",
    description="a convolutional neural network library for sparse tensors",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/NVIDIA/MinkowskiEngine",
    keywords=[
        "pytorch",
        "Minkowski Engine",
        "Sparse Tensor",
        "Convolutional Neural Networks",
        "3D Vision",
        "Deep Learning",
    ],
    zip_safe=False,
    classifiers=[
        # https: // pypi.org/classifiers/
        "Environment :: Console",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Other Audience",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.6",
)
