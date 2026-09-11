#!/usr/bin/env bash
# gsplat (github.com/nerfstudio-project/gsplat) needs a Python/PyTorch/CUDA
# stack the base conda env doesn't have, so it gets its own conda env and its
# own CUDA toolkit (via conda, since the base image ships CUDA runtime libs
# but no nvcc). /opt/gsplat is expected to already hold the third_party/gsplat
# submodule checkout (copied in by the Dockerfile's gsplatsrc build context).
set -e

GSPLAT_DIR="/opt/gsplat"

source /opt/miniconda3/etc/profile.d/conda.sh
conda create -yn gsplat python=3.10 pip
conda activate gsplat

# cuda-12.8.1 matches the cu128 tag of the pinned torch/torchvision wheels
# below. Installing just the pieces the CUDA extensions' headers need
# (nvcc, cudart, cub/thrust, and the BLAS/sparse/solver/RNG/FFT headers
# torch's own ATen CUDA headers pull in transitively, e.g.
# ATen/cuda/CUDAContextLight.h -> cusparse.h/cublas_v2.h/cusolverDn.h)
# instead of the full cuda-toolkit metapackage skips nsight-compute et al.
# — multi-GB packages that add nothing here.
conda install -y -c nvidia/label/cuda-12.8.1 cuda-nvcc cuda-cudart-dev cuda-cccl \
  libcublas-dev libcusparse-dev libcusolver-dev libcurand-dev libcufft-dev libnvjitlink-dev
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CC=gcc-10
export CXX=g++-10

nvcc --version

pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128

# Broad architecture coverage (Turing through Hopper) so the compiled
# extensions run on whatever GPU the image ends up on, not just the dev box.
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
cd "${GSPLAT_DIR}"
pip install --no-build-isolation -e .
pip install -r examples/requirements.txt --no-build-isolation

# segment_people.py calls the EasyTensorRT segmentation model over gRPC, and it
# runs in this env for torchvision, so it needs the Triton client here too.
pip install --no-cache-dir "tritonclient[grpc]"

rm -rf build "${GSPLAT_DIR}"/*.egg-info
pip cache purge
conda clean -afy
