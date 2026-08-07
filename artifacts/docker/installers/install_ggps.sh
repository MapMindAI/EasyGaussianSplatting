#!/usr/bin/env bash
# GGPS (github.com/Insta360-Research-Team/GGPS) needs a Python/PyTorch/CUDA
# stack the base conda env doesn't have, so it gets its own conda env and its
# own CUDA toolkit (via conda, since the base image ships CUDA runtime libs
# but no nvcc). Released under CC BY-NC 4.0 (non-commercial use only).
set -e

GGPS_DIR="/opt/ggps"
# Latest upstream commit as of when this installer was written; bump by
# re-running `git ls-remote https://github.com/Insta360-Research-Team/GGPS.git HEAD`.
GGPS_COMMIT="9f1f007ed07994e3583cc31dafb5de460832eeb0"

mkdir -p "${GGPS_DIR}"
curl -fsSL "https://github.com/Insta360-Research-Team/GGPS/archive/${GGPS_COMMIT}.tar.gz" \
  | tar xz -C "${GGPS_DIR}" --strip-components=1
cd "${GGPS_DIR}"

source /opt/miniconda3/etc/profile.d/conda.sh
conda create -yn ggps python=3.9 pip
conda activate ggps

# The unlabeled "nvidia" channel's cuda-toolkit=12.1 spec resolves to whatever
# nvcc is newest (13.x as of this writing) rather than pinning to 12.1, which
# then mismatches the cu121 torch wheel below. The release-labeled channel
# pins the actual CUDA 12.1.0 release. Installing just the pieces the two
# extensions' headers need (nvcc, cudart, cub/thrust) instead of the full
# cuda-toolkit metapackage skips nsight-compute et al. — multi-GB packages
# that added nothing here and kept failing to download intact.
conda install -y -c nvidia/label/cuda-12.1.0 cuda-nvcc cuda-cudart-dev cuda-cccl
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CC=gcc-10
export CXX=g++-10

nvcc --version

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Broad architecture coverage (Turing through Hopper) so the compiled
# extensions run on whatever GPU the image ends up on, not just the dev box.
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization ./submodules/simple-knn

# The in-tree extension builds leave multi-arch object files behind — several
# hundred MB that would otherwise ship in the image layer.
rm -rf submodules/*/build submodules/*/*.egg-info
pip cache purge
conda clean -afy
