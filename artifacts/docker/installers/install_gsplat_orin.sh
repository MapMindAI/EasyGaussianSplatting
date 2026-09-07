#!/usr/bin/env bash
# Install gsplat into the Jetson image's system Python. Unlike the x86 image's
# artifacts/docker/installers/install_gsplat.sh there is no conda env and no
# separate CUDA toolkit: JetPack ships CUDA 12.6 and a Python 3.10 already.
# /opt/gsplat is expected to already hold the third_party/gsplat submodule
# checkout (copied in by the Dockerfile's gsplatsrc build context).
set -e

GSPLAT_DIR="${GSPLAT_DIR:-/opt/gsplat}"

# PyPI has no aarch64 torch build that targets Jetson (its aarch64 wheels are
# server-GRACE builds) and no aarch64 pycolmap at all, so these three come from
# NVIDIA's Jetson index by direct URL -- pinning by name there instead would let
# pip pick PyPI's same-versioned wheel, which ranks higher by platform tag.
JETSON_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126/+f"
TORCH_WHEEL="${JETSON_INDEX}/02f/de421eabbf626/torch-2.9.1-cp310-cp310-linux_aarch64.whl"
TORCHVISION_WHEEL="${JETSON_INDEX}/d5b/caaf709f11750/torchvision-0.24.1-cp310-cp310-linux_aarch64.whl"
PYCOLMAP_WHEEL="${JETSON_INDEX}/03b/9a0699b153f0f/pycolmap-3.13.0.dev0-cp310-cp310-linux_aarch64.whl"

export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
nvcc --version

pip3 install --no-cache-dir --upgrade pip setuptools wheel
pip3 install --no-cache-dir "${TORCH_WHEEL}" "${TORCHVISION_WHEEL}" "${PYCOLMAP_WHEEL}"
python3 -c "import torch; assert torch.version.cuda, 'installed a CPU-only torch build'"

# 8.7 is the AGX Orin's compute capability; the x86 image's broad list would only
# lengthen an already long aarch64 build.
export TORCH_CUDA_ARCH_LIST="8.7"
cd "${GSPLAT_DIR}"
pip3 install --no-cache-dir --no-build-isolation -e .

# pycolmap is installed above from the Jetson index; ppisp and fused-bilagrid
# back --post_processing modes the training script never selects and have no
# aarch64 build.
grep -vE '^(pycolmap|ppisp|.*fused-bilagrid)' examples/requirements.txt > /tmp/gsplat-requirements.txt
pip3 install --no-cache-dir --no-build-isolation -r /tmp/gsplat-requirements.txt

pip3 install --no-cache-dir grpcio

rm -rf build "${GSPLAT_DIR}"/*.egg-info /tmp/gsplat-requirements.txt
pip3 cache purge
