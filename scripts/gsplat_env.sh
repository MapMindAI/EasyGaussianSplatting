#!/usr/bin/env bash
# Activate the container's gsplat conda env. Source it; don't execute it.

source /opt/miniconda3/etc/profile.d/conda.sh
# conda's cuda-nvcc activation hook references NVCC_PREPEND_FLAGS without a
# default, which trips the callers' `set -u`.
set +u
conda activate gsplat
set -u
