#!/usr/bin/env bash
# Activate the container's gsplat conda env. Source it; don't execute it.

# The Jetson image installs gsplat into the system Python, so there is no env
# to activate there.
if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
  source /opt/miniconda3/etc/profile.d/conda.sh
  # conda's cuda-nvcc activation hook references NVCC_PREPEND_FLAGS without a
  # default, which trips the callers' `set -u`.
  set +u
  conda activate gsplat
  set -u
fi
