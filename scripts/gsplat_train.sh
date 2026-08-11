#!/usr/bin/env bash
# Train a Gaussian Splatting model with gsplat from a cube-map reconstruction
# (see cubemap_convert.sh). Run inside the container.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <cubemap_reconstruction_dir> [iterations] [data_factor]" >&2
  exit 1
fi

CUBEMAP_DIR="$(cd "$1" && pwd)"
ITERATIONS="${2:-30000}"
DATA_FACTOR="${3:-1}"

RESULT_DIR="${CUBEMAP_DIR}/gsplat_output"

source /opt/miniconda3/etc/profile.d/conda.sh
# conda's cuda-nvcc activation hook references NVCC_PREPEND_FLAGS without a
# default, which trips `set -u` above.
set +u
conda activate gsplat
set -u
cd /opt/gsplat/examples
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
python3 simple_trainer.py default \
  --data_dir "${CUBEMAP_DIR}" \
  --data_factor "${DATA_FACTOR}" \
  --max_steps "${ITERATIONS}" \
  --eval_steps "${ITERATIONS}" \
  --save_steps "${ITERATIONS}" \
  --disable_viewer \
  --result_dir "${RESULT_DIR}"

echo "Model written to ${RESULT_DIR}"
