#!/usr/bin/env bash
# Train a Gaussian Splatting model with gsplat from a cube-map reconstruction
# (see cubemap_convert.sh). Run inside the container.
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <cubemap_reconstruction_dir> <parameters.proto.txt>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUBEMAP_DIR="$(cd "$1" && pwd)"
PARAMETERS_FILE="$2"

RESULT_DIR="${CUBEMAP_DIR}/gsplat_output"

source "${SCRIPT_DIR}/gsplat_env.sh"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
python3 "${SCRIPT_DIR}/../mapping/train_gsplat_with_masks.py" default \
  --job_parameters "${PARAMETERS_FILE}" \
  --data_dir "${CUBEMAP_DIR}" \
  --save_ply \
  --antialiased \
  --disable_viewer \
  --result_dir "${RESULT_DIR}"

echo "Model written to ${RESULT_DIR}"
