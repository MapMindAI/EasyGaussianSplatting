#!/usr/bin/env bash
# Train a Gaussian Splatting model with gsplat from a cube-map reconstruction
# (see cubemap_convert.sh). Run inside the container.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <cubemap_reconstruction_dir> [iterations] [data_factor] [floater_reg_weight]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gsplat_train_defaults.sh"

CUBEMAP_DIR="$(cd "$1" && pwd)"
ITERATIONS="${2:-30000}"
DATA_FACTOR="${3:-1}"
# Applied to both opacity_reg and scale_reg, matching gsplat's own "mcmc"
# preset -- penalizes low-opacity/oversized Gaussians (floaters).
FLOATER_REG_WEIGHT="${4:-${GSPLAT_FLOATER_REG_WEIGHT}}"

RESULT_DIR="${CUBEMAP_DIR}/gsplat_output"

source "${SCRIPT_DIR}/gsplat_env.sh"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
python3 "${SCRIPT_DIR}/../mapping/train_gsplat_with_masks.py" default \
  --data_dir "${CUBEMAP_DIR}" \
  --data_factor "${DATA_FACTOR}" \
  --max_steps "${ITERATIONS}" \
  --eval_steps "${ITERATIONS}" \
  --save_steps "${ITERATIONS}" \
  --ply_steps "${ITERATIONS}" \
  --save_ply \
  --pose_opt \
  --antialiased \
  --opacity_reg "${FLOATER_REG_WEIGHT}" \
  --scale_reg "${FLOATER_REG_WEIGHT}" \
  --disable_viewer \
  --result_dir "${RESULT_DIR}" \
  --strategy.grow-grad2d "${GSPLAT_GROW_GRAD2D}" \
  ${GSPLAT_STRATEGY_OPTIONS}

echo "Model written to ${RESULT_DIR}"
