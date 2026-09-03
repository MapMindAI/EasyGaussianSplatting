#!/usr/bin/env bash
# Train a Gaussian Splatting model with gsplat from an existing cube-map
# reconstruction (see cubemap_convert.sh), via the project's Docker image.
# Runs on the host (not inside the container) and drives `docker run` itself.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <cubemap_reconstruction_dir> [iterations] [data_factor] [floater_reg_weight]" >&2
  exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/docker_common.sh"

REL_CUBEMAP_DIR="$(repo_relative_path "$1")"
ITERATIONS="${2:-30000}"
DATA_FACTOR="${3:-1}"
FLOATER_REG_WEIGHT="${4:-0.01}"

docker run "${DOCKER_RUN_FLAGS[@]}" "${DOCKER_IMAGE}" \
  scripts/gsplat_train.sh "/workspace/${REL_CUBEMAP_DIR}" \
  "${ITERATIONS}" "${DATA_FACTOR}" "${FLOATER_REG_WEIGHT}"

echo "Model written to ${REPO_ROOT}/${REL_CUBEMAP_DIR}/gsplat_output"
