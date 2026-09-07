#!/usr/bin/env bash
# Mask out people and train a Gaussian Splatting model with gsplat from an
# existing cube-map reconstruction (see cubemap_convert.sh), via the project's
# Docker image.
# Runs on the host (not inside the container) and drives `docker run` itself.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <cubemap_reconstruction_dir> [parameters.proto.txt]" >&2
  exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/docker_common.sh"

REL_CUBEMAP_DIR="$(repo_relative_path "$1")"
PARAMETERS_FILE="${2:-gsplat_server/config/gsplat_train_defaults.proto.txt}"

docker run "${DOCKER_RUN_FLAGS[@]}" \
  -e CUBEMAP_DIR="/workspace/${REL_CUBEMAP_DIR}" \
  -e PARAMETERS_FILE="${PARAMETERS_FILE}" \
  "${DOCKER_IMAGE}" \
  bash -c 'set -euo pipefail
scripts/segment_people.sh "$CUBEMAP_DIR"
scripts/gsplat_train.sh "$CUBEMAP_DIR" "$PARAMETERS_FILE"'

echo "Model written to ${REPO_ROOT}/${REL_CUBEMAP_DIR}/gsplat_output"
