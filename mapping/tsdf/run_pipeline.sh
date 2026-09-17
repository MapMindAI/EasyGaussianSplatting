#!/usr/bin/env bash
# Render Gaussian-splat depths, then fuse them in the dedicated Open3D image.
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: $0 <reconstruction_dir> [gsplat.ply]" >&2
  exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/docker_common.sh"

RECONSTRUCTION_DIR="$(repo_relative_path "$1")"
TSDF_DOCKER_IMAGE="${TSDF_DOCKER_IMAGE:-ghcr.io/mapmindai/gaussiansplatting-tsdf:latest}"
MODEL_ARGUMENTS=()
if [ $# -eq 2 ]; then
  MODEL_ARGUMENTS=(--model-path "/workspace/$(repo_relative_path "$2")")
fi

docker run -i "${DOCKER_RUN_FLAGS[@]}" "${DOCKER_IMAGE}" \
  conda run --no-capture-output -n gsplat python3 -m mapping.tsdf.render_depth \
  "/workspace/${RECONSTRUCTION_DIR}" "${MODEL_ARGUMENTS[@]}"

docker run --rm -v "${REPO_ROOT}:/workspace" -w /workspace \
  "${TSDF_DOCKER_IMAGE}" \
  python3 -m mapping.tsdf.fuse "/workspace/${RECONSTRUCTION_DIR}"
