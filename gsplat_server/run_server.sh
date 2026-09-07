#!/usr/bin/env bash
# Start the gsplat training server via Docker.
# Runs on the host (not inside the container) and drives `docker run` itself.
#
# Defaults suit an x86 host with the standard image. On a Jetson, whose JetPack
# runtime is wired up as a named runtime rather than through `--gpus`:
#   DOCKER_IMAGE=ghcr.io/mapmindai/gaussiansplatting-jetson:latest \
#   DOCKER_GPU_FLAGS="--runtime nvidia" gsplat_server/run_server.sh
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../scripts/docker_common.sh"

PORT="${1:-50051}"
JOBS_DIR="${2:-data/gsplat_server}"
mkdir -p "${JOBS_DIR}"
REL_JOBS_DIR="$(repo_relative_path "${JOBS_DIR}")"

docker run "${DOCKER_RUN_FLAGS[@]}" -p "${PORT}:${PORT}" \
  "${DOCKER_IMAGE}" \
  gsplat_server/serve.sh "/workspace/${REL_JOBS_DIR}" "${PORT}"
