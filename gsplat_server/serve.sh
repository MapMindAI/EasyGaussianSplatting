#!/usr/bin/env bash
# Serve gsplat training over gRPC (see server.py). Run inside the container.
set -euo pipefail

JOBS_DIR="${1:-/workspace/data/gsplat_server}"
PORT="${2:-50051}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${REPO_ROOT}/scripts/gsplat_env.sh"

# The image installs gsplat at /opt/gsplat; third_party/gsplat is only source.
export GSPLAT_DIR="${GSPLAT_DIR:-/opt/gsplat}"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python3 "${SCRIPT_DIR}/server.py" --jobs-dir "${JOBS_DIR}" --port "${PORT}"
