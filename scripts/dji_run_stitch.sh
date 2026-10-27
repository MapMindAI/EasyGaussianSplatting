#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
  echo "Usage: $0 <workspace> [frame_rate] [face_size]" >&2
  exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/docker_common.sh"
workspace="$(repo_relative_path "$1")"
arguments=(--frame-rate "${2:-2}")
if [ $# -eq 3 ]; then arguments+=(--face-size "$3"); fi

docker run "${DOCKER_RUN_FLAGS[@]}" "${DOCKER_IMAGE}" \
  python3 -m mapping.dji.stitch "/workspace/${workspace}" "${arguments[@]}"
