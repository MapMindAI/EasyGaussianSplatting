#!/usr/bin/env bash
# Stitch every Insta360 capture under a workspace into an equirectangular video.
# Runs on the host (not inside the container) and drives one `docker run` per
# capture.
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: $0 <workspace> [output_size]" >&2
  exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/docker_common.sh"

WORKSPACE_PATH="$(cd "$1" && pwd)"
repo_relative_path "${WORKSPACE_PATH}" >/dev/null
OUTPUT_SIZE="${2:-4000x2000}"

mapfile -d '' input_videos < <(
  find "${WORKSPACE_PATH}" -type f -iname '*.insv' -print0 | sort -z
)
if [ "${#input_videos[@]}" -eq 0 ]; then
  echo "No INSV files found under ${WORKSPACE_PATH}" >&2
  exit 1
fi

for input_video in "${input_videos[@]}"; do
  output_video="${input_video%.*}.mp4"
  relative_input="$(repo_relative_path "${input_video}")"
  relative_output="$(repo_relative_path "${output_video}")"

  docker run "${DOCKER_RUN_FLAGS[@]}" -e MODEL_ROOT_DIR="${MODEL_ROOT_DIR:-}" \
    -e STITCH_TIMEOUT_SECONDS="${STITCH_TIMEOUT_SECONDS:-}" \
    "${DOCKER_IMAGE}" bash scripts/stitch_video.sh \
    "/workspace/${relative_input}" "/workspace/${relative_output}" "${OUTPUT_SIZE}"
  echo "Stitched video written to ${REPO_ROOT}/${relative_output}"
done
