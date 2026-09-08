#!/usr/bin/env bash
# Stitch an Insta360 capture into an equirectangular video. Runs on the host
# (not inside the container) and drives one `docker run`. This is the stitching
# stage of scripts/run_pipeline.sh on its own, for when you want the panorama
# video without reconstructing or training from it.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <input.insv> [output.mp4] [output_size]" >&2
  exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/docker_common.sh"

REL_INSV="$(repo_relative_path "$1")"
OUTPUT_VIDEO="${2:-${1%.*}_pano.mp4}"
OUTPUT_SIZE="${3:-8000x4000}"

mkdir -p "$(dirname "${OUTPUT_VIDEO}")"
REL_OUTPUT_VIDEO="$(repo_relative_path "${OUTPUT_VIDEO}")"

docker run -i "${DOCKER_RUN_FLAGS[@]}" \
  -e INPUT_INSV="/workspace/${REL_INSV}" \
  -e OUTPUT_VIDEO="/workspace/${REL_OUTPUT_VIDEO}" \
  -e OUTPUT_SIZE="${OUTPUT_SIZE}" \
  -e MODEL_ROOT_DIR="${MODEL_ROOT_DIR:-/EasyGaussianSplatting/data/sdk_dir}" \
  "${DOCKER_IMAGE}" bash scripts/stitch_video.sh

echo "Stitched video written to ${REPO_ROOT}/${REL_OUTPUT_VIDEO}"
