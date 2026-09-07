#!/usr/bin/env bash
# End-to-end pipeline: stitch an Insta360 capture, extract frames, run COLMAP
# reconstruction, convert it to a cube-map model, and train a Gaussian
# Splatting model (gsplat), all via the project's Docker image. Runs on the
# host (not inside the container) and drives `docker run` itself.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <input.insv> [frame_rate] [output_size] [face_size] [parameters.proto.txt]" >&2
  exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/docker_common.sh"

REL_INSV="$(repo_relative_path "$1")"
FRAME_RATE="${2:-2}"
OUTPUT_SIZE="${3:-8000x4000}"
FACE_SIZE="${4:-1024}"
PARAMETERS_FILE="${5:-${REPO_ROOT}/gsplat_server/config/gsplat_train_defaults.proto.txt}"
REL_PARAMETERS_FILE="$(repo_relative_path "${PARAMETERS_FILE}")"
REL_OUTPUT_DIR="${REL_INSV%.*}_reconstruction"
REL_OUTPUT_VIDEO="${REL_OUTPUT_DIR}/pano.mp4"
# Mirrors colmap_reconstruct.sh's work-dir rule: <video stem>_mapping.
REL_RECONSTRUCTION_DIR="${REL_OUTPUT_VIDEO%.*}_mapping"

mkdir -p "${REPO_ROOT}/${REL_OUTPUT_DIR}"

docker run "${DOCKER_RUN_FLAGS[@]}" \
  -e INPUT_INSV="/workspace/${REL_INSV}" \
  -e OUTPUT_VIDEO="/workspace/${REL_OUTPUT_VIDEO}" \
  -e OUTPUT_SIZE="${OUTPUT_SIZE}" \
  -e FRAME_RATE="${FRAME_RATE}" \
  -e RECONSTRUCTION_DIR="/workspace/${REL_RECONSTRUCTION_DIR}" \
  -e FACE_SIZE="${FACE_SIZE}" \
  -e PARAMETERS_FILE="/workspace/${REL_PARAMETERS_FILE}" \
  "${DOCKER_IMAGE}" \
  bash -c 'set -euo pipefail
scripts/stitch_pano.sh
scripts/colmap_reconstruct.sh "$OUTPUT_VIDEO" "$FRAME_RATE"
scripts/cubemap_convert.sh "$RECONSTRUCTION_DIR" "$FACE_SIZE"
scripts/segment_people.sh "${RECONSTRUCTION_DIR}_cubemap"
scripts/gsplat_train.sh "${RECONSTRUCTION_DIR}_cubemap" "$PARAMETERS_FILE"'

echo "Model written to ${REPO_ROOT}/${REL_RECONSTRUCTION_DIR}_cubemap/gsplat_output"
