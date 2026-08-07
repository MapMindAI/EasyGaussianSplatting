#!/usr/bin/env bash
# End-to-end pipeline: stitch an Insta360 capture, extract frames, run COLMAP
# reconstruction, and train a Gaussian Splatting model (GGPS), all via the
# project's Docker image. Runs on the host (not inside the container) and
# drives `docker run` itself.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <input.insv> [frame_rate] [output_size] [gs_iterations] [gs_resolution]" >&2
  exit 1
fi

INPUT_INSV="$1"
FRAME_RATE="${2:-2}"
OUTPUT_SIZE="${3:-8000x4000}"
GS_ITERATIONS="${4:-30000}"
GS_RESOLUTION="${5:-1}"
DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/mapmindai/gaussiansplatting:latest}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSV_ABS="$(cd "$(dirname "${INPUT_INSV}")" && pwd)/$(basename "${INPUT_INSV}")"

case "${INSV_ABS}" in
  "${REPO_ROOT}"/*) ;;
  *)
    echo "${INPUT_INSV} must live under the repo checkout (${REPO_ROOT}) so the container can see it" >&2
    exit 1
    ;;
esac

REL_INSV="${INSV_ABS#"${REPO_ROOT}"/}"
REL_OUTPUT_DIR="${REL_INSV%.*}_reconstruction"
REL_OUTPUT_VIDEO="${REL_OUTPUT_DIR}/pano.mp4"
# Mirrors colmap_reconstruct.sh's work-dir rule: <video stem>_mapping.
REL_RECONSTRUCTION_DIR="${REL_OUTPUT_VIDEO%.*}_mapping"

mkdir -p "${REPO_ROOT}/${REL_OUTPUT_DIR}"

docker run --rm --gpus all \
  -e INPUT_INSV="/workspace/${REL_INSV}" \
  -e OUTPUT_VIDEO="/workspace/${REL_OUTPUT_VIDEO}" \
  -e OUTPUT_SIZE="${OUTPUT_SIZE}" \
  -e FRAME_RATE="${FRAME_RATE}" \
  -e RECONSTRUCTION_DIR="/workspace/${REL_RECONSTRUCTION_DIR}" \
  -e GS_ITERATIONS="${GS_ITERATIONS}" \
  -e GS_RESOLUTION="${GS_RESOLUTION}" \
  -v "${REPO_ROOT}:/workspace" -w /workspace "${DOCKER_IMAGE}" \
  bash -c 'set -euo pipefail
scripts/stitch_pano.sh
scripts/colmap_reconstruct.sh "$OUTPUT_VIDEO" "$FRAME_RATE"
scripts/ggps_train.sh "$RECONSTRUCTION_DIR" "$GS_ITERATIONS" "$GS_RESOLUTION"'

echo "Model written to ${REPO_ROOT}/${REL_RECONSTRUCTION_DIR}/ggps_output"
