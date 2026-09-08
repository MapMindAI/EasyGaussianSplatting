#!/usr/bin/env bash
# The whole pipeline in one command: stitch an Insta360 capture into an
# equirectangular video, reconstruct it as a cube-map rig with learned features
# and global SfM, mask out people, and train a Gaussian Splatting model. Runs
# on the host (not inside the container) and drives one `docker run` through
# every stage.
#
# Reconstruction infers against a Triton server serving superpoint, lightglue,
# and salad; TRITON_URL locates it (see doc/panorama_mapping.md).
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <input.insv> [frame_rate] [output_size] [face_size] [parameters.proto.txt]" >&2
  exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/docker_common.sh"

REL_INSV="$(repo_relative_path "$1")"
# frame_rate and face_size stay empty unless given, so mapping_pipeline.py
# holds their defaults; output_size is the stitcher's own.
FRAME_RATE="${2:-}"
OUTPUT_SIZE="${3:-8000x4000}"
FACE_SIZE="${4:-}"
PARAMETERS_FILE="${5:-${REPO_ROOT}/gsplat_server/config/gsplat_train_defaults.proto.txt}"
REL_PARAMETERS_FILE="$(repo_relative_path "${PARAMETERS_FILE}")"
REL_OUTPUT_DIR="${REL_INSV%.*}_reconstruction"
REL_OUTPUT_VIDEO="${REL_OUTPUT_DIR}/pano.mp4"
# The mapping workspace: images/<face>/, database.db, and sparse/0/.
REL_RECONSTRUCTION_DIR="${REL_OUTPUT_VIDEO%.*}_mapping"

mkdir -p "${REPO_ROOT}/${REL_OUTPUT_DIR}"

docker run -i "${DOCKER_RUN_FLAGS[@]}" \
  -e INPUT_INSV="/workspace/${REL_INSV}" \
  -e OUTPUT_VIDEO="/workspace/${REL_OUTPUT_VIDEO}" \
  -e OUTPUT_SIZE="${OUTPUT_SIZE}" \
  -e FRAME_RATE="${FRAME_RATE}" \
  -e RECONSTRUCTION_DIR="/workspace/${REL_RECONSTRUCTION_DIR}" \
  -e FACE_SIZE="${FACE_SIZE}" \
  -e TRITON_URL="${TRITON_URL}" \
  -e PARAMETERS_FILE="/workspace/${REL_PARAMETERS_FILE}" \
  -e MODEL_ROOT_DIR="${MODEL_ROOT_DIR:-/EasyGaussianSplatting/data/sdk_dir}" \
  "${DOCKER_IMAGE}" bash -s <<'CONTAINER'
set -euo pipefail

# The repo root, so mapping/ can import gsplat_server for the parameter schema.
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

# --- 1. Stitch the capture into an equirectangular video --------------------
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/lib"

# Exit 0 iff the video decodes a frame and matches the requested size. The
# decode check matters even for pre-existing files: insta360_media_stitcher can
# exit 0 while writing a corrupt file (e.g. encoder init failure mid-run).
check_video() {
  python3 - "$1" "$2" <<'PY'
import sys
import cv2

path, output_size = sys.argv[1], sys.argv[2]
width, height = (int(x) for x in output_size.split("x"))
cap = cv2.VideoCapture(path)
ok, _ = cap.read()
if not ok:
    sys.exit(f"could not decode any frame from {path}")
actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
if actual != (width, height):
    sys.exit(f"{path} is {actual[0]}x{actual[1]}, expected {output_size}")
print(f"{path}: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames, {actual[0]}x{actual[1]}")
PY
}

if [ -s "${OUTPUT_VIDEO}" ] && check_video "${OUTPUT_VIDEO}" "${OUTPUT_SIZE}" 2>/dev/null; then
  echo "${OUTPUT_VIDEO} already exists at ${OUTPUT_SIZE}, skipping stitching"
else
  # Bounded: the SDK has been observed to deadlock indefinitely on Vulkan device
  # init failure when a GPU is passed to the container but has no usable ICD.
  timeout 1800 insta360_media_stitcher \
    -inputs "${INPUT_INSV}" \
    -output "${OUTPUT_VIDEO}" \
    -model_root_dir "${MODEL_ROOT_DIR}" \
    -stitch_type aistitch -enable_stitchfusion \
    -output_size "${OUTPUT_SIZE}" -bitrate 150000000 \
    -enable_h265_encoder -enable_flowstate -enable_directionlock

  check_video "${OUTPUT_VIDEO}" "${OUTPUT_SIZE}" || {
    echo "Stitching failed: ${OUTPUT_VIDEO} is not a usable ${OUTPUT_SIZE} video" >&2
    exit 1
  }
fi

# --- 2. Reconstruct it as a cube-map rig ------------------------------------
MAPPING_ARGUMENTS=(--triton-url "${TRITON_URL}")
if [ -n "${FRAME_RATE}" ]; then MAPPING_ARGUMENTS+=(--frame-rate "${FRAME_RATE}"); fi
if [ -n "${FACE_SIZE}" ]; then MAPPING_ARGUMENTS+=(--face-size "${FACE_SIZE}"); fi

python3 -m mapping.mapping_pipeline \
  --video_path "${OUTPUT_VIDEO}" --workspace_path "${RECONSTRUCTION_DIR}" \
  "${MAPPING_ARGUMENTS[@]}"

# --- 3. Mask out people, then train ----------------------------------------
# Both need the gsplat conda env, which the mapping stage above must not use:
# pycolmap lives in the base env.
source scripts/gsplat_env.sh

python3 mapping/segment_people.py \
  "${RECONSTRUCTION_DIR}/images" "${RECONSTRUCTION_DIR}/masks" \
  --score-threshold 0.5 --dilation 8

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
python3 mapping/train_gsplat_with_masks.py default \
  --job_parameters "${PARAMETERS_FILE}" \
  --data_dir "${RECONSTRUCTION_DIR}" \
  --save_ply \
  --antialiased \
  --disable_viewer \
  --result_dir "${RECONSTRUCTION_DIR}/gsplat_output"
CONTAINER

echo "Model written to ${REPO_ROOT}/${REL_RECONSTRUCTION_DIR}/gsplat_output"
