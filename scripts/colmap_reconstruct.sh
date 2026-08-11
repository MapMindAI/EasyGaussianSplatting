#!/usr/bin/env bash
# Reconstruct a scene from a stitched equirectangular video using COLMAP's
# panorama_sfm: reconstructs from a rig of virtual perspective views rendered
# from each panorama (avoiding SIFT's blind spot for spherical distortion),
# then reprojects the result back to a native EQUIRECTANGULAR camera per
# frame. Run inside the container.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <video_path> [frame_rate]" >&2
  exit 1
fi

INPUT_VIDEO="$1"
FRAME_RATE="${2:-2}"

VIDEO_DIR="$(dirname "${INPUT_VIDEO}")"
VIDEO_NAME="$(basename "${INPUT_VIDEO}")"
VIDEO_NAME="${VIDEO_NAME%.*}"
WORK_DIR="${VIDEO_DIR}/${VIDEO_NAME}_mapping"

IMAGE_DIR="${WORK_DIR}/images"
PANORAMA_SFM_DIR="${WORK_DIR}/panorama_sfm"
SPARSE_DIR="${WORK_DIR}/sparse"

if [ -s "${SPARSE_DIR}/0/cameras.bin" ]; then
  echo "${SPARSE_DIR}/0 already exists, skipping reconstruction"
  exit 0
fi

mkdir -p "${IMAGE_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 \
  "${SCRIPT_DIR}/../mapping/extract_images.py" \
  "${INPUT_VIDEO}" "${IMAGE_DIR}" --frame-rate "${FRAME_RATE}"

python3 /opt/colmap/panorama_sfm.py \
  --input_image_path "${IMAGE_DIR}" \
  --output_path "${PANORAMA_SFM_DIR}" \
  --use_gpu

mv "${PANORAMA_SFM_DIR}/sparse_equirectangular" "${SPARSE_DIR}"
rm -rf "${PANORAMA_SFM_DIR}"

echo "Reconstruction written to ${SPARSE_DIR}"
