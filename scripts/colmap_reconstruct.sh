#!/usr/bin/env bash
# Reconstruct a scene from a stitched equirectangular video using COLMAP's
# native EQUIRECTANGULAR camera model. Run inside the container.
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
DATABASE_PATH="${WORK_DIR}/database.db"
SPARSE_DIR="${WORK_DIR}/sparse"

if [ -s "${SPARSE_DIR}/0/cameras.bin" ]; then
  echo "${SPARSE_DIR}/0 already exists, skipping reconstruction"
  exit 0
fi

mkdir -p "${IMAGE_DIR}" "${SPARSE_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 \
  "${SCRIPT_DIR}/../mapping/extract_images.py" \
  "${INPUT_VIDEO}" "${IMAGE_DIR}" --frame-rate "${FRAME_RATE}"

colmap feature_extractor \
  --database_path "${DATABASE_PATH}" \
  --image_path "${IMAGE_DIR}" \
  --ImageReader.camera_model EQUIRECTANGULAR \
  --ImageReader.single_camera 1 \
  --FeatureExtraction.use_gpu 0

colmap sequential_matcher \
  --database_path "${DATABASE_PATH}" \
  --FeatureMatching.use_gpu 0

colmap mapper \
  --database_path "${DATABASE_PATH}" \
  --image_path "${IMAGE_DIR}" \
  --output_path "${SPARSE_DIR}"

echo "Reconstruction written to ${SPARSE_DIR}"
