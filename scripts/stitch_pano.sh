#!/usr/bin/env bash
# Stitch a raw Insta360 capture into an equirectangular video via insta360_media_stitcher.
# Run inside the container (see README "Using the tools").
set -euo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/lib"

INPUT_INSV="${INPUT_INSV:-data/VID_20260422_153814_00_004.insv}"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-data/VID_20260422_153814_00_004_pano.mp4}"
MODEL_ROOT_DIR="${MODEL_ROOT_DIR:-/EasyGaussianSplatting/data/sdk_dir}"
OUTPUT_SIZE="${OUTPUT_SIZE:-8000x4000}"

# Bounded: the SDK has been observed to deadlock indefinitely on Vulkan device
# init failure when a GPU is passed to the container but has no usable ICD.
timeout 1800 insta360_media_stitcher \
  -inputs "${INPUT_INSV}" \
  -output "${OUTPUT_VIDEO}" \
  -model_root_dir "${MODEL_ROOT_DIR}" \
  -stitch_type aistitch -enable_stitchfusion \
  -output_size "${OUTPUT_SIZE}" -bitrate 150000000 \
  -enable_h265_encoder -enable_flowstate -enable_directionlock

if [ ! -s "${OUTPUT_VIDEO}" ]; then
  echo "Stitching failed: ${OUTPUT_VIDEO} was not created" >&2
  exit 1
fi

# insta360_media_stitcher can exit 0 while writing a corrupt file (e.g. encoder
# init failure mid-run), so confirm a frame actually decodes before calling it done.
/opt/miniconda3/envs/gaussian_splatting/bin/python3 - "${OUTPUT_VIDEO}" <<'PY'
import sys
import cv2

path = sys.argv[1]
cap = cv2.VideoCapture(path)
ok, _ = cap.read()
if not ok:
    sys.exit(f"Stitching failed: could not decode any frame from {path}")
print(f"Stitched -> {path} ({int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames, "
      f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
PY
