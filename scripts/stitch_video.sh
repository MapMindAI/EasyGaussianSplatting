#!/usr/bin/env bash
# Stitch an Insta360 capture into an equirectangular video with the AI-stitch
# settings we reconstruct from. Runs inside the container; scripts/run_stitch.sh
# and scripts/run_pipeline.sh drive it. Reads INPUT_INSV, OUTPUT_VIDEO,
# OUTPUT_SIZE, and MODEL_ROOT_DIR from the environment.
set -euo pipefail

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
  exit 0
fi

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
