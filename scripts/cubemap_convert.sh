#!/usr/bin/env bash
# Convert a COLMAP EQUIRECTANGULAR reconstruction (see colmap_reconstruct.sh)
# into a cube-map (PINHOLE) one gsplat can train on. Run inside the container.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <reconstruction_dir> [face_size] [faces]" >&2
  exit 1
fi

RECONSTRUCTION_DIR="$(cd "$1" && pwd)"
FACE_SIZE="${2:-1024}"
FACES="${3:-front,right,back,left,up,down}"

OUTPUT_DIR="${RECONSTRUCTION_DIR}_cubemap"

if [ -s "${OUTPUT_DIR}/sparse/0/cameras.txt" ]; then
  echo "${OUTPUT_DIR} already exists, skipping conversion"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/../mapping/equirect_to_cubemap.py" \
  "${RECONSTRUCTION_DIR}" "${OUTPUT_DIR}" --face-size "${FACE_SIZE}" --faces "${FACES}"

echo "Cube-map reconstruction written to ${OUTPUT_DIR}"
