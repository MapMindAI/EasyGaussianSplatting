#!/usr/bin/env bash
# Mask out people in a cube-map reconstruction's images (see cubemap_convert.sh)
# so gsplat doesn't train on passers-by. Run inside the container.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <cubemap_reconstruction_dir> [score_threshold] [dilation]" >&2
  exit 1
fi

CUBEMAP_DIR="$(cd "$1" && pwd)"
SCORE_THRESHOLD="${2:-0.5}"
DILATION="${3:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gsplat_env.sh"

python3 "${SCRIPT_DIR}/../mapping/segment_people.py" \
  "${CUBEMAP_DIR}/images" "${CUBEMAP_DIR}/masks" \
  --score-threshold "${SCORE_THRESHOLD}" --dilation "${DILATION}"
