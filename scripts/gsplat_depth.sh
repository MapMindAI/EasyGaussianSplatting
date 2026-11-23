#!/usr/bin/env bash
# Container: generate VGGT-Omega depths and their merged point cloud in the
# active gsplat environment when any parameters file sets run_depth. Existing
# depth files are kept and only missing images are predicted.
set -euo pipefail

cd "$(dirname "$0")/.."
WORKSPACE_PATH=${1:?usage: $0 <workspace> <parameters.proto.txt> [parameters.proto.txt...]}
: "${2:?parameters file required}"

if [ ! -f gsplat_server/proto/gsplat_pb2.py ]; then
  bash gsplat_server/proto/build.sh
fi

# The first parameters file that sets run_depth supplies the voxel factor and
# the confidence floor.
DEPTH_OPTIONS=$(python3 -c 'import sys; from gsplat_server.parameters import load_parameters
parameters = [load_parameters(path) for path in sys.argv[1:]]
print(next((f"{p.depth_voxel_factor:.7g} {p.depth_min_confidence:.7g}" for p in parameters if p.run_depth), ""))' "${@:2}")
if [ -n "${DEPTH_OPTIONS}" ]; then
  read -r VOXEL_FACTOR MIN_CONFIDENCE <<< "${DEPTH_OPTIONS}"
  python3 -m mapping.vggt_omega.generate_depth \
    "${WORKSPACE_PATH}" "${WORKSPACE_PATH}/depths" \
    --mask-dir "${WORKSPACE_PATH}/masks" \
    --voxel-factor "${VOXEL_FACTOR}" \
    --min-confidence "${MIN_CONFIDENCE}"
else
  echo "Depth disabled; skipping depth generation"
fi
