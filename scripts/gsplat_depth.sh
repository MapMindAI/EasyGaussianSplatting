#!/usr/bin/env bash
# Container: generate VGGT-Omega depths and their merged point cloud in the
# active gsplat environment when any parameters file sets run_depth. Existing
# depths/ and depth_points.ply are kept.
set -euo pipefail

cd "$(dirname "$0")/.."
WORKSPACE_PATH=${1:?usage: $0 <workspace> <parameters.proto.txt> [parameters.proto.txt...]}
: "${2:?parameters file required}"

if [ ! -f gsplat_server/proto/gsplat_pb2.py ]; then
  bash gsplat_server/proto/build.sh
fi

if python3 -c 'import sys; from gsplat_server.parameters import load_parameters
sys.exit(not any(load_parameters(path).run_depth for path in sys.argv[1:]))' "${@:2}"; then
  python3 -m mapping.vggt_omega.generate_depth \
    "${WORKSPACE_PATH}" "${WORKSPACE_PATH}/depths" \
    --mask-dir "${WORKSPACE_PATH}/masks" \
    --samples-per-image -1
else
  echo "Depth disabled; skipping depth generation"
fi
