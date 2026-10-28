#!/usr/bin/env bash
# Container: train a Gaussian Splat from a COLMAP workspace.
#
# Usage: scripts/gsplat_train.sh <workspace> [parameters.proto.txt] [result_dir]
#
# The workspace holds images/ and sparse/0/. The parameters file is layered over
# gsplat_server/config/gsplat_train_defaults.proto.txt. Results default to
# <workspace>/gsplat_output.
set -euo pipefail

cd "$(dirname "$0")/.."
WORKSPACE_PATH=${1:?usage: $0 <workspace> [parameters.proto.txt] [result_dir]}
PARAMETERS_PATH=${2:-gsplat_server/config/gsplat_train_defaults.proto.txt}
RESULT_DIR=${3:-${WORKSPACE_PATH}/gsplat_output}

source scripts/gsplat_env.sh
export PYTHONPATH="$PWD"
export PYTORCH_ALLOC_CONF=expandable_segments:True

if [ ! -f gsplat_server/proto/gsplat_pb2.py ]; then
  bash gsplat_server/proto/build.sh
fi

python3 mapping/train_gsplat_with_masks.py default \
  --job_parameters "${PARAMETERS_PATH}" \
  --data_dir "${WORKSPACE_PATH}" \
  --save_ply \
  --antialiased \
  --disable_viewer \
  --result_dir "${RESULT_DIR}" \
  2>&1 | tee "${RESULT_DIR}.log"
