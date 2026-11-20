#!/usr/bin/env bash
# Container: train a Gaussian Splat from a COLMAP workspace.
#
# Usage: scripts/gsplat_train.sh <workspace> [parameters.proto.txt] [result_dir] [trainer flags...]
#
# The workspace holds images/ and sparse/0/. The parameters file is layered over
# gsplat_server/config/gsplat_train_defaults.proto.txt; mask_people and mask_sky
# generate masks before training, then run_depth generates VGGT-Omega depths and
# their merged point cloud unless depths/ already holds some. Results default to
# <workspace>/gsplat_output_<YYYYmmdd_HHMMSS>, with the resolved parameters in
# cfg.proto.txt. Further flags go to
# train_gsplat_with_masks.py. A final PLY of 2 GiB or more is also bisected
# into ply/split/ until every part is smaller.
set -euo pipefail

cd "$(dirname "$0")/.."
WORKSPACE_PATH=${1:?usage: $0 <workspace> [parameters.proto.txt] [result_dir]}
PARAMETERS_PATH=${2:-gsplat_server/config/gsplat_train_drone.proto.txt}
RESULT_DIR=${3:-${WORKSPACE_PATH}/gsplat_output_$(date +%Y%m%d_%H%M%S)}
echo "Training output: ${RESULT_DIR}"

source scripts/gsplat_env.sh
export PYTHONPATH="$PWD"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "${RESULT_DIR}"
python3 -c 'import sys; from gsplat_server.parameters import load_parameters, parameters_to_complete_text
sys.stdout.write(parameters_to_complete_text(load_parameters(sys.argv[1])))' \
  "${PARAMETERS_PATH}" > "${RESULT_DIR}/cfg.proto.txt"

bash scripts/gsplat_segment.sh "${WORKSPACE_PATH}" "${PARAMETERS_PATH}"

if python3 -c 'import sys; from gsplat_server.parameters import load_parameters
sys.exit(not load_parameters(sys.argv[1]).run_depth)' "${PARAMETERS_PATH}"; then
  python3 -m mapping.vggt_omega.generate_depth \
    "${WORKSPACE_PATH}" "${WORKSPACE_PATH}/depths" \
    --mask-dir "${WORKSPACE_PATH}/masks" \
    --samples-per-image -1
fi

python3 mapping/train_gsplat_with_masks.py default \
  --job_parameters "${PARAMETERS_PATH}" \
  --data_dir "${WORKSPACE_PATH}" \
  --save_ply \
  --antialiased \
  --disable_viewer \
  --result_dir "${RESULT_DIR}" \
  "${@:4}" \
  2>&1 | tee "${RESULT_DIR}.log"

FINAL_PLY=$(ls -v "${RESULT_DIR}"/ply/point_cloud_*.ply 2>/dev/null | tail -n 1)
if [ -n "${FINAL_PLY}" ]; then
  python3 -m mapping.split_ply "${FINAL_PLY}" "${RESULT_DIR}/ply/split"
fi
