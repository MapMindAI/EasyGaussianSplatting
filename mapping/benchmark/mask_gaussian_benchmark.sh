#!/usr/bin/env bash
# Host: run the MaskGaussian tests in the image, then train and compare
# STRATEGY_MCMC, STRATEGY_DEFAULT, and STRATEGY_DEFAULT with mask_gaussians at
# two mask_lambda values on one cube-map reconstruction, each layered over the
# base parameters, and write summarize_gsplat_sweep.py's report.
#
# Set DOCKER_IMAGE to the image the Docker workflow built for this branch,
# e.g. ghcr.io/mapmindai/gaussiansplatting:pr-<number>.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <cubemap_reconstruction_dir> [base_parameters.proto.txt]" >&2
  exit 1
fi

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)"
source "${SCRIPTS_DIR}/docker_common.sh"

REL_CUBEMAP_DIR="$(repo_relative_path "$1")"
BASE_PARAMETERS="${2:-${REPO_ROOT}/gsplat_server/config/gsplat_train_defaults.proto.txt}"
export SWEEP_NAME=mask_gaussian_sweep
SWEEP_DIR="${REPO_ROOT}/${REL_CUBEMAP_DIR}/${SWEEP_NAME}"
CONFIGURATIONS_DIR="${SWEEP_DIR}/configurations"
echo "Image: ${DOCKER_IMAGE}"

echo "=== Tests"
docker run "${DOCKER_RUN_FLAGS[@]}" "${DOCKER_IMAGE}" bash -c 'set -euo pipefail
source scripts/gsplat_env.sh
pip install --quiet pytest pytest-check  # The gsplat pytest config loads pytest-check.
bash gsplat_server/proto/build.sh  # Generated against the protobuf in the image.
python3 -m pytest -q -p no:cacheprovider mapping/mask_gaussian_test.py gsplat_server/parameters_test.py
(cd /opt/gsplat && python3 -m pytest -q -p no:cacheprovider tests/test_masked_opacities.py)
# Not gating: the image lacks nerfacc and compiles only 3 and 4 colour
# channels, which some of these tests need; those fail the same way unmodified.
(cd /opt/gsplat && python3 -m pytest -q -p no:cacheprovider tests/test_basic.py \
  -k "rasterize_to_pixels or projection or isect or test_rasterization" | tail -n 30) || true'

# Each configuration is the base parameters with these lines appended; a later
# value overrides an earlier one.
mkdir -p "${CONFIGURATIONS_DIR}"
write_configuration() {
  local name="$1" overrides="$2"
  { cat "${BASE_PARAMETERS}"; printf '\n# %s\n%b\n' "${name}" "${overrides}"; } \
    >"${CONFIGURATIONS_DIR}/${name}.proto.txt"
}
write_configuration mcmc "strategy: STRATEGY_MCMC"
write_configuration default "strategy: STRATEGY_DEFAULT"
write_configuration default_mask_0.0005 "strategy: STRATEGY_DEFAULT\nmask_gaussians: true\nmask_lambda: 0.0005"
write_configuration default_mask_0.005 "strategy: STRATEGY_DEFAULT\nmask_gaussians: true\nmask_lambda: 0.005"

echo "=== Training"
bash "${REPO_ROOT}/mapping/benchmark/gsplat_sweep.sh" "$1" "${CONFIGURATIONS_DIR}"

echo "=== Report"
docker run "${DOCKER_RUN_FLAGS[@]}" "${DOCKER_IMAGE}" bash -c "set -euo pipefail
source scripts/gsplat_env.sh
python3 mapping/benchmark/summarize_gsplat_sweep.py \
  --runs-root ${REL_CUBEMAP_DIR}/${SWEEP_NAME} --output ${REL_CUBEMAP_DIR}/${SWEEP_NAME}/report.md"
echo "Report: ${SWEEP_DIR}/report.md; each run's log: ${SWEEP_DIR}/<name>/train.log"
