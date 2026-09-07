#!/usr/bin/env bash
# Train one gsplat model per parameter configuration from a single cube-map
# reconstruction, so the configurations can be compared against each other
# (see summarize_gsplat_sweep.py for the report).
# Runs on the host (not inside the container) and drives `docker run` itself.
#
# Runs are sequential: the configurations are compared partly on training time
# and peak memory, which sharing a GPU would make meaningless. A configuration
# whose run already recorded evaluation metrics is skipped, so an interrupted
# sweep resumes where it stopped.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <cubemap_reconstruction_dir>" >&2
  exit 1
fi

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)"
source "${SCRIPTS_DIR}/docker_common.sh"

REL_CUBEMAP_DIR="$(repo_relative_path "$1")"
SWEEP_DIR="${REPO_ROOT}/${REL_CUBEMAP_DIR}/sweep"

# The configurations are the same parameter files the server sweep uses
# (mapping/benchmark/configurations), so the two drivers cannot drift apart.
CONFIGURATIONS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/configurations" && pwd)"

for configuration in "${CONFIGURATIONS_DIR}"/*.proto.txt; do
  name="$(basename "${configuration}" .proto.txt)"
  run_dir="${SWEEP_DIR}/${name}"

  if compgen -G "${run_dir}/stats/val_step*.json" >/dev/null; then
    echo "=== ${name}: already evaluated, skipping"
    continue
  fi
  # A run that exhausted the GPU fails the same way every time, so only an
  # interrupted run is worth resuming. Delete its directory to force a retry.
  if grep -qs '^exit_status [^0]' "${run_dir}/sweep_status.txt"; then
    echo "=== ${name}: failed previously, skipping (rm -r ${run_dir} to retry)"
    continue
  fi

  mkdir -p "${run_dir}"
  echo "=== ${name}"
  start_seconds="${SECONDS}"
  # A configuration that exhausts the GPU must not abort the rest of the sweep.
  status=0
  docker run "${DOCKER_RUN_FLAGS[@]}" \
    -e RESULT_DIR="/workspace/${REL_CUBEMAP_DIR}/sweep/${name}" \
    -e DATA_DIR="/workspace/${REL_CUBEMAP_DIR}" \
    -e PARAMETERS="/workspace/$(repo_relative_path "${configuration}")" \
    "${DOCKER_IMAGE}" \
    bash -c 'set -euo pipefail
source scripts/gsplat_env.sh
export PYTHONPATH="/workspace${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
python3 mapping/train_gsplat_with_masks.py default \
  --job_parameters "$PARAMETERS" \
  --data_dir "$DATA_DIR" \
  --save_ply \
  --antialiased \
  --disable_viewer \
  --result_dir "$RESULT_DIR"' >"${run_dir}/train.log" 2>&1 || status=$?

  printf 'exit_status %d\nwall_seconds %d\nconfiguration %s\n' \
    "${status}" "$((SECONDS - start_seconds))" "${configuration}" \
    >"${run_dir}/sweep_status.txt"
  if [ "${status}" -ne 0 ]; then
    echo "=== ${name}: FAILED (exit ${status}), see ${run_dir}/train.log"
  fi
done

echo "Sweep written to ${SWEEP_DIR}"
