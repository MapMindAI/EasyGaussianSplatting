#!/usr/bin/env bash
# Train one model per depth_lambda_begin, plus one without depth, through
# scripts/gsplat_train.sh (see doc/tools.md). Runs where gsplat_train.sh does.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <workspace> [parameters.proto.txt] [depth_lambda_begin...]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_PATH="$(cd "$1" && pwd)"
PARAMETERS_FILE="$(realpath "${2:-${REPO_ROOT}/gsplat_server/config/gsplat_train_defaults.proto.txt}")"
DEPTH_LAMBDAS=("${@:3}")
if [ ${#DEPTH_LAMBDAS[@]} -eq 0 ]; then
  DEPTH_LAMBDAS=(0 0.001 0.003 0.01 0.03 0.1)
fi
SWEEP_DIR="${WORKSPACE_PATH}/depth_lambda_sweep"
mkdir -p "${SWEEP_DIR}"

for name in "${DEPTH_LAMBDAS[@]/#/depth_lambda_}" no_depth; do
  run_dir="${SWEEP_DIR}/${name}"
  configuration="${run_dir}.proto.txt"
  if compgen -G "${run_dir}/stats/val_step*.json" >/dev/null; then
    echo "=== ${name}: already evaluated, skipping"
    continue
  fi
  # A run that exhausted the GPU fails the same way every time.
  if grep -qs '^exit_status [^0]' "${run_dir}/sweep_status.txt"; then
    echo "=== ${name}: failed previously, skipping (rm -r ${run_dir} to retry)"
    continue
  fi

  # text_format merges later fields over earlier ones, so the overrides go last.
  if [ "${name}" = no_depth ]; then
    overrides='run_depth: false'
  else
    overrides="run_depth: true
depth_lambda_begin: ${name#depth_lambda_}"
  fi
  printf '%s\n\n%s\n' "$(cat "${PARAMETERS_FILE}")" "${overrides}" >"${configuration}"

  echo "=== ${name}"
  mkdir -p "${run_dir}"
  start_seconds="${SECONDS}"
  status=0
  bash "${REPO_ROOT}/scripts/gsplat_train.sh" "${WORKSPACE_PATH}" "${configuration}" "${run_dir}" \
    || status=$?
  printf 'exit_status %d\nwall_seconds %d\nconfiguration %s\n' \
    "${status}" "$((SECONDS - start_seconds))" "${configuration}" >"${run_dir}/sweep_status.txt"
  if [ "${status}" -ne 0 ]; then
    echo "=== ${name}: FAILED (exit ${status}), see ${run_dir}.log"
  fi
done

echo "Sweep written to ${SWEEP_DIR}"
