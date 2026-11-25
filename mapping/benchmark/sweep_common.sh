#!/usr/bin/env bash
# Shared steps for the gsplat_train.sh sweeps. Source it; don't execute it.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Set WORKSPACE_PATH, PARAMETERS_FILE, and SWEEP_DIR=<workspace>/<sweep_name>.
# Usage: start_sweep <sweep_name> <workspace> [parameters.proto.txt]
start_sweep() {
  WORKSPACE_PATH="$(cd "$2" && pwd)"
  PARAMETERS_FILE="$(realpath "${3:-${REPO_ROOT}/gsplat_server/config/gsplat_train_defaults.proto.txt}")"
  SWEEP_DIR="${WORKSPACE_PATH}/$1"
  mkdir -p "${SWEEP_DIR}"
}

# Train ${SWEEP_DIR}/<name> from PARAMETERS_FILE with <overrides> appended,
# unless it was already evaluated or failed before.
# Usage: train_sweep_run <name> <overrides>
train_sweep_run() {
  local name="$1" overrides="$2"
  local run_dir="${SWEEP_DIR}/$1" configuration="${SWEEP_DIR}/$1.proto.txt"
  local start_seconds="${SECONDS}" status=0
  if compgen -G "${run_dir}/stats/val_step*.json" >/dev/null; then
    echo "=== ${name}: already evaluated, skipping"
    return
  fi
  # A run that exhausted the GPU fails the same way every time.
  if grep -qs '^exit_status [^0]' "${run_dir}/sweep_status.txt"; then
    echo "=== ${name}: failed previously, skipping (rm -r ${run_dir} to retry)"
    return
  fi

  # text_format merges later fields over earlier ones, so the overrides go last.
  printf '%s\n\n%s\n' "$(cat "${PARAMETERS_FILE}")" "${overrides}" >"${configuration}"

  echo "=== ${name}"
  mkdir -p "${run_dir}"
  bash "${REPO_ROOT}/scripts/gsplat_train.sh" "${WORKSPACE_PATH}" "${configuration}" "${run_dir}" \
    || status=$?
  printf 'exit_status %d\nwall_seconds %d\nconfiguration %s\n' \
    "${status}" "$((SECONDS - start_seconds))" "${configuration}" >"${run_dir}/sweep_status.txt"
  if [ "${status}" -ne 0 ]; then
    echo "=== ${name}: FAILED (exit ${status}), see ${run_dir}.log"
  fi
}
