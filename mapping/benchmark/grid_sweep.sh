#!/usr/bin/env bash
# Train one model per combination of JobParameters values through
# scripts/gsplat_train.sh (see doc/tools.md). Runs where gsplat_train.sh does.
#
# Example: grid_sweep.sh <workspace> opacities_lr=0.1,0.05,0.01 min_opacity=0.01,0.1
# A value of a repeated field lists its entries joined by +: mcmc_prune_steps=10000+20000,15000
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <workspace> [parameters.proto.txt] field=value[,value...] ..." >&2
  exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/sweep_common.sh"
WORKSPACE_ARGUMENT="$1"
shift
PARAMETERS_ARGUMENT=()
if [[ "$1" != *=* ]]; then
  PARAMETERS_ARGUMENT=("$1")
  shift
fi
if [ $# -eq 0 ]; then
  echo "No field=value[,value...] axes given" >&2
  exit 1
fi
AXES=("$@")
start_sweep grid_sweep "${WORKSPACE_ARGUMENT}" "${PARAMETERS_ARGUMENT[@]}"
SWEEP_RUN_COUNT=1
for axis in "${AXES[@]}"; do
  IFS=',' read -ra values <<<"${axis#*=}"
  SWEEP_RUN_COUNT=$((SWEEP_RUN_COUNT * ${#values[@]}))
done

# Usage: train_combinations <axis index> <name so far> <overrides so far>
train_combinations() {
  local index="$1" name="$2" overrides="$3"
  if [ "${index}" -eq "${#AXES[@]}" ]; then
    train_sweep_run "${name}" "${overrides}"
    return
  fi
  local field="${AXES[index]%%=*}" values value
  IFS=',' read -ra values <<<"${AXES[index]#*=}"
  for value in "${values[@]}"; do
    local setting="${value}"
    [[ "${value}" == *+* ]] && setting="[${value//+/, }]"
    train_combinations "$((index + 1))" "${name:+${name}__}${field}_${value}" \
      "${overrides:+${overrides}
}${field}: ${setting}"
  done
}

train_combinations 0 "" ""

echo "Sweep written to ${SWEEP_DIR}"
