#!/usr/bin/env bash
# Train one model per anisotropy_reg_weight through scripts/gsplat_train.sh
# (see doc/tools.md). Runs where gsplat_train.sh does. max_scale_ratio comes
# from the parameters file.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <workspace> [parameters.proto.txt] [weight ...]" >&2
  exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/sweep_common.sh"
start_sweep anisotropy_sweep "$@"
WEIGHTS=("${@:3}")
if [ ${#WEIGHTS[@]} -eq 0 ]; then
  WEIGHTS=(0 0.01 0.03 0.1 0.3 1)
fi

for weight in "${WEIGHTS[@]}"; do
  train_sweep_run "anisotropy_reg_weight_${weight}" "anisotropy_reg_weight: ${weight}"
done

echo "Sweep written to ${SWEEP_DIR}"
