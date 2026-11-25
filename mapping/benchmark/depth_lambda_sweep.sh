#!/usr/bin/env bash
# Train one model per depth weight, plus one without depth, through
# scripts/gsplat_train.sh (see doc/tools.md). Runs where gsplat_train.sh does.
# A weight is a constant depth_lambda_begin, or begin:final for one that
# decreases linearly to depth_lambda_final at refine_stop_iter.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <workspace> [parameters.proto.txt] [weight | begin:final ...]" >&2
  exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/sweep_common.sh"
start_sweep depth_lambda_sweep "$@"
DEPTH_LAMBDAS=("${@:3}")
if [ ${#DEPTH_LAMBDAS[@]} -eq 0 ]; then
  DEPTH_LAMBDAS=(0 0.001 0.003 0.01 0.03 0.1 0.05:0.01 0.1:0.01 0.2:0.01)
fi

for weight in "${DEPTH_LAMBDAS[@]}"; do
  final=0
  if [[ "${weight}" == *:* ]]; then
    final="${weight#*:}"
  fi
  train_sweep_run "depth_lambda_${weight/:/_to_}" "run_depth: true
depth_lambda_begin: ${weight%%:*}
depth_lambda_final: ${final}"
done
train_sweep_run no_depth 'run_depth: false'

echo "Sweep written to ${SWEEP_DIR}"
