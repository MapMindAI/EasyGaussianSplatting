#!/usr/bin/env bash
# Re-evaluate every run of a sweep from its final checkpoint with a new
# test_every, into <sweep_dir>/<run>/renders_test_every_<N>/ (see doc/tools.md).
# Runs where gsplat_train.sh does; the workspace is the sweep directory's parent.
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <sweep_dir> <test_every>" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SWEEP_DIR="$(cd "$1" && pwd)"
TEST_EVERY="$2"
WORKSPACE_PATH="$(dirname "${SWEEP_DIR}")"

cd "${REPO_ROOT}"
source scripts/gsplat_env.sh
export PYTHONPATH="${REPO_ROOT}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

RUN_DIRS=("${SWEEP_DIR}"/*/)
index=0
for run_dir in "${RUN_DIRS[@]}"; do
  index=$((index + 1))
  name="[${index}/${#RUN_DIRS[@]}] $(basename "${run_dir}")"
  # sweep_status.txt is written once training ends, so this also skips a run in progress.
  if ! grep -qs '^exit_status 0$' "${run_dir}sweep_status.txt"; then
    echo "=== ${name}: failed or unfinished, skipping"
    continue
  fi
  checkpoint="$(ls -v "${run_dir}"ckpts/ckpt_*_rank0.pt 2>/dev/null | tail -n 1)"
  if [ -z "${checkpoint}" ] || [ ! -f "${run_dir}cfg.proto.txt" ]; then
    echo "=== ${name}: no checkpoint or cfg.proto.txt, skipping"
    continue
  fi
  output_dir="${run_dir}renders_test_every_${TEST_EVERY}"
  if compgen -G "${output_dir}/stats/val_step*.json" >/dev/null; then
    echo "=== ${name}: already evaluated, skipping"
    continue
  fi

  echo "=== ${name}"
  mkdir -p "${output_dir}"
  printf '%s\n\ntest_every: %s\n' \
    "$(cat "${run_dir}cfg.proto.txt")" "${TEST_EVERY}" >"${output_dir}/cfg.proto.txt"
  python3 mapping/train_gsplat_with_masks.py default \
    --job_parameters "${output_dir}/cfg.proto.txt" \
    --data_dir "${WORKSPACE_PATH}" \
    --antialiased \
    --disable_viewer \
    --disable_video \
    --ckpt "${checkpoint}" \
    --result_dir "${output_dir}" \
    2>&1 | tee "${output_dir}/evaluate.log" \
    || echo "=== ${name}: FAILED, see ${output_dir}/evaluate.log"
  # simple_trainer creates these for training; evaluation leaves them empty.
  find "${output_dir}" -mindepth 1 -type d -empty -delete
done

echo "Evaluations written to ${SWEEP_DIR}/*/renders_test_every_${TEST_EVERY}"
