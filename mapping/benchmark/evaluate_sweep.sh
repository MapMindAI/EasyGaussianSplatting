#!/usr/bin/env bash
# Re-evaluate every training run directly under <runs_dir> (a sweep directory,
# or a workspace holding gsplat_output_* runs) from its final checkpoint with a
# new test_every, into <run>/renders_test_every_<N>/ (see doc/tools.md).
# Runs where gsplat_train.sh does.
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <sweep_dir | workspace> <test_every>" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS_DIR="$(cd "$1" && pwd)"
TEST_EVERY="$2"
WORKSPACE_PATH="${RUNS_DIR}"
if [ ! -d "${WORKSPACE_PATH}/sparse" ]; then
  WORKSPACE_PATH="$(dirname "${RUNS_DIR}")"
fi

cd "${REPO_ROOT}"
source scripts/gsplat_env.sh
export PYTHONPATH="${REPO_ROOT}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# A training run is a directory with ckpts/; the workspace's own folders have none.
RUN_DIRS=()
for run_dir in "${RUNS_DIR}"/*/; do
  if [ -d "${run_dir}ckpts" ]; then
    RUN_DIRS+=("${run_dir}")
  fi
done
index=0
for run_dir in "${RUN_DIRS[@]}"; do
  index=$((index + 1))
  name="[${index}/${#RUN_DIRS[@]}] $(basename "${run_dir}")"
  # The checkpoint is saved only at the last step, so this skips runs that failed or are still training.
  checkpoint="$(ls -v "${run_dir}"ckpts/ckpt_*_rank0.pt 2>/dev/null | tail -n 1 || true)"
  if [ -z "${checkpoint}" ]; then
    echo "=== ${name}: failed or unfinished, skipping"
    continue
  fi
  if [ ! -f "${run_dir}cfg.proto.txt" ]; then
    echo "=== ${name}: no cfg.proto.txt, skipping"
    continue
  fi
  output_dir="${run_dir}renders_test_every_${TEST_EVERY}"
  if compgen -G "${output_dir}/stats/val_step*.json" >/dev/null; then
    echo "=== ${name}: already evaluated, skipping"
    continue
  fi

  echo "=== ${name}"
  mkdir -p "${output_dir}"
  # Older runs' configs use the names of since-renamed JobParameters fields.
  printf '%s\n\ntest_every: %s\n' \
    "$(sed -e 's/^depth_lambda:/depth_lambda_begin:/' \
           -e 's/^floater_reg_weight:/opacity_reg_weight:/' \
           -e 's/^run_segmentation:/mask_people:/' "${run_dir}cfg.proto.txt")" \
    "${TEST_EVERY}" >"${output_dir}/cfg.proto.txt"
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

echo "Evaluations written to ${RUNS_DIR}/*/renders_test_every_${TEST_EVERY}"
