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
  echo "Usage: $0 <cubemap_reconstruction_dir> [iterations]" >&2
  exit 1
fi

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)"
source "${SCRIPTS_DIR}/docker_common.sh"
source "${SCRIPTS_DIR}/gsplat_train_defaults.sh"

REL_CUBEMAP_DIR="$(repo_relative_path "$1")"
ITERATIONS="${2:-30000}"
SWEEP_DIR="${REPO_ROOT}/${REL_CUBEMAP_DIR}/sweep"

# The shipped configuration, so 00_baseline reproduces scripts/gsplat_train.sh.
BASELINE_OPTIONS="--opacity_reg ${GSPLAT_FLOATER_REG_WEIGHT} \
--scale_reg ${GSPLAT_FLOATER_REG_WEIGHT} \
--strategy.grow-grad2d ${GSPLAT_GROW_GRAD2D}"

# name | simple_trainer subcommand | configuration under test
#
# Every `default` configuration after the baseline drops --opacity_reg and
# --scale_reg, since penalizing opacity and scale is what erases fine detail.
# On a 4 GB GPU they then all OOM, because DefaultStrategy has no ceiling and
# reaches 1.6-2.9 M Gaussians by step 3-9 k; the `mcmc` ones cap the count
# outright and are the usable configurations there.
CONFIGURATIONS=(
  "00_baseline|default|${BASELINE_OPTIONS} --pose_opt"
  "01_no_floater_reg|default|--strategy.grow-grad2d 0.0006 --pose_opt"
  "02_grow_grad2d_0003|default|--strategy.grow-grad2d 0.0003 --pose_opt"
  "03_grow_grad2d_0002|default|--strategy.grow-grad2d 0.0002 --packed --pose_opt"
  "04_refine_longer|default|--strategy.grow-grad2d 0.0003 --strategy.refine-stop-iter 25000 --strategy.reset-every 5000 --packed --pose_opt"
  "05_ssim_050|default|--strategy.grow-grad2d 0.0003 --ssim_lambda 0.5 --pose_opt"
  "06_no_pose_opt|default|--strategy.grow-grad2d 0.0003"
  "07_sh_degree_2|default|--strategy.grow-grad2d 0.0002 --sh_degree 2 --packed --pose_opt"
  "08_mcmc_cap_500k|mcmc|--strategy.cap-max 500000 --pose_opt"
  "09_mcmc_cap_900k|mcmc|--strategy.cap-max 900000 --packed --pose_opt"
  "10_mcmc_cap_1500k|mcmc|--strategy.cap-max 1500000 --packed --pose_opt"
  "11_mcmc_cap_900k_ssim050|mcmc|--strategy.cap-max 900000 --ssim_lambda 0.5 --packed --pose_opt"
  "12_mcmc_cap_2000k_sh2|mcmc|--strategy.cap-max 2000000 --sh_degree 2 --packed --pose_opt"
)

for configuration in "${CONFIGURATIONS[@]}"; do
  IFS='|' read -r name subcommand flags <<<"${configuration}"
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

  if [ "${subcommand}" = "default" ]; then
    flags="${GSPLAT_STRATEGY_OPTIONS[*]} ${flags}"
  fi

  mkdir -p "${run_dir}"
  echo "=== ${name}: ${subcommand} ${flags}"
  start_seconds="${SECONDS}"
  # A configuration that exhausts the GPU must not abort the rest of the sweep.
  status=0
  docker run "${DOCKER_RUN_FLAGS[@]}" \
    -e SUBCOMMAND="${subcommand}" \
    -e RESULT_DIR="/workspace/${REL_CUBEMAP_DIR}/sweep/${name}" \
    -e DATA_DIR="/workspace/${REL_CUBEMAP_DIR}" \
    -e ITERATIONS="${ITERATIONS}" \
    -e FLAGS="${flags}" \
    "${DOCKER_IMAGE}" \
    bash -c 'set -euo pipefail
source scripts/gsplat_env.sh
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
python3 mapping/train_gsplat_with_masks.py "$SUBCOMMAND" \
  --data_dir "$DATA_DIR" \
  --data_factor 1 \
  --max_steps "$ITERATIONS" \
  --eval_steps "$ITERATIONS" \
  --save_steps "$ITERATIONS" \
  --ply_steps "$ITERATIONS" \
  --save_ply \
  --antialiased \
  --disable_viewer \
  --result_dir "$RESULT_DIR" \
  $FLAGS' >"${run_dir}/train.log" 2>&1 || status=$?

  printf 'exit_status %d\nwall_seconds %d\nsubcommand %s\nflags %s\n' \
    "${status}" "$((SECONDS - start_seconds))" "${subcommand}" "${flags}" \
    >"${run_dir}/sweep_status.txt"
  if [ "${status}" -ne 0 ]; then
    echo "=== ${name}: FAILED (exit ${status}), see ${run_dir}/train.log"
  fi
done

echo "Sweep written to ${SWEEP_DIR}"
