#!/usr/bin/env bash
# Container: train a large scene block by block (doc/block_training.md).
#
# Usage: scripts/gsplat_train_blocks.sh <workspace> [max_images] [coarse.proto.txt] [block.proto.txt]
#
# Partitions sparse/0 into tree blocks of at most max_images training images
# (default 4000), trains a coarse model of the whole scene, fine-tunes each
# block from it in turn, and concatenates the Gaussians each block owns.
# Results go to <workspace>/block_training. A stage whose output exists is
# skipped, so a rerun resumes at the first unfinished block.
set -euo pipefail

cd "$(dirname "$0")/.."
WORKSPACE_PATH=${1:?usage: $0 <workspace> [max_images] [coarse.proto.txt] [block.proto.txt]}
MAX_IMAGES=${2:-4000}
COARSE_PARAMETERS=${3:-gsplat_server/config/gsplat_train_coarse.proto.txt}
BLOCK_PARAMETERS=${4:-gsplat_server/config/gsplat_train_block.proto.txt}
OUTPUT_DIR=${WORKSPACE_PATH}/block_training
MANIFEST=${OUTPUT_DIR}/partition/tree.json

source scripts/gsplat_env.sh
export PYTHONPATH="$PWD"

# Checkpoints are only written at the last step, so one means the run finished.
latest_checkpoint() {
  ls -v "$1"/ckpts/ckpt_*_rank0.pt 2>/dev/null | tail -n 1
}

if [ ! -f "${MANIFEST}" ]; then
  python3 -m mapping.block_partition "${WORKSPACE_PATH}/sparse/0" "${OUTPUT_DIR}/partition" \
    --method tree --max-images "${MAX_IMAGES}"
fi

if [ -z "$(latest_checkpoint "${OUTPUT_DIR}/coarse")" ]; then
  bash scripts/gsplat_train.sh "${WORKSPACE_PATH}" "${COARSE_PARAMETERS}" "${OUTPUT_DIR}/coarse"
fi
COARSE_CHECKPOINT=$(latest_checkpoint "${OUTPUT_DIR}/coarse")

BLOCKS=$(python3 -c 'import json, sys; print(*(block["name"] for block in json.load(open(sys.argv[1]))["blocks"]))' \
  "${MANIFEST}")
for BLOCK in ${BLOCKS}; do
  if [ -z "$(latest_checkpoint "${OUTPUT_DIR}/blocks/${BLOCK}")" ]; then
    bash scripts/gsplat_train.sh "${WORKSPACE_PATH}" "${BLOCK_PARAMETERS}" "${OUTPUT_DIR}/blocks/${BLOCK}" \
      --block "${MANIFEST}:${BLOCK}" --coarse_checkpoint "${COARSE_CHECKPOINT}"
  fi
done

python3 -m mapping.block_gaussians "${MANIFEST}" "${OUTPUT_DIR}/blocks" "${OUTPUT_DIR}/merged"
