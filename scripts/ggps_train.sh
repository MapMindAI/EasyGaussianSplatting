#!/usr/bin/env bash
# Train a Gaussian Splatting model with GGPS from a COLMAP EQUIRECTANGULAR
# reconstruction (see colmap_reconstruct.sh). Run inside the container.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <reconstruction_dir> [iterations] [resolution]" >&2
  exit 1
fi

RECONSTRUCTION_DIR="$(cd "$1" && pwd)"
ITERATIONS="${2:-30000}"
RESOLUTION="${3:-1}"

SPARSE_DIR="${RECONSTRUCTION_DIR}/sparse/0"
if [ ! -f "${SPARSE_DIR}/cameras.txt" ]; then
  # GGPS's COLMAP reader falls back to text format for camera models (like
  # EQUIRECTANGULAR) it doesn't recognize in the binary format.
  colmap model_converter --input_path "${SPARSE_DIR}" --output_path "${SPARSE_DIR}" --output_type TXT
fi

MODEL_DIR="${RECONSTRUCTION_DIR}/ggps_output"
CONFIG_PATH="${RECONSTRUCTION_DIR}/ggps_train.yaml"
mkdir -p "${MODEL_DIR}"

cat > "${CONFIG_PATH}" <<YAML
model_params: {
    model_config: { name: "GaussianModel", kwargs: {} },
    sh_degree: 3,
    source_path: "${RECONSTRUCTION_DIR}",
    model_path: "${MODEL_DIR}",
    images: "images",
    alpha_masks: "",
    use_alpha_masks: False,
    use_sky_masks: False,
    use_depth: False,
    resolution: ${RESOLUTION},
    white_background: False,
    data_device: "cpu",
    eval: True,
    skybox_num: 100000,
    skybox_locked: False,
    default_camera_type: 3,
}

pipeline_params: {
    convert_SHs_python: False,
    compute_cov3D_python: False,
    debug: False
}

optim_params: {
    iterations: ${ITERATIONS},
    position_lr_init: 0.00016,
    position_lr_final: 0.0000016,
    position_lr_delay_mult: 0.01,
    position_lr_max_steps: 30_000,
    feature_lr: 0.0025,
    opacity_lr: 0.05,
    scaling_lr: 0.005,
    rotation_lr: 0.001,
    percent_dense: 0.01,
    lambda_dssim: 0.2,
    densification_interval: 100,
    opacity_reset_interval: 3000,
    densify_from_iter: 500,
    densify_until_iter: 15_000,
    densify_grad_threshold: 0.0002,
    densify_grad_abs_threshold: 0.0004,
    prune_by_extent: false,
    depth_l1_weight_init: 0,
    depth_l1_weight_final: 0,
    max_cache_num: 512,
    skip_bottom_ratio: 0,
}
YAML

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ggps
cd /opt/ggps
python3 train_large.py --config "${CONFIG_PATH}" \
  --test_iterations "${ITERATIONS}" --save_iterations "${ITERATIONS}"

echo "Model written to ${MODEL_DIR}"
