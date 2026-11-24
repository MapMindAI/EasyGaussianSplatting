# Local gsplat training

`mapping/train_gsplat_with_masks.py` trains a Gaussian Splat directly from a
cube-map COLMAP reconstruction. It runs in the project's GPU image, whose
gsplat environment includes the CUDA extensions the trainer needs.

## Prerequisites

Build or pull the image, and create a workspace with
`mapping.mapping_pipeline`. The workspace must contain `images/` and
`sparse/0/`:

```text
data/pano_mapping/
├── images/
└── sparse/0/
```

Optionally create masks before training. White pixels are supervised and black
pixels are ignored:

```bash
WORKSPACE_PATH=data/pano_mapping

docker run -it --rm --gpus all --shm-size=8g \
  -v "$PWD":/workspace -w /workspace \
  --mount type=volume,source=easygaussiansplatting-torch-cache,target=/root/.cache/torch \
  ghcr.io/mapmindai/gaussiansplatting:latest \
  bash -lc '
    source scripts/gsplat_env.sh
    python3 mapping/triton/segment_people.py \
      "'"${WORKSPACE_PATH}"'/images" "'"${WORKSPACE_PATH}"'/masks"
  '
```

Direct invocation of `segment_people.py` enables both models by default.
`--no-mask-people` skips Mask R-CNN, and `--no-mask-sky` skips the U²-Net sky
model, which needs `models/skyseg.onnx` (see [tools.md](tools.md#sky-segmentation)).
Training works without `masks/`.

## Train

```bash
WORKSPACE_PATH=data/pano_mapping

docker run -it --rm --gpus all --shm-size=8g \
  -v "$PWD":/workspace -w /workspace \
  --mount type=volume,source=easygaussiansplatting-torch-cache,target=/root/.cache/torch \
  ghcr.io/mapmindai/gaussiansplatting:latest \
  bash -lc '
    source scripts/gsplat_env.sh
    export PYTHONPATH="$PWD"
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    python3 mapping/train_gsplat_with_masks.py default \
      --job_parameters gsplat_server/config/gsplat_train_defaults.proto.txt \
      --data_dir "'"${WORKSPACE_PATH}"'" \
      --save_ply \
      --antialiased \
      --disable_viewer \
      --result_dir "'"${WORKSPACE_PATH}"'/gsplat_output"
  '
```

Inside the container, `scripts/gsplat_train.sh <workspace> [parameters] [result_dir]`
runs segmentation according to `mask_people` and `mask_sky`, then trains into
`<workspace>/gsplat_output_<YYYYmmdd_HHMMSS>` by default, and saves the resolved
parameters, defaults included, as `cfg.proto.txt` there. Existing
masks are reused. Both masks are inferred locally.
Pass the script in place of the `bash -lc '...'` block. If the final PLY is
2 GiB or larger, it also bisects it along the longer horizontal axis into
`ply/split/point_cloud_<step>_NN.ply`, each under 2 GiB, for browser viewers.

The final model and PLY files are written under
`<workspace>/gsplat_output/`. `--shm-size=8g` is required for the data-loader
workers; a smaller allocation can cause a bus error.
The named Torch-cache volume preserves downloaded model weights between runs.
The resolved training configuration is printed before training starts.

Use `easygaussiansplatting:dev` in place of the published image when you built
the image locally.

## Parameters

`--job_parameters` takes a text-format `JobParameters` file. It is layered
over `gsplat_server/config/gsplat_train_defaults.proto.txt`, so a custom file
only needs the values it changes:

```text
iterations: 5000
cap_max: 250000
min_opacity: 0.01
```

With MCMC, `min_opacity` relocates Gaussians at or below the threshold during
refinement; `cap_max` limits the count. `prune_opa` applies only to
DefaultStrategy.

A `data_factor` above 1 trains on `<workspace>/images_<data_factor>/`, which
the first run writes from `images/`.

Calling `mapping/train_gsplat_with_masks.py` directly does not generate masks
or depth; `scripts/gsplat_train.sh` does both when the parameters ask for them,
or run them separately. When `run_depth: true`, the trainer initializes from the SfM points
plus any `<workspace>/depth_points.ply` (thinned to keep the total within
0.8 × `cap_max` under MCMC), and with a positive `depth_lambda_begin` uses any
existing `<workspace>/depths/*.npz` files for depth supervision. It logs generated-depth
minus sparse-COLMAP-depth statistics for samples within two pixels of each
sparse point.
