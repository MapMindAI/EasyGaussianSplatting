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
GSPLAT_HOST=192.168.11.194

docker run -it --rm --gpus all --shm-size=8g \
  -v "$PWD":/workspace -w /workspace \
  --mount type=volume,source=easygaussiansplatting-torch-cache,target=/root/.cache/torch \
  --add-host host.docker.internal:host-gateway \
  ghcr.io/mapmindai/gaussiansplatting:latest \
  bash -lc '
    source scripts/gsplat_env.sh
    python3 mapping/triton/segment_people.py \
      "'"${WORKSPACE_PATH}"'/images" "'"${WORKSPACE_PATH}"'/masks" \
      --triton-url "'"${GSPLAT_HOST}"':8011
  '
```

Segmentation requires Triton with the SegFormer model available; see
[panorama mapping](panorama_mapping.md). Training works without `masks/`.

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

The final model and PLY files are written under
`<workspace>/gsplat_output/`. `--shm-size=8g` is required for the data-loader
workers; a smaller allocation can cause a bus error.
The named Torch-cache volume preserves downloaded model weights between runs.

Use `easygaussiansplatting:dev` in place of the published image when you built
the image locally.

## Parameters

`--job_parameters` takes a text-format `JobParameters` file. It is layered
over `gsplat_server/config/gsplat_train_defaults.proto.txt`, so a custom file
only needs the values it changes:

```text
iterations: 5000
cap_max: 250000
```

The local command does not run segmentation or depth generation merely because
`run_segmentation` or `run_depth` is set in the parameters file. Run those
stages separately. When `run_depth: true`, the trainer uses any existing
`<workspace>/depths/*.npy` files for depth supervision. It logs generated-depth
minus sparse-COLMAP-depth statistics for samples within two pixels of each
sparse point.
