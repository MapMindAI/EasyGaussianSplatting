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

Inside the container, `scripts/gsplat_train.sh <workspace> [parameters] [result_dir]`
runs the same training into `<workspace>/gsplat_output_<YYYYmmdd_HHMMSS>` by
default; pass it in place of the `bash -lc '...'` block. If the final PLY is
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

## Prune during MCMC training

MCMC has no built-in population reduction: it relocates low-opacity Gaussians.
DefaultStrategy prunes by opacity and 3D size, with optional image-plane size
pruning in gsplat; switching strategies also changes densification.

The training wrapper supports optional LightGaussian-style significance pruning:

```text
iterations: 12000
strategy: STRATEGY_MCMC
cap_max: 500000
mcmc_prune_steps: 6000
mcmc_prune_steps: 9000
mcmc_prune_keep_fraction: 0.8
mcmc_prune_scoring_views: 32
mcmc_prune_resolution: 256
```

Steps count completed optimizer updates, starting at 1. Each event scores the
current model on evenly sampled training views, retains the requested fraction
of its current population, and lowers `cap_max` to prevent regrowth. MCMC
relocation and noise continue according to their existing settings. The schedule
must be increasing and leave at least one optimization step after the last prune.
An empty `mcmc_prune_steps` disables pruning.

The score is `accepted_pixel_hits × sigmoid(opacity) × volume**0.1`.
Scoring uses gsplat's occlusion-aware contributor indices and integer hit
reduction, rather than LightGaussian's non-atomic counter updates. It uses the
training rasterization mode and masks, with the longest scoring image dimension
capped by `mcmc_prune_resolution`. The common volume reference is omitted because
it does not affect ranking. This follows the unclipped upstream volume weighting;
it does not reproduce the paper's clipped formula. A zero score on sampled views
does not prove a Gaussian is invisible from other viewpoints.

Deletion updates every Gaussian attribute, Adam state, and scene bookkeeping;
the MCMC binomial table is preserved. `pruning.jsonl` records iteration, population,
cap, and scoring settings. Final checkpoints and PLYs contain the reduced model.
Checkpoint/export events on a pruning iteration occur before that iteration's
optimizer and pruning updates; use a later save to inspect the pruned population.

This path supports one GPU, complete COLMAP training images, and standard pinhole
rasterization. Frozen block context, `with_ut`, and `with_eval3d` are unsupported.
Scoring processes one view at a time and subdivides crowded image regions to
bound contributor-index buffers; model and training memory still need measurement.
Do not assume the full training run fits in 4 GB from its final model size.

Compare against direct MCMC training at the same final count and SH degree, plus
an unpruned run with the same optimization budget, before attributing quality
changes to pruning.

Validation on the GTX 1650 Ti: a synthetic 12-step run pruned 64 → 51 → 41
Gaussians at steps 4 and 8, continued training, and exported a checkpoint and PLY
with 41 rows. CUDA tests cover packed/unpacked scoring, masks, image subdivision,
and preservation of Adam and scene state. This verifies integration, not real-scene
appearance preservation.

## Image and supervision settings

A `data_factor` above 1 trains on `<workspace>/images_<data_factor>/`, which
the first run writes from `images/`.

The local command does not run segmentation or depth generation merely because
`run_segmentation` or `run_depth` is set in the parameters file. Run those
stages separately. When `run_depth: true`, the trainer uses any existing
`<workspace>/depths/*.npy` files for depth supervision. It logs generated-depth
minus sparse-COLMAP-depth statistics for samples within two pixels of each
sparse point.
