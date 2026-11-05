# Block training (design)

Status: design, not implemented. Background: the
[large-scale 3DGS survey](https://github.com/MapMindAI/EasyGaussianSplatting/wiki/Paper-Survey---Large-Scale-3D-Gaussian-Splatting).

A single gsplat run holds every Gaussian, its gradient, and two Adam moments on
the GPU, about 1 KB per Gaussian at SH degree 3. The 10M cap in
`gsplat_train_drone.proto.txt` therefore needs about 10 GB before rasterization
buffers. Block training bounds the peak at one block, following CityGaussian
for the training and keeping StitchGS's overlap candidates for a later merge.

## Scope

1. **Coarse:** train the full scene once at reduced resolution and Gaussian count.
2. **Blocks:** partition the coarse model, then fine-tune each block on its own,
   one process at a time.
3. **Merge:** crop each block to the cells it owns and concatenate (CityGaussian).

Next step, not in this design: StitchGS's overlap stitching (SIS) and global
appearance refinement with frozen geometry (GCR). Stage 2 keeps every block's
expanded-region Gaussians so SIS can replace the stage-3 crop. LoD is out of
scope.

```mermaid
flowchart LR
    WS["Workspace<br/>images, sparse/0, masks, depths"] --> COARSE["Coarse training<br/>data_factor 4, small cap_max"]
    COARSE --> CKPT["coarse ckpt"]
    CKPT --> PART["block_partition.py<br/>grid, margins, cameras"]
    PART --> MANIFEST["blocks/manifest.json"]
    MANIFEST --> TRAIN["Block training<br/>one block per process"]
    CKPT --> TRAIN
    TRAIN --> BLOCKS["block_NN ckpt + PLY"]
    BLOCKS --> MERGE["block_merge.py<br/>crop to core, concatenate"]
    MERGE --> OUT["merged ckpt + PLY"]
```

## Coordinate frames

The mapping pipeline levels the map so gravity is -Z; blocks are therefore
columns over an XY grid with unbounded Z.

gsplat's parser normalizes the world by a similarity computed from the whole
COLMAP model (`mapping/gsplat_world_frame.py`). Every block is trained with the
**full** sparse model as its parser input and only restricts which images it
samples, so the coarse run, every block, and the merge share one normalized
frame. Checkpoints can then be cropped and concatenated without any transform,
and the merged checkpoint evaluates with `simple_trainer --ckpt` unchanged.
Block bounds are stored in the COLMAP frame, where Z is up; the manifest keeps
the parser's COLMAP-to-normalized similarity to test membership.

## Stage 1: coarse model

The existing trainer with a new `gsplat_server/config/gsplat_train_coarse.proto.txt`
layered over the defaults:

| Field | Value | Why |
| --- | --- | --- |
| `data_factor` | 4 | Coarse geometry needs little resolution; image buffers shrink 16x. |
| `cap_max` | 1,000,000 | Sized to fit the smallest target GPU. |
| `iterations` | 30,000 | A prior, not a deliverable. |

Output: `<workspace>/block_training/coarse/ckpts/ckpt_<step>_rank0.pt`.

## Stage 2a: partition (`mapping/block_partition.py`)

Runs in the gsplat environment, since it reads the checkpoint and renders it.

1. **Grid.** Map the coarse means to the COLMAP frame. Split X into `columns`
   slabs at Gaussian-count quantiles, then each slab's Y into `rows` cells the
   same way, so blocks hold similar Gaussian counts rather than similar areas.
   Outer cell edges extend to infinity, so distant Gaussians and sky belong to
   an edge block instead of their own.
2. **Expanded bounds.** Grow each core cell by `margin` (default 0.2) of its
   width and depth. The expanded region is what the block trains; the core is
   what it keeps at merge.
3. **Cameras.** An image joins a block if its center lies in the expanded
   bounds, or if the coarse Gaussians inside the core cover at least
   `coverage` (default 0.25) of its pixels, rendered at 1/8 resolution with
   opacity accumulation above 0.5. The coverage test replaces CityGaussian's
   SSIM contribution test: it needs one render per image and block rather than
   two plus an SSIM.
4. **Manifest.** `<workspace>/block_training/manifest.json`:

```json
{
  "coarse_checkpoint": "coarse/ckpts/ckpt_29999_rank0.pt",
  "colmap_to_normalized": [[...4x4...]],
  "blocks": [
    {
      "name": "block_00",
      "core": {"x": [-Infinity, 12.5], "y": [-Infinity, -3.0]},
      "expanded": {"x": [-Infinity, 16.1], "y": [-Infinity, 1.2]},
      "images": ["face_0/frame_000123.jpg", "..."],
      "gaussians": 241532
    }
  ]
}
```

Options: `--rows`, `--columns`, `--margin`, `--coverage`.

## Stage 2b: block training

`mapping/train_gsplat_with_masks.py` gains `--block <manifest.json>:<name>`.
Like the existing mask and depth patches, it wraps gsplat before `simple_trainer`
runs:

* **Images.** `Dataset.__init__` keeps only indices whose image name is in the
  block. The train/test split is still taken over the full model, so held-out
  images stay held out in every block.
* **Initialization.** `create_splats_with_optimizers` builds the trainable
  splats from the coarse checkpoint's Gaussians inside the expanded bounds
  instead of SfM points.
* **Context.** The coarse Gaussians outside the expanded bounds are rendered
  with every view but are frozen: no gradient, no optimizer state, not seen by
  the strategy. Without them, a block's images show content the block cannot
  represent, and it grows floaters to explain it; occluders in front of the
  block would also be baked into its colors. Frozen, they cost about a quarter
  of their training memory. The rasterization patch appends them after the
  trainable Gaussians and slices per-Gaussian outputs (`radii`, `means2d`)
  back to the trainable count before the strategy reads them.
* **Budget.** `cap_max` is the per-block count; at least the block's initial
  count.

A shell driver, `scripts/gsplat_train_blocks.sh <workspace> [coarse parameters]
[block parameters] [rows] [columns]`, runs stage 1, the partition, every block
in sequence, and the merge. Each stage skips output already on disk, so an
interrupted run resumes at the next unfinished block.

## Stage 3: merge (`mapping/block_merge.py`)

For each block, keep the Gaussians whose means, mapped to the COLMAP frame,
fall in its core cell, and concatenate the checkpoints' `splats`. Core cells
tile the plane, so every position has exactly one owner. Write
`merged/ckpts/ckpt_merged.pt` and its PLY through `export_splats`, which the
existing patch returns to the COLMAP frame.

Block checkpoints are kept with their expanded-region Gaussians; SIS will read
them in place of the crop.

## Evaluation

On `data/wixi_chongansi`, compare against run 13 of the
[drone benchmark](https://github.com/MapMindAI/EasyGaussianSplatting/wiki/Benchmark---Wixi-Chongansi-Drone-Scene):

* PSNR, SSIM, and LPIPS of the merged checkpoint on the shared test split
  (`simple_trainer --ckpt`).
* Peak GPU memory of each stage.
* Seams: renders of test views that straddle a core boundary.

## Tests

* `block_partition_test.py`: quantile splits balance counts; outer cells are
  unbounded; expanded bounds contain the core; camera assignment by center and
  by coverage.
* `block_merge_test.py`: every Gaussian lands in exactly one core; a 1x1 grid
  reproduces its input.
* `train_gsplat_with_masks` block patches: image filtering keeps the global
  test split; frozen context receives no gradient and is sliced out of the
  strategy's inputs.

## Risks

* **Coarse stage memory.** It still holds the whole scene. If 1M Gaussians at
  `data_factor` 4 does not fit, lower `sh_degree` for this stage only.
* **Appearance jumps** between blocks trained on different image subsets. That
  is what GCR addresses in the next step.
* **MCMC relocation** can move Gaussians outside the expanded bounds. They are
  dropped at merge, which wastes budget but does not corrupt the result.
