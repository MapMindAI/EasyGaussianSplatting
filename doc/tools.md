# Using the tools

Run the container with your data directory mounted. Add `--gpus all` if the host
has an NVIDIA GPU — `insta360_media_stitcher` uses it for hardware video
encode/decode (NVENC/NVDEC) and falls back to software encoding without it:

```
docker run -it --rm --gpus all --shm-size=1g -v $(pwd)/data:/workspace ghcr.io/mapmindai/gaussiansplatting:latest bash
```

Add `--add-host host.docker.internal:host-gateway` too if you will reconstruct
against a Triton server on this host: Docker on Linux, unlike Mac and Windows,
does not map that name on its own.

Stitch a raw Insta360 capture into an equirectangular video:

```
insta360_media_stitcher -inputs /workspace/VID_xxx.insv -output /workspace/pano.mp4 -stitch_type optflow
```

`scripts/insta360_run_stitch.sh <workspace> [output_size]` stitches every INSV
under a workspace from the host, with the AI-stitch settings we use for
reconstruction (4000x2000, H.265, flowstate, direction lock). Each MP4 is
written beside its INSV with the same name, which lets the mapping pipeline
find it and its LRV.
The workspace must live under the repo checkout:

```
scripts/insta360_run_stitch.sh data/pano_mapping
```

A stitch runs about ten times the capture's length, longer at larger output
sizes, and is bounded at four hours to catch an SDK deadlock that hangs before
the first frame. `STITCH_TIMEOUT_SECONDS` moves that bound. A run the bound
kills exits 124 and leaves an mp4 with no moov atom, which nothing can decode;
re-running restitches it, since the skip above only keeps a video that decodes.

`scripts/run_pipeline.sh` runs the same stage as the first of its four.

For DJI Avata 360 captures, export each OSV as an equirectangular MP4 and
keep its SRT beside it under the same basename. Mapping tags the faces with the
SRT's latitude, longitude, and absolute altitude.

Read metadata off the source file:

```
exiftool /workspace/VID_xxx.insv
```

Run `conda run --no-capture-output -n gsplat python3 -m mapping.mapping_pipeline --workspace_path <dir>` to reconstruct
every MP4 under the workspace as a cube-map rig: SuperPoint and SALAD features
into a COLMAP database, LightGlue matching, and global SfM. It needs a Triton
server serving those models — see
[panorama_mapping.md](panorama_mapping.md), which also covers each stage and
the pair-selection knobs. `--frame-rate` (frames/sec sampled from the video)
defaults to 2, `--face-size` to a quarter of the video width, and `--faces` to
all six; drop `down` to leave out the nadir, which mostly shows whoever is
carrying the rig:

```
conda run --no-capture-output -n gsplat python3 -m mapping.mapping_pipeline \
  --workspace_path data/pano_mapping \
  --triton-url host.docker.internal:8011
```

The cube-map model lands in `data/pano_mapping`, ready for gsplat: the faces
are 90-degree-FOV `PINHOLE` cameras, which gsplat's COLMAP loader supports and
`EQUIRECTANGULAR` it does not.

`scripts/run_pipeline.sh` runs Mask R-CNN when `mask_people` is true and
SegFormer when `mask_sky` is true, then trains a
[gsplat](https://github.com/nerfstudio-project/gsplat) model from that cube-map
reconstruction. Both flags default to false.
Every training setting comes from the parameters file, whose
schema is `gsplat_server/proto/gsplat.proto`. The container needs about 8 GiB of
shared memory for gsplat's data-loader workers; below that they die mid-run with
a bus error.

The densification settings live in `gsplat_server/config/gsplat_train_defaults.proto.txt`, shared
with the sweep so its baseline cannot drift from what ships; `mapping/train_gsplat_with_masks.py` only attaches the masks and passes every argument through to simple_trainer. The trained model lands in `data/pano_mapping/gsplat_output`.

## Masks and depth from Triton

Two stages under `mapping/triton/` annotate a reconstruction before training,
both against the Triton server that already serves the mapping features (see
[panorama_mapping.md](panorama_mapping.md)). `--triton-url` locates it, or
`$TRITON_URL` when the flag is left out. Run segmentation in the gsplat
environment, which includes torchvision; depth uses the image's default
`python3`, which has pycolmap and the Triton gRPC client. Mount the repo at
`/workspace` so `data/` and `mapping/` both come along.

`segment_people.py` writes the training masks -- white where gsplat should
supervise, black over people and sky:

```bash
GSPLAT_HOST=192.168.11.194
WORKSPACE_PATH=data/VID_20260904_155849_009
DOCKER_RUN="docker run --rm -v $(pwd):/workspace -w /workspace \
  --add-host host.docker.internal:host-gateway \
  ghcr.io/mapmindai/gaussiansplatting:latest"

${DOCKER_RUN} conda run --no-capture-output -n gsplat python3 -m mapping.triton.segment_people \
  ${WORKSPACE_PATH}/images ${WORKSPACE_PATH}/masks \
  --debug-dir ${WORKSPACE_PATH}/debug --no-mask-sky \
  --triton-url ${GSPLAT_HOST}:8011
```

`scripts/run_pipeline.sh` and `scripts/gsplat_train.sh` read `mask_people` and
`mask_sky` from their parameters file through `scripts/gsplat_segment.sh`.
`scripts/gsplat_train_blocks.sh` generates masks enabled in either its coarse
or block parameters before partitioning.
Direct invocation of `segment_people.py` enables both by default:
`--no-mask-people` skips Mask R-CNN, and `--no-mask-sky` skips SegFormer.
`--dilation` grows each person mask (1 pixel), and `--num-threads` sets how many
images infer at once (4). Masks mirror
any nested folders under `images/`, so each keeps the image's relative path and
adds `.png` to its filename. Existing masks are kept unless `--overwrite` is
given. SegFormer runs a 512-pixel model on overlapping tiles, so masks retain
the image's local detail; large images take proportionally longer to segment.

`--debug-dir` additionally writes each image with people in red over the source
image; when sky is enabled, it also washes the SegFormer label map over it.
It also writes `label_colours.png`, the exact ADE20K label-index colour legend.

Adding it to an already-segmented directory re-runs the images whose overlay is
missing, so the masks do not have to be thrown away to get the views.

`mapping/depth/generate_depth.py` predicts depth with Depth Anything 3 on
Triton. It is not part of `run_pipeline.sh`, so run it yourself. Segmentation
is optional: run it first and its masks gate what depth samples; without
masks, or for an image without one, every pixel is eligible:

```bash
${DOCKER_RUN} conda run --no-capture-output -n gsplat python3 -m mapping.depth.generate_depth \
  ${WORKSPACE_PATH} ${WORKSPACE_PATH}/depths \
  --mask-dir ${WORKSPACE_PATH}/masks \
  --debug-dir ${WORKSPACE_PATH}/debug \
  --samples-per-image -1 \
  --triton-url ${GSPLAT_HOST}:8011
```

It reads `images/` and `sparse/` from the model directory and groups images
that view the same part of the scene (`mapping/depth/covisibility_groups.py`,
ported from ArSession): two images are linked when they share at least 20
sparse points with 20% overlap, at least 0.05 map units of baseline, at most
45 degrees of relative rotation, and 0.5 to 25 degrees of median parallax.
Groups are built greedily around anchors, favouring supports near 5 degrees of
parallax, spread-out cameras, and new coverage. DA3 takes exactly five views,
so only full groups are formed; an image no group can hold gets no depth.

Each group's depth is in its own arbitrary scale, so it is fitted to the COLMAP
points its images observe and to their camera baselines
(`mapping/depth/scale_alignment.py`). The model's predicted cameras give a
first scale when at least three agree with COLMAP within 5 degrees; a
frame-balanced weighted median of the log depth ratios, a MAD inlier band, and
Huber reweighting then refine it, ignoring references beyond 20 map units.
Without a camera scale, a group needs 30 inlier points over two frames, or is
skipped rather than written at a guessed scale. An image in several groups
keeps the prediction whose scaled depth best matches its own sparse depths.
Groups share images, so an existing `depths/` is kept whole; pass
`--overwrite` to regenerate it.
The `up/` and `down/` cube faces are skipped: their sky and ground views do not
need depth supervision.
`--samples-per-image` (4096) sets how many pixels each `.npy` keeps,
`--min-confidence` (0.5) the confidence floor, and `--min-depth` (0.1) /
`--max-depth` (unbounded) discard depths outside that range, in the depth
model's units before scaling.
Colour and DA3-depth discontinuities are also excluded, so depth supervision
does not pull geometry across foliage and other detail edges. Debug overlays
show only the depth that remains after filtering.
`--debug-dir <directory>` writes a colourized depth overlay over every source
image, preserving nested image paths and adding `_depth.png` to the filename.
[gsplat_server.md](gsplat_server.md#depth-supervision) covers the file format
and the limits of the depth term.

### VGGT-Omega instead of DA3

`mapping/vggt_omega/generate_depth.py` is the same stage with VGGT-Omega in
place of DA3, run locally through ONNX Runtime rather than on Triton, and is
what the server's `run_depth` uses. The weights are not public: place the
export at `models/vggt_omega_416.onnx` (with its `.onnx.data`), or pass
`--model`. Groups hold up to 12 views, though the export's `.json` lists
10 as its maximum. Inputs are a fixed 416x416; images are resized to that,
ignoring aspect ratio, and the maps resized back. Groups of two views or more
are kept. Once depth is written, it merges every `.npy` into
`<model_dir>/depth_points.ply` (`mapping/depth/point_cloud.py`): rows lifted
through the COLMAP intrinsics and poses, coloured from the images, and averaged
per voxel. `--voxel-factor` (0.001) sets the voxel edge as a fraction of the
scene scale, the largest camera distance from the cameras' mean, as gsplat
measures it; the server and training scripts pass the parameters file's
`depth_voxel_factor`. It takes the flags above except
`--triton-url`, and defaults `--num-threads` to 2. Its confidence is VGGT's
divided by 5, so it starts at 0.2 and the default `--min-confidence` of 0.5 drops
pixels below VGGT's 2.5: mostly sky and far background.

```bash
${DOCKER_RUN} python3 -m mapping.vggt_omega.generate_depth \
  ${WORKSPACE_PATH} ${WORKSPACE_PATH}/depths \
  --mask-dir ${WORKSPACE_PATH}/masks \
  --samples-per-image -1
```

To try the model on a few images, `example.py` writes each one's raw depth
(`<stem>_depth.npy`, in the model's arbitrary scale), its confidence
(`<stem>_confidence.npy`), a colourized overlay
(`<stem>_depth.png`), and `points.ply`: the more confident half of every
image's pixels, lifted through the model's own predicted cameras into one
coloured cloud:

```bash
python3 -m mapping.vggt_omega.example output/vggt_omega a.jpg b.jpg c.jpg
```

Training reads `depths/` only when the parameters file sets `run_depth: true`
and a positive `depth_lambda`, which weights gsplat's depth term; with
`run_depth: true`, a `depth_points.ply` in the workspace is also added to the
sparse SfM points that initialize the Gaussians, its points randomly thinned
under MCMC to keep the total within 0.8 × `cap_max`:

```
run_depth: true
depth_lambda: 0.01
```

Calling `mapping/train_gsplat_with_masks.py` directly does not generate masks
or depth. The server, `scripts/gsplat_train.sh`, and
`scripts/gsplat_train_blocks.sh` honor `mask_people`, `mask_sky`, and
`run_depth`; the pipeline script honors the two mask flags (see
[gsplat_server.md](gsplat_server.md#submitting-a-model)).

## TSDF mesh from a Gaussian Splat

`mapping/tsdf/run_pipeline.sh` renders an expected projective z-depth map
for every registered cube face, then fuses the maps with TSDF and writes
`tsdf/mesh.ply`. Depth rendering runs in the Gaussian Splatting image; fusion
runs in a dedicated `python:3.11-slim` image with the official
`open3d==0.19.0` wheel. The mesh uses the same COLMAP coordinates as `sparse/0/`.
Run it after training:

```bash
bash mapping/tsdf/run_pipeline.sh ${WORKSPACE_PATH}
```

By default it reads the highest-step PLY under the newest `gsplat_output*/ply/`, retains
existing depths, and writes `tsdf/mesh.ply`. Pass an exported PLY as the second
argument when training through gRPC: `bash mapping/tsdf/run_pipeline.sh
${WORKSPACE_PATH} ${WORKSPACE_PATH}/gsplat.ply`. The dedicated image is built
from [artifacts/docker_o3d/Dockerfile](../artifacts/docker_o3d/Dockerfile) and
published as `ghcr.io/mapmindai/gaussiansplatting-tsdf:latest`. Set
`TSDF_DOCKER_IMAGE` to use another image.

### Inspecting the rendered depths

`mapping/tsdf/view_depth.py` opens the rendered maps in a window, colour-mapped
against a fixed scale so the same colour means the same distance in every frame
and face. It needs only NumPy and Matplotlib, so run it on the host rather than
through Docker:

```bash
python3 -m mapping.tsdf.view_depth ${WORKSPACE_PATH}/tsdf/depths --face front
```

Left and right step through frames, up and down through faces, and the cursor
readout gives the depth in metres under the pointer. Unrendered pixels stay
grey. Raise `--maximum-depth` above the 20 m default to keep far structure
apart from the sky.

## Comparing parameter configurations

`mapping/benchmark/gsplat_sweep.sh` trains one model per configuration from a single
cube-map reconstruction, into `<reconstruction>/sweep/<name>/`. Pass a directory
of text-format `JobParameters` files inside the repo. Runs are sequential, and
a configuration that already recorded evaluation metrics is skipped:

```
mapping/benchmark/gsplat_sweep.sh data/panorama path/to/configurations
```

The [historical sweep configurations](https://github.com/MapMindAI/EasyGaussianSplatting/wiki/Benchmark---Parameter-Sweep)
are documented in the wiki. To sweep on a remote GPU instead,
`mapping/benchmark/gsplat_server_sweep.py` submits the same files to a training
server (see `doc/gsplat_server.md`):

```
python3 mapping/benchmark/gsplat_server_sweep.py data/panorama \
  --server host:50051 --configurations path/to/configurations \
  --results data/panorama/sweep_windows
```

The report is generated output and is not tracked: it links
to point clouds and thumbnails under `data/`, so regenerate it rather than
reading a stale copy.

`mapping/benchmark/summarize_gsplat_sweep.py` then writes a Markdown comparison of those
runs -- evaluation metrics, the geometry their point clouds actually contain,
the settings that differ between them, and each run's renders beside the ground
truth -- to `doc/gsplat_parameter_sweep.md`. Runs still training show as pending:

```
python3 mapping/benchmark/summarize_gsplat_sweep.py --runs-root data/panorama/sweep
```

The `Radius / spacing` column is the one that explains lost detail: it divides
the median Gaussian radius by the estimated median distance to the nearest
neighbour, so a value well below 1 means the Gaussians are too small to tile a
surface and leave gaps.
