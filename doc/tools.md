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

`scripts/run_stitch.sh <workspace> [output_size]` stitches every INSV under a
workspace from the host, with the AI-stitch settings we use for reconstruction
(4000x2000, H.265, flowstate, direction lock). Each MP4 is written beside its
INSV with the same name, which lets the mapping pipeline find it and its LRV.
The workspace must live under the repo checkout:

```
scripts/run_stitch.sh data/pano_mapping
```

A stitch runs about ten times the capture's length, longer at larger output
sizes, and is bounded at four hours to catch an SDK deadlock that hangs before
the first frame. `STITCH_TIMEOUT_SECONDS` moves that bound. A run the bound
kills exits 124 and leaves an mp4 with no moov atom, which nothing can decode;
re-running restitches it, since the skip above only keeps a video that decodes.

`scripts/run_pipeline.sh` runs the same stage as the first of its four.

Read metadata off the source file:

```
exiftool /workspace/VID_xxx.insv
```

Run `python3 -m mapping.mapping_pipeline --workspace_path <dir>` to reconstruct
every MP4 under the workspace as a cube-map rig: SuperPoint and SALAD features
into a COLMAP database, LightGlue matching, and global SfM. It needs a Triton
server serving those models — see
[panorama_mapping.md](panorama_mapping.md), which also covers each stage and
the pair-selection knobs. `--frame-rate` (frames/sec sampled from the video)
defaults to 2, `--face-size` to a quarter of the video width, and `--faces` to
all six; drop `down` to leave out the nadir, which mostly shows whoever is
carrying the rig:

```
python3 -m mapping.mapping_pipeline \
  --workspace_path data/pano_mapping \
  --triton-url host.docker.internal:8011
```

The cube-map model lands in `data/pano_mapping`, ready for gsplat: the faces
are 90-degree-FOV `PINHOLE` cameras, which gsplat's COLMAP loader supports and
`EQUIRECTANGULAR` it does not.

`scripts/run_pipeline.sh` then masks out people and trains a
[gsplat](https://github.com/nerfstudio-project/gsplat) model from that cube-map
reconstruction. Every training setting comes from the parameters file, whose
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
VIDEO_NAME=VID_20260904_155849_00_009
WORKSPACE_PATH=data/${VIDEO_NAME}_reconstruction
DOCKER_RUN="docker run --rm -v $(pwd):/workspace -w /workspace \
  --add-host host.docker.internal:host-gateway \
  ghcr.io/mapmindai/gaussiansplatting:latest"

${DOCKER_RUN} conda run --no-capture-output -n gsplat python3 -m mapping.triton.segment_people \
  ${WORKSPACE_PATH}/images ${WORKSPACE_PATH}/masks \
  --debug-dir ${WORKSPACE_PATH}/debug --no-mask-sky \
  --triton-url ${GSPLAT_HOST}:8011
```

`scripts/run_pipeline.sh` runs this with its defaults; run it by hand to change
them. People use torchvision Mask R-CNN. `--no-mask-sky` masks people only and
does not call SegFormer; otherwise SegFormer supplies sky masks. `--dilation` grows each person mask (1
pixel), and `--num-threads` sets how many images infer at once (4). Masks mirror
any nested folders under `images/`, so each keeps the image's relative path and
adds `.png` to its filename. Existing masks are kept unless `--overwrite` is
given. SegFormer runs a 512-pixel model on overlapping tiles, so masks retain
the image's local detail; large images take proportionally longer to segment.

`--debug-dir` additionally writes each image with people in red over the source
image; when sky is enabled, it also washes the SegFormer label map over it.
It also writes `label_colours.png`, the exact ADE20K label-index colour legend.

Adding it to an already-segmented directory re-runs the images whose overlay is
missing, so the masks do not have to be thrown away to get the views.

`generate_depth.py` predicts depth with Depth Anything 3. It is not part of
`run_pipeline.sh`, so run it yourself, after segmentation so the masks can gate
what it samples:

```bash
${DOCKER_RUN} python3 -m mapping.triton.generate_depth \
  ${WORKSPACE_PATH} ${WORKSPACE_PATH}/depths \
  --mask-dir ${WORKSPACE_PATH}/masks \
  --debug-dir ${WORKSPACE_PATH}/debug \
  --triton-url ${GSPLAT_HOST}:8011
```

It reads `images/` and `sparse/` from the model directory, groups each camera's
images into consecutive groups of five (the served model's input shape), and
fits each group to the scale of the COLMAP points its images already observe --
DA3's own scale is arbitrary and differs per group. A group carrying too few of
those points is skipped rather than written at a guessed scale.
`--samples-per-image` (4096) sets how many pixels each `.npy` keeps,
`--min-confidence` (2.0) the DA3 confidence floor.
`--debug-dir <directory>` writes a colourized depth overlay over every source
image, preserving nested image paths and adding `_depth.png` to the filename.
[gsplat_server.md](gsplat_server.md#depth-supervision) covers the file format
and the limits of the depth term.

Training reads `depths/` only when the parameters file sets `run_depth: true`,
which adds gsplat's depth term weighted by `depth_lambda`:

```
run_depth: true
depth_lambda: 0.01
```

Neither stage runs from the parameters file locally -- `run_segmentation` and
`run_depth` only tell the server to run them (see
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

By default it reads the highest-step PLY under `gsplat_output/ply/`, retains
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
cube-map reconstruction, into `<reconstruction>/sweep/<name>/`. Runs are
sequential, and a configuration that already recorded evaluation metrics is
skipped, so an interrupted sweep resumes where it stopped:

```
mapping/benchmark/gsplat_sweep.sh data/panorama
```

Both this and the server-side sweep read the same parameter files from
`mapping/benchmark/configurations/`, so add or edit a file there to change what
is compared. To sweep on a remote GPU instead, `mapping/benchmark/gsplat_server_sweep.py`
submits the same configurations to a training server (see `doc/gsplat_server.md`):

```
python3 mapping/benchmark/gsplat_server_sweep.py data/panorama \
  --server host:50051 --results data/panorama/sweep_windows
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
