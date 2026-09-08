# EasyGaussianSplatting

An end-to-end pipeline for turning Insta360 360° captures into Gaussian Splatting
reconstructions. Each panorama frame is reprojected into a rig of pinhole cube
faces, matched with learned features (SuperPoint, LightGlue, and SALAD served
over gRPC by [EasyTensorRT](https://github.com/MapMindAI/EasyTensorRT)), and
solved with global SfM.

## Running the full pipeline in one command

`scripts/run_pipeline.sh <input.insv> [frame_rate] [output_size] [face_size] [parameters.proto.txt]`
is the whole pipeline: it runs on the host and drives the container itself,
chaining stitching, cube-map reconstruction, person masking, and gsplat
training through a single `docker run --gpus all --shm-size=8g`. `input.insv`
must live under the repo checkout (it gets bind-mounted as `/workspace`).
Reconstruction infers against a Triton server, which `TRITON_URL` locates and
[doc/panorama_mapping.md](doc/panorama_mapping.md) covers starting.
`frame_rate` defaults to 2, `output_size` to `8000x4000`, `face_size`
(cube-face width/height in pixels) to a quarter of the video width — which
keeps the panorama's angular resolution — and `parameters.proto.txt` to
`gsplat_server/config/gsplat_train_defaults.proto.txt`. Pass a different
text-format `JobParameters` file as the fifth argument to change training.
Every stage skips work already on disk, so an interrupted run resumes:

```
scripts/run_pipeline.sh data/VID_xxx.insv 2 4000x2000
```

Results land next to the capture, in `<capture_name>_reconstruction/`:
`pano.mp4` (the stitched video), `pano_mapping/sparse` (the cube-map model
gsplat trains on), and `pano_mapping/gsplat_output` (the trained model). Set
`DOCKER_IMAGE` to use a locally built image instead of the published one. The
script keeps downloaded PyTorch model weights in the persistent Docker volume
`easygaussiansplatting-torch-cache`, so later runs reuse them.
Existing person-segmentation masks are reused on later runs.

To stitch without reconstructing, `scripts/run_stitch.sh <input.insv>` runs
that stage on its own; see [doc/tools.md](doc/tools.md).

![COLMAP sparse reconstruction viewer](assets/reconstruction_viewer.jpg)

## Status

This branch is a ground-up remake of the pipeline. What's done so far:

- [x] Docker image with COLMAP, the Insta360 Media SDK, and ExifTool
- [x] Script to stitch raw Insta360 footage into equirectangular video
- [x] Script to reconstruct a panorama capture as a cube-map rig, with
      SuperPoint/LightGlue/SALAD features and global SfM
- [x] Script to train a Gaussian Splatting model from the cube-map reconstruction (gsplat)
- [x] gRPC training server
- [ ] Export/viewer wired to the above

## What's in the Docker image

Built from `artifacts/docker/dev.dockerfile`:

- **COLMAP** (>= 4.1.0), with rig/frame support and the GLOMAP global mapper.
- **Insta360 Media SDK**, exposed as `insta360_media_stitcher`, for stitching raw
  `.insv`/`.lrv` footage into a panorama video or image sequence.
- **ExifTool**, for reading GPS/timestamp metadata off the source footage.
- **pycolmap** and the `tritonclient` gRPC client, for the mapping pipeline in
  `mapping/`.
- A `gsplat` conda environment with [gsplat](https://github.com/nerfstudio-project/gsplat)
  (Gaussian Splatting training, vendored as the `third_party/gsplat` submodule)
  and its compiled CUDA extensions.

## Getting the image

Pull the image CI publishes on every push to `master`:

```
docker pull ghcr.io/mapmindai/gaussiansplatting:latest
```

Or build it locally from your checkout. The build needs your `third_party/gsplat`
submodule checked out (`git submodule update --init`) and passed in as an
additional build context, since the Dockerfile's own build context is just
`artifacts/docker/` (kept small so it doesn't have to send `data/`):

```
git submodule update --init third_party/gsplat
docker build -f artifacts/docker/dev.dockerfile -t easygaussiansplatting:dev \
  --build-context gsplatsrc=./third_party/gsplat artifacts/docker
```

## Training over gRPC

`gsplat_server/` turns a GPU host into a training appliance: clients stream a
zipped COLMAP model over gRPC, the host trains it, and they download the point
cloud. [doc/gsplat_server.md](doc/gsplat_server.md) covers it on an x86_64
Linux host and links the Jetson AGX Orin and Windows guides.

```
gsplat_server/run_server.sh                     # on the GPU host
gsplat_server/client.py data/pano_mapping --server gsplat-host:50051
```

## Using the tools

See [doc/tools.md](doc/tools.md).

## Reconstructing a capture

1. [Stitch to video](doc/tools.md) `scripts/run_stitch.sh data/${VIDEO_NAME}.insv`
2. [Mapping a panorama capture](doc/panorama_mapping.md) run with video.

```bash
GSPLAT_HOST=192.168.11.194
VIDEO_NAME=VID_20260904_155849_00_009
docker run -it --rm --gpus all -v $(pwd):/workspace -w /workspace \
  --add-host host.docker.internal:host-gateway \
  ghcr.io/mapmindai/gaussiansplatting:latest \
  python3 -m mapping.mapping_pipeline \
    --video_path data/${VIDEO_NAME}_pano.mp4 \
    --workspace_path data/${VIDEO_NAME}_reconstruction \
    --triton-url ${GSPLAT_HOST}:8011 --num-threads 4
```

3. [Training a Gaussian Splatting model](doc/gsplat_server.md)

```bash
WORKSPACE_PATH=${VIDEO_NAME}_reconstruction
gsplat_server/client.py data/${WORKSPACE_PATH} \
  --parameters gsplat_server/config/gsplat_train_defaults.proto.txt \
  --server ${GSPLAT_HOST}:50051 --output ${WORKSPACE_PATH}/gsplat.ply
```

## Tests

Each `*_test.py` sits beside the module it covers, and between them they cover
what runs without a GPU, COLMAP, or Triton: the `JobParameters` layering
(`gsplat_server/parameters_test.py`), the training server's job store and
archive handling (`gsplat_server/server_test.py`), the sweep report generator
(`mapping/benchmark/summarize_gsplat_sweep_test.py`), and the reconstruction's
gravity levelling (`mapping/mapping_pipeline_test.py`).

CI runs them on a slim Python image rather than the 20 GB pipeline one. To do
the same locally:

```
docker run --rm -v "$PWD":/workspace -w /workspace python:3.11-slim bash -c '
    pip install -r artifacts/requirements-test.txt
    bash gsplat_server/proto/build.sh
    pytest'
```

Or on the host, with the proto bindings already generated (see
[doc/gsplat_server.md](doc/gsplat_server.md)):
`pip install -r artifacts/requirements-test.txt && pytest`.

`mapping/train_gsplat_with_masks.py` and `mapping/segment_people.py` are
uncovered: both import torch at module scope, and the former also runs the
trainer at import time, so covering its flag building needs a `__main__`
guard first.
