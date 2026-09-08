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

`mapping/mapping_pipeline.py` reprojects the panorama frames into a rig of
pinhole cube faces, extracts SuperPoint and SALAD features, matches with
LightGlue, and solves the scene with global SfM. It needs the repo checkout
itself, for `mapping/` and the `third_party/EasyTensorRT` submodule, so mount
the whole checkout rather than only `data/`, and it needs a Triton server
serving those three models:

```
git submodule update --init third_party/EasyTensorRT
```

```
docker run -it --rm -v $(pwd):/workspace -w /workspace \
  --add-host host.docker.internal:host-gateway \
  ghcr.io/mapmindai/gaussiansplatting:latest \
  python3 -m mapping.mapping_pipeline \
    --video_path data/pano.mp4 --workspace_path data/pano_mapping \
    --triton-url host.docker.internal:8001
```

The reconstruction lands in `data/pano_mapping/` (`images/<face>/` +
`sparse/0/`) on the host, since `/workspace` is a bind mount of your checkout.
[doc/panorama_mapping.md](doc/panorama_mapping.md) covers serving the models,
each stage, and the pair-selection knobs.

## Training a Gaussian Splatting model

The pipeline's last stage masks out people and trains a model with
[gsplat](https://github.com/nerfstudio-project/gsplat) from the cube-map
reconstruction. Every setting comes from the `JobParameters` file passed as
`run_pipeline.sh`'s fifth argument: `iterations` defaults to `30000`,
`data_factor` (a COLMAP-style downsample factor) to `1`, and
`floater_reg_weight` (opacity/scale regularization strength) to `0.01`.

The trained model lands in `<reconstruction_dir>/gsplat_output/`, including a
final point cloud under `ply/`. The trainer also enables camera pose
refinement, opacity/scale regularization (to suppress floaters), and
antialiased rendering — gsplat defaults these off, but they consistently help
on cube-map panorama captures. Export/viewer integration isn't wired yet — see
Status.

Every stage skips work already on disk, so re-running the script after a
parameter change redoes only what is missing. To retrain alone, delete
`gsplat_output/`.

Training renders at the cube face size divided by `data_factor`, and the
rasterizer's per-iteration buffers scale with that pixel count times the
(growing, via densification) number of Gaussians, times one image per cube face
per frame. On GPUs with less than ~8GB VRAM, drop `face_size` and/or raise
`data_factor` to fit. The script also sets
`PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce allocator fragmentation
from those buffers, which otherwise depletes VRAM before the process leaks it.
