# EasyGaussianSplatting

An end-to-end pipeline for turning Insta360 360° captures into Gaussian Splatting
reconstructions, reconstructing panoramic frames with COLMAP's `panorama_sfm`
(rig of virtual perspective views, reprojected back to a native equirectangular
camera per frame).

## Running the full pipeline in one command

`scripts/run_pipeline.sh <input.insv> [frame_rate] [output_size] [face_size] [gs_iterations] [gs_data_factor] [gs_floater_reg_weight]`
runs on the host and drives the container itself, chaining stitching, frame
extraction, COLMAP reconstruction, cube-map conversion, and gsplat training
in a single `docker run --gpus all --shm-size=1g`. `input.insv` must live under the repo
checkout (it gets bind-mounted as `/workspace`). `frame_rate` defaults to 2,
`output_size` to `8000x4000`, `face_size` (cube-face width/height in pixels)
to `1024`, `gs_iterations` to `30000`, `gs_data_factor` (a COLMAP-style
downsample factor) to `1`, and `gs_floater_reg_weight` (opacity/scale
regularization strength) to `0.01`:

```
scripts/run_pipeline.sh data/VID_xxx.insv 2 4000x2000
```

Results land next to the capture, in `<capture_name>_reconstruction/`:
`pano.mp4` (the stitched video), `pano_mapping/sparse` (the COLMAP sparse
model), `pano_mapping_cubemap/sparse` (the cube-map model gsplat trains on),
and `pano_mapping_cubemap/gsplat_output` (the trained model). Set
`DOCKER_IMAGE` to use a locally built image instead of the published one. The
script keeps downloaded PyTorch model weights in the persistent Docker volume
`easygaussiansplatting-torch-cache`, so later runs reuse them.
Existing person-segmentation masks are reused on later runs.

![COLMAP sparse reconstruction viewer](assets/reconstruction_viewer.jpg)

## Status

This branch is a ground-up remake of the pipeline. What's done so far:

- [x] Docker image with COLMAP, the Insta360 Media SDK, and ExifTool
- [x] Script to stitch raw Insta360 footage into equirectangular video
- [x] Script to run COLMAP reconstruction via `panorama_sfm`
- [x] Script to convert an equirect COLMAP reconstruction to a cube-map one
- [x] Script to train a Gaussian Splatting model from the cube-map reconstruction (gsplat)
- [ ] Export/viewer wired to the above

## What's in the Docker image

Built from `artifacts/docker/dev.dockerfile`:

- **COLMAP** (>= 4.1.0), built with native `EQUIRECTANGULAR` camera model support.
- **Insta360 Media SDK**, exposed as `insta360_media_stitcher`, for stitching raw
  `.insv`/`.lrv` footage into a panorama video or image sequence.
- **ExifTool**, for reading GPS/timestamp metadata off the source footage.
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
git submodule update --init third_party/gsplat third_party/colmap
docker build -f artifacts/docker/dev.dockerfile -t easygaussiansplatting:dev \
  --build-context gsplatsrc=./third_party/gsplat \
  --build-context colmapsrc=./third_party/colmap artifacts/docker
```

## Using the tools

See [doc/tools.md](doc/tools.md).

## Running the reconstruction script in Docker

`scripts/colmap_reconstruct.sh` needs the repo checkout itself for
`mapping/extract_images.py` and the `third_party/colmap` submodule. Initialize
that submodule and mount the whole checkout instead of only `data/`. The script
installs the matching `pycolmap` wheel and its panorama dependencies in the
disposable container at runtime.

```
git submodule update --init third_party/colmap
```

```
docker run -it --rm -v $(pwd):/workspace -w /workspace \
  ghcr.io/mapmindai/gaussiansplatting:latest \
  scripts/colmap_reconstruct.sh data/pano.mp4 2
```

Or drop into a shell and run it interactively:

```
docker run -it --rm -v $(pwd):/workspace -w /workspace \
  ghcr.io/mapmindai/gaussiansplatting:latest bash

scripts/colmap_reconstruct.sh data/pano.mp4 2
```

The sparse model lands in `data/pano_mapping/sparse` on the host, since
`/workspace` is a bind mount of your checkout.

## Converting to a cube-map reconstruction

gsplat's COLMAP loader only supports perspective/fisheye camera models, not
COLMAP's `EQUIRECTANGULAR` model that `colmap_reconstruct.sh` produces, so the
equirect reconstruction has to be split into a 6-face cube map first.
`scripts/cubemap_convert.sh <reconstruction_dir> [face_size] [faces]`
reprojects each frame into per-face pinhole images and rebuilds the sparse
model with per-face poses/intrinsics. `face_size` (cube-face width/height in
pixels) defaults to `1024`; `faces` (comma-separated subset of
`front,right,back,left,up,down`) defaults to all but `down` — the nadir
usually shows whoever is carrying the rig, so it's excluded unless you pass
all six explicitly:

```
docker run -it --rm -v $(pwd):/workspace -w /workspace \
  ghcr.io/mapmindai/gaussiansplatting:latest \
  scripts/cubemap_convert.sh data/pano_mapping 1024
```

The cube-map model lands in `<reconstruction_dir>_cubemap/` (`images/` +
`sparse/0/`), same shape as `colmap_reconstruct.sh`'s output.

## Training a Gaussian Splatting model

`scripts/gsplat_train.sh <cubemap_reconstruction_dir> [iterations] [data_factor] [floater_reg_weight]`
trains a model with [gsplat](https://github.com/nerfstudio-project/gsplat)
from a `cubemap_convert.sh` output directory. `iterations` defaults to
`30000`, `data_factor` (a COLMAP-style downsample factor) to `1`,
`floater_reg_weight` (opacity/scale regularization strength) to `0.01`:

```
docker run -it --rm --gpus all --shm-size=1g -v $(pwd):/workspace -w /workspace \
  ghcr.io/mapmindai/gaussiansplatting:latest \
  scripts/gsplat_train.sh data/pano_mapping_cubemap 30000 2
```

`scripts/run_gsplat.sh <cubemap_reconstruction_dir> [iterations] [data_factor] [floater_reg_weight]`
is the host-side equivalent: same arguments, but it drives the `docker run`
itself (GPU flags, repo bind-mount, and the persistent PyTorch weight cache),
so retraining an existing cube-map model needs no pipeline rerun. The
directory must live under the repo checkout:

```
scripts/run_gsplat.sh data/panorama 30000 1 0.01
```

The trained model lands in `<cubemap_reconstruction_dir>/gsplat_output/`,
including a final point cloud under `ply/`. The trainer also
enables camera pose refinement, opacity/scale regularization (to suppress
floaters), and antialiased rendering — gsplat defaults these off, but they
consistently help on cube-map-converted panorama captures.
Export/viewer integration isn't wired yet — see Status.

Training renders at the cube face size divided by `data_factor`, and the
rasterizer's per-iteration buffers scale with that pixel count times the
(growing, via densification) number of Gaussians, times 6 (one image per
cube face per frame). On GPUs with less than ~8GB VRAM, drop `face_size` in
`cubemap_convert.sh` and/or raise `data_factor` here to fit. `scripts/gsplat_train.sh`
also sets `PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce
allocator fragmentation from those buffers, which otherwise depletes VRAM
before the process leaks it.
