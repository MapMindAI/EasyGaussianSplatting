# EasyGaussianSplatting

An end-to-end pipeline for turning Insta360 360° captures into Gaussian Splatting
reconstructions, using COLMAP's native spherical (equirectangular) camera model to
reconstruct directly from panoramic frames.

## Status

This branch is a ground-up remake of the pipeline. What's done so far:

- [x] Docker image with COLMAP, the Insta360 Media SDK, and ExifTool
- [x] Script to stitch raw Insta360 footage into equirectangular video
- [x] Script to run COLMAP reconstruction with the spherical camera model
- [ ] Gaussian Splatting training/export wired to the above

## What's in the Docker image

Built from `artifacts/docker/dev.dockerfile`:

- **COLMAP** (>= 4.1.0), built with native `EQUIRECTANGULAR` camera model support,
  so panoramic frames can be reconstructed directly without reprojecting to
  perspective views first.
- **Insta360 Media SDK**, exposed as `insta360_media_stitcher`, for stitching raw
  `.insv`/`.lrv` footage into a panorama video or image sequence.
- **ExifTool**, for reading GPS/timestamp metadata off the source footage.
- A conda environment (`gaussian_splatting`) with PyTorch and OpenCV for the
  training step.

## Getting the image

Pull the image CI publishes on every push to `master`:

```
docker pull ghcr.io/mapmindai/gaussiansplatting:latest
```

Or build it locally from your checkout:

```
cd artifacts/docker
docker build -f dev.dockerfile -t easygaussiansplatting:dev .
```

## Using the tools

Run the container with your data directory mounted:

```
docker run -it --rm -v $(pwd)/data:/workspace ghcr.io/mapmindai/gaussiansplatting:latest bash
```

Stitch a raw Insta360 capture into an equirectangular video:

```
insta360_media_stitcher -inputs /workspace/VID_xxx.insv -output /workspace/pano.mp4 -stitch_type optflow
```

Or run `scripts/stitch_pano.sh`, which wraps the AI-stitch settings we use for
reconstruction (8000x4000, H.265, flowstate, direction lock). Override
`INPUT_INSV`, `OUTPUT_VIDEO`, and `MODEL_ROOT_DIR` as needed:

```
INPUT_INSV=data/VID_xxx.insv OUTPUT_VIDEO=data/pano.mp4 scripts/stitch_pano.sh
```

Read metadata off the source file:

```
exiftool /workspace/VID_xxx.insv
```

Run `scripts/colmap_reconstruct.sh <video_path> [frame_rate]` to extract frames
from the panorama video (via `mapping/extract_images.py`) and reconstruct the
scene with COLMAP's `EQUIRECTANGULAR` camera model. `frame_rate` (frames/sec
sampled from the video) defaults to 2. Results land next to the video, in a
`<video_name>_mapping/` directory:

```
scripts/colmap_reconstruct.sh data/pano.mp4 2
```

It runs `feature_extractor`, `sequential_matcher`, and `mapper` on CPU —
COLMAP's default GPU path hard-aborts when the container has no CUDA device,
so the script always requests `use_gpu 0`. The sparse model lands in
`data/pano_mapping/sparse`.

## Running the reconstruction script in Docker

`scripts/colmap_reconstruct.sh` needs the repo checkout itself (for
`mapping/extract_images.py`), not just your data, so mount the whole checkout
instead of only `data/`:

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

Gaussian Splatting training/export isn't wired to the above yet — see Status.
