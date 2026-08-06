# EasyGaussianSplatting

An end-to-end pipeline for turning Insta360 360° captures into Gaussian Splatting
reconstructions, using COLMAP's native spherical (equirectangular) camera model to
reconstruct directly from panoramic frames.

## Status

This branch is a ground-up remake of the pipeline. What's done so far:

- [x] Docker image with COLMAP, the Insta360 Media SDK, and ExifTool
- [ ] Script to stitch raw Insta360 footage into equirectangular video
- [ ] Script to run COLMAP reconstruction with the spherical camera model
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

Read metadata off the source file:

```
exiftool /workspace/VID_xxx.insv
```

Extract frames from the panorama video, then feed them to COLMAP using the
spherical camera model:

```
colmap feature_extractor \
  --database_path /workspace/db.db \
  --image_path /workspace/images \
  --ImageReader.camera_model EQUIRECTANGULAR
colmap exhaustive_matcher --database_path /workspace/db.db
colmap mapper --database_path /workspace/db.db --image_path /workspace/images --output_path /workspace/sparse
```

The stitching and reconstruction steps above aren't automated into a single
script yet — see Status.
