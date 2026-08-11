# Using the tools

Run the container with your data directory mounted. Add `--gpus all` if the host
has an NVIDIA GPU — `insta360_media_stitcher` uses it for hardware video
encode/decode (NVENC/NVDEC) and falls back to software encoding without it:

```
docker run -it --rm --gpus all --shm-size=1g -v $(pwd)/data:/workspace ghcr.io/mapmindai/gaussiansplatting:latest bash
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
`<video_name>_mapping/` directory. Terminal output is also appended to
`<video_name>_mapping/colmap_reconstruct.log`:

```
scripts/colmap_reconstruct.sh data/pano.mp4 2
```

It runs `feature_extractor`, `sequential_matcher`, and `mapper` on CPU —
COLMAP's default GPU path hard-aborts when the container has no CUDA device,
so the script always requests `use_gpu 0`. The sparse model lands in
`data/pano_mapping/sparse`.

Run `scripts/cubemap_convert.sh <reconstruction_dir> [face_size] [faces]` to
turn that equirect reconstruction into a 6-face cube map (gsplat only
supports perspective/fisheye COLMAP camera models, not `EQUIRECTANGULAR`).
`face_size` defaults to 1024; `faces` defaults to all but `down`
(`front,right,back,left,up`) since the nadir usually shows whoever is
carrying the rig — pass all six explicitly to include it:

```
scripts/cubemap_convert.sh data/pano_mapping 1024
```

The cube-map model lands in `data/pano_mapping_cubemap`.

Run `scripts/gsplat_train.sh <cubemap_reconstruction_dir> [iterations] [data_factor]`
to train a [gsplat](https://github.com/nerfstudio-project/gsplat) model from
that cube-map reconstruction. `iterations` defaults to 30000, `data_factor`
(a COLMAP-style downsample factor) to 1. The container needs at least 1 GiB of
shared memory for gsplat's data-loader workers:

```
scripts/gsplat_train.sh data/pano_mapping_cubemap 30000 2
```

The trained model lands in `data/pano_mapping_cubemap/gsplat_output`.
