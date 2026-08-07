# Using the tools

Run the container with your data directory mounted. Add `--gpus all` if the host
has an NVIDIA GPU — `insta360_media_stitcher` uses it for hardware video
encode/decode (NVENC/NVDEC) and falls back to software encoding without it:

```
docker run -it --rm --gpus all -v $(pwd)/data:/workspace ghcr.io/mapmindai/gaussiansplatting:latest bash
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
