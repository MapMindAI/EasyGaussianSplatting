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

Run `scripts/gsplat_train.sh <cubemap_reconstruction_dir> [iterations] [data_factor] [floater_reg_weight]`
to train a [gsplat](https://github.com/nerfstudio-project/gsplat) model from
that cube-map reconstruction. `iterations` defaults to 30000, `data_factor`
(a COLMAP-style downsample factor) to 1, and `floater_reg_weight`
(opacity/scale regularization strength) to 0.01. The container needs at least
1 GiB of shared memory for gsplat's data-loader workers:

```
scripts/gsplat_train.sh data/pano_mapping_cubemap 30000 2
```

The densification settings live in `scripts/gsplat_train_defaults.sh`, shared
with the sweep so its baseline cannot drift from what ships; `mapping/train_gsplat_with_masks.py` only attaches the masks and passes every argument through to simple_trainer. The trained model lands in `data/pano_mapping_cubemap/gsplat_output`.

Person segmentation masks mirror any nested folders under the reconstruction's
`images/` directory, so each mask keeps the same relative image path and adds
`.png` to the image filename.

## Comparing parameter configurations

`scripts/gsplat_sweep.sh` trains one model per configuration from a single
cube-map reconstruction, into `<reconstruction>/sweep/<name>/`. Runs are
sequential, and a configuration that already recorded evaluation metrics is
skipped, so an interrupted sweep resumes where it stopped:

```
scripts/gsplat_sweep.sh data/panorama 30000
```

Edit the `CONFIGURATIONS` array in that script to change what is compared; each
entry is a name, a simple_trainer subcommand (`default` or `mcmc`) and the
flags under test. The report is generated output and is not tracked: it links
to point clouds and thumbnails under `data/`, so regenerate it rather than
reading a stale copy.

`mapping/summarize_gsplat_sweep.py` then writes a Markdown comparison of those
runs -- evaluation metrics, the geometry their point clouds actually contain,
the settings that differ between them, and each run's renders beside the ground
truth -- to `doc/gsplat_parameter_sweep.md`. Runs still training show as pending:

```
python3 mapping/summarize_gsplat_sweep.py --runs-root data/panorama/sweep
```

The `Radius / spacing` column is the one that explains lost detail: it divides
the median Gaussian radius by the estimated median distance to the nearest
neighbour, so a value well below 1 means the Gaussians are too small to tile a
surface and leave gaps.
