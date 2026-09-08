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

`scripts/run_pipeline.sh` stitches with the AI-stitch settings we use for
reconstruction (8000x4000, H.265, flowstate, direction lock), and skips the
step when the video is already there and decodes at the requested size.

Read metadata off the source file:

```
exiftool /workspace/VID_xxx.insv
```

Run `python3 -m mapping.mapping_pipeline --video_path <video> --workspace_path <dir>`
to reconstruct the panorama video as a cube-map rig: SuperPoint and SALAD features
into a COLMAP database, LightGlue matching, and global SfM. It needs a Triton
server serving those models — see
[panorama_mapping.md](panorama_mapping.md), which also covers each stage and
the pair-selection knobs. `--frame-rate` (frames/sec sampled from the video)
defaults to 2, `--face-size` to a quarter of the video width, and `--faces` to
`front,right,back,left,up` since the nadir usually shows whoever is carrying
the rig — pass all six explicitly to include it:

```
python3 -m mapping.mapping_pipeline \
  --video_path data/pano.mp4 --workspace_path data/pano_mapping \
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

Person segmentation masks mirror any nested folders under the reconstruction's
`images/` directory, so each mask keeps the same relative image path and adds
`.png` to the image filename.

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
