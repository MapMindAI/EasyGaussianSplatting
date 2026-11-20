# Gaussian Splatting server on Linux

`gsplat_server/` turns a Linux box with an NVIDIA GPU into a training
appliance: clients upload a COLMAP model over gRPC, the server trains it with
gsplat, and they download the resulting point cloud. It carries only the
training half of the pipeline — clients reconstruct their capture themselves
(`mapping/mapping_pipeline.py`), since gsplat's COLMAP loader accepts only
perspective and fisheye cameras.

| File | |
| --- | --- |
| `artifacts/docker/dev.dockerfile` | x86_64 image: COLMAP, the gsplat conda env, and the pipeline tools. |
| `artifacts/docker/installers/install_gsplat.sh` | Image installation script for the gsplat env. |
| `server.py` | The gRPC service and its job queue. |
| `serve.sh` | Container: start the service. |
| `run_server.sh` | Host: start the service via Docker. |
| `client.py` | Command-line gRPC client. Requires `grpcio`. |
| `proto/build.sh` | Generates ignored Python bindings from `proto/gsplat.proto`. |

## Prerequisites

* x86_64 Linux with Docker, an NVIDIA driver, and the NVIDIA Container
  Toolkit, so `docker run --gpus all` reaches the GPU.
* A checkout of this repo on the host, which the container runs bind-mounted at
  `/workspace`.
* `protoc` and the gRPC Python plugin, to generate the bindings below:
  `sudo apt install protobuf-compiler protobuf-compiler-grpc`, or
  `pip install grpcio-tools` for the copy it bundles.

## Getting the image

CI publishes it on every push to `master`:

```
docker pull ghcr.io/mapmindai/gaussiansplatting:latest
```

<details>
<summary>Building it locally</summary>

The `gsplat` submodule has to be checked out and passed in as an extra build
context, so the build doesn't have to send `data/`:

```
git submodule update --init third_party/gsplat
docker build -f artifacts/docker/dev.dockerfile -t easygaussiansplatting:dev \
  --build-context gsplatsrc=./third_party/gsplat artifacts/docker
```

Point `DOCKER_IMAGE` at the local tag to run it instead of the published one.

</details>

## Generating the proto bindings

`gsplat_pb2.py` and `gsplat_pb2_grpc.py` are gitignored, so generate them in
the checkout before the first start:

```
bash gsplat_server/proto/build.sh
```

Without `protoc` on `PATH` the script uses the one `grpcio-tools` bundles, as CI
does; the dev image's gsplat env includes `grpcio-tools`, so it can run there.

Rerun it whenever `proto/gsplat.proto` changes, on both the server host and any
client machine.

<details>
<summary>Verifying GPU access</summary>

```
docker run --rm --gpus all ghcr.io/mapmindai/gaussiansplatting:latest \
  conda run --no-capture-output -n gsplat python3 -c \
  "import torch, gsplat; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The output must include `True` and the GPU's name. A missing driver or
Container Toolkit instead ends in `RuntimeError: No CUDA GPUs are available`,
which is also how jobs would fail.

</details>

## Running the server

From the checkout:

```
gsplat_server/run_server.sh [port] [jobs_dir]
```

`port` defaults to 50051 and `jobs_dir` to `data/gsplat_server`, which is where
uploads, logs, and trained models land. Under Docker `jobs_dir` has to live
under the checkout, because that is what the container sees. The script runs in
the foreground; to start it at boot, wrap it in a systemd unit:

```
[Unit]
Description=gsplat training server
After=docker.service
Requires=docker.service

[Service]
WorkingDirectory=/srv/EasyGaussianSplatting
ExecStart=/srv/EasyGaussianSplatting/gsplat_server/run_server.sh 50051 data/gsplat_server
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

The unit runs as root, which is also how the container runs, so everything
under `jobs_dir` is root-owned; a `User=` needs `docker` group membership.

`run_server.sh` passes `--gpus all` (override `DOCKER_GPU_FLAGS`) and
`--shm-size=8g`. The shared memory matters: below roughly 8g the trainer's
dataloader workers die partway through a run with a bus error, which surfaces
as a failed job whose log ends in `DataLoader worker ... killed by signal: Bus
error`.

Training, segmentation, and depth all use the `gsplat` interpreter that starts
the server.

Jobs run one at a time — a single training run already saturates the GPU — and
the queue lives in memory, so a restart fails whatever was queued or training.
Finished jobs survive it. A queued job carries its `queue_position`: the jobs
the worker trains before it. The clock for that is `queued_at`, stamped when
the upload finishes and the job reaches the worker, not `created_at`, which is
stamped when the upload starts — so a slow upload does not appear to hold up
jobs submitted after it.

Stop queued work and terminate the active job without restarting the service:

```
gsplat_server/client.py --server gsplat-host:50051 --stop-all
```

## Submitting a model

`client.py` zips a reconstruction, uploads it, follows the training log, and
downloads the point cloud. While the job is queued it prints `Queued behind N
jobs` from the server's `queue_position`, so a wait behind another run is
visible rather than silent. The client requires Python 3 and `grpcio` and loads
`gsplat_server/config/gsplat_train_defaults.proto.txt` for training parameters:
```
gsplat_server/client.py data/pano_mapping --server gsplat-host:50051 \
  --output pano.ply
```
The downloaded point cloud is in the uploaded `sparse/` frame: gsplat
normalizes world space for training (rotate, recenter, rescale), and the server
reverses that similarity on export. So a point cloud overlays the COLMAP model
it was trained from, and `local_to_world.json` still takes it to UTM.

Pass `--parameters` with another text-format JobParameters file for a custom
run. It is layered over the shipped defaults, so it only names what it changes:

```
cap_max: 1500000
min_opacity: 0.01
```

For the Wixi Chongansi drone scene, use
`gsplat_server/config/gsplat_train_drone.proto.txt`; its
`opacity_reg_weight` and `scale_reg_weight` of 0.003 produced the best reported PSNR and SSIM in the
[scene benchmark](https://github.com/MapMindAI/EasyGaussianSplatting/wiki/Benchmark---Wixi-Chongansi-Drone-Scene).

With MCMC, `min_opacity` relocates Gaussians at or below the threshold during
refinement; it does not reduce their count. `prune_opa` applies only to
DefaultStrategy.

The Gaussian learning rates match `simple_trainer` by default and can be
overridden as `means_lr`, `scales_lr`, `opacities_lr`, `quats_lr`, `sh0_lr`,
or `shN_lr`; `pose_opt_lr` controls pose optimization when `pose_opt` is true.

Every `test_every`-th image (default 8) is held out of training and rendered,
scored and saved once at the end, so raise it on large captures, e.g.
`test_every: 64`.

The defaults are `STRATEGY_MCMC` at 900k Gaussians, which measured best on
`data/panorama`. `STRATEGY_MCMC` holds the count at `cap_max`, so it is how to
ask for a specific budget on a given GPU; quality saturated around 900k there,
and 1.5M and 2M were no better.

`STRATEGY_DEFAULT` instead grows the count from the image-plane gradient, shaped
by `grow_grad2d`, `prune_opa`, `reset_every` and `absgrad`. It is harder to
operate: `prune_opa` also sets the opacity reset floor (gsplat resets to
`prune_opa * 2`), so lowering it does not simply prune less -- combined with
`reset_every` it can collapse a model to a few thousand Gaussians. See
[doc/gsplat_parameter_sweep.md](gsplat_parameter_sweep.md) for measured
comparisons.

The directory must hold `images/` and `sparse/`; a `masks/` directory is
uploaded too if present, and training then skips the masked pixels (see
`mapping/triton/segment_people.py`). Nothing else is uploaded, so an earlier
`gsplat_output/` in the same directory costs nothing.

Set `mask_people` and `mask_sky` independently to generate masks before
training when the upload has no `masks/`:

```
mask_people: true
mask_sky: true
```

`mask_people` runs torchvision's COCO-trained Mask R-CNN locally. `mask_sky`
runs the U²-Net sky model at `models/skyseg.onnx` locally (see
[tools.md](tools.md#sky-segmentation)), so that file has to be present in the
repo checkout the server runs from. Both default to false; enabling both
combines their masks.

Before upgrading, rename `run_segmentation` to `mask_people` in existing
parameter files and saved `job.json` parameters. Protobuf field number 20
is unchanged.

`opacity_reg_weight` and `scale_reg_weight` replace `floater_reg_weight`,
which set both. Rename it to `opacity_reg_weight` (field number 3 is unchanged)
and add `scale_reg_weight` with the same value to keep a run's behaviour; an
omitted `scale_reg_weight` takes the shipped default.

`stop_regularization_after_refinement: true` drops both terms from
`refine_stop_iter` on. With MCMC nothing relocates after that step, so the
opacity term only dims every Gaussian; position noise still runs.

An uploaded `masks/` always wins. Segmentation runs inference on
`segment_people.py --num-threads` worker threads, through the same helper the
feature stages use.

### Depth supervision

`run_depth: true` predicts depth for the uploaded images with VGGT-Omega,
seeds the Gaussians from the sparse SfM points plus the merged depth cloud
(`depth_points.ply`), and adds gsplat's depth term to the loss, weighted by
`depth_lambda`; `depth_lambda: 0` keeps the initialization without the term. Under MCMC, the depth
points are randomly thinned to keep the total within 0.8 × `cap_max`.
`depth_voxel_factor` sets the cloud's voxel edge as a fraction of the scene
scale:

```
run_depth: true
depth_lambda: 0.01
depth_voxel_factor: 0.001
```

It runs after segmentation and before training, writing `depths/` into the job
directory. An uploaded `depths/` always wins, the same way `masks/` does.

VGGT-Omega reconstructs several views at once, so
`mapping/vggt_omega/generate_depth.py` groups up to 12 images that view the
same part of the scene, by their shared sparse points and relative poses (see
[tools.md](tools.md#depth-from-triton)). It runs the ONNX export at
`models/vggt_omega_416.onnx` on the server's own GPU, so that file has to be
present in the repo checkout the server runs from (see
[tools.md](tools.md#vggt-omega-instead-of-da3)). The Triton DA3 stage,
`mapping/depth/generate_depth.py`, stays available for manual runs.

Server preprocessing runs in the `gsplat` environment, which includes ONNX
Runtime and pycolmap. Rebuild the image after
upgrading an existing server so depth and sky masking use those dependencies.

Each group is reconstructed in its own arbitrary scale, which is the part worth
understanding: the depth VGGT-Omega returns is not in the input model's units,
and a different group gets a different scale. So each group's depth is fitted to the
COLMAP points its own images observe, and to the COLMAP camera baselines
against the model's predicted ones, before anything is written. A group with
neither enough sparse points nor consistent cameras is skipped rather than
written at a guessed scale, and the stage reports how many it skipped. An image
in several groups keeps the best-matching prediction. An image left without
rows trains without a depth term:
gsplat's own sparse-point depths do not stand in for them, since the images it
skips are the ones SfM barely covered anyway.

What lands in `depths/` is one `<image_name>.npy` per image holding `(M, 3)`
float32 rows of `(x, y, depth)`: the sampled pixels worth supervising rather
than a dense map, which is the difference between megabytes and gigabytes. `x`
and `y` are normalized to `[0, 1]`, so `data_factor` cannot desynchronize them
from the images the trainer loads; the trainer scales them back and converts
depth into gsplat's normalized world.

Three limits to know. gsplat's depth term is an L1 on *inverse* depth, so
near-field error dominates and far geometry is barely constrained -- and a
pixel no Gaussian covers yet renders a depth near zero, whose disparity would
bury every other term, so those are left out of it until geometry reaches
them. The term is
not masked: `masks/` gates the L1 and SSIM terms only, so depth supervision is
confined to the masked-in pixels here instead, when the segmentation stage has
written masks by the time depth runs. And depth is predicted on the uploaded
images as they are, so a camera with distortion parameters would have its depth
registered against the raw image while training renders the undistorted one --
the cube-map rig registers `PINHOLE`, which COLMAP leaves undistorted, so this
does not arise today.

### Background colour

`background_color` is what the splats are composited over, so whatever no
Gaussian covers -- the masked sky above all -- renders as that colour instead of
black:

```
background_color { red: 0.53 green: 0.71 blue: 0.92 }
```

Channels are in `[0, 1]`. Leave the field out and gsplat renders onto black as
before. It reaches the rasterizer for every render, so it shapes the trained
model rather than only the preview: a background the sky is plausibly near
leaves less for stray Gaussians to explain. The exported PLY is unaffected --
it carries Gaussians, not a background -- so a viewer still paints its own.

## Other hosts

* [doc/gsplat_server_orin.md](gsplat_server_orin.md) — Jetson AGX Orin: the
  arm64 image, `--runtime nvidia`, and a direct install without Docker.
* [doc/gsplat_server_wins.md](gsplat_server_wins.md) — Windows: the same x86
  image under Docker Desktop, with the checkout on `D:`.

## gRPC API

| Request | |
| --- | --- |
| `SubmitJob` (client stream) | First message has parameters; remaining messages carry zip chunks. |
| `ListJobs` | Every job, oldest first. |
| `GetJob` | One job, its state, and its queue position. |
| `StreamLog` | Training log chunks from a byte offset. |
| `DownloadResult` | Streams the trained point cloud once the job succeeds. |
| `DeleteJob` | Drop a finished job and its files. |
| `StopAllJobs` | Stop the running job and fail queued jobs. |
