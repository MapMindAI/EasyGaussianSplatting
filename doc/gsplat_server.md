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

`gsplat_pb2.py` and `gsplat_pb2_grpc.py` are gitignored, and the image ships
`grpcio` but no `protoc`, so generate them in the checkout before the first
start:

```
bash gsplat_server/proto/build.sh
```

Without `protoc` on `PATH` the script uses the one `grpcio-tools` bundles, which
is how CI generates them.

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
```

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

To mask people and sky without segmenting them first, set `run_segmentation: true`
and upload no `masks/`. The server then segments the uploaded images before
training, writing its output into the job's own directory:

```
run_segmentation: true
```

People come from torchvision's COCO-trained Mask R-CNN. When `mask_sky` is
true, sky comes from the SegFormer model in `third_party/EasyTensorRT` (ADE20K
class 2), over gRPC at `TRITON_URL`. A person-only job needs no Triton server.

`mask_sky` selects which of the two is masked. It defaults to true: sky is at
infinity, so the Gaussians that chase it are floaters that cost memory and blur
the geometry below. Set it false to keep the sky and mask only people:

```
run_segmentation: true
mask_sky: false
```

An uploaded `masks/` always wins, so the flag is a fallback rather than an
override. Segmentation adds a few minutes to a job: it runs inference on
`segment_people.py --num-threads` worker threads, through the same helper the
feature stages use.

### Depth supervision

`run_depth: true` predicts depth for the uploaded images with Depth Anything 3
and adds gsplat's depth term to the loss, weighted by `depth_lambda`:

```
run_depth: true
depth_lambda: 0.01
```

It runs after segmentation and before training, writing `depths/` into the job
directory. An uploaded `depths/` always wins, the same way `masks/` does.

DA3 takes a fixed number of views at once, so `generate_depth.py` groups the
model's images by camera and splits each camera into consecutive groups of five
-- for the cube-map rig, neighbouring frames of one face. The served model's
input shape fixes that count, so it is a constant rather than a flag: changing
it means deploying a different model (see `third_party/EasyTensorRT`).

Each group is reconstructed in its own arbitrary scale, which is the part worth
understanding: the depth DA3 returns is not in the input model's units, and a
different group gets a different scale. So each group's depth is fitted to the
COLMAP points its own images already observe, by the median ratio between the
two, before anything is written. A group whose images carry fewer than 20 such
points is skipped rather than written at a guessed scale, and the stage reports
how many it skipped.

What lands in `depths/` is one `<image_name>.npy` per image holding `(M, 3)`
float32 rows of `(x, y, depth)`: the sampled pixels worth supervising rather
than a dense map, which is the difference between megabytes and gigabytes. `x`
and `y` are normalized to `[0, 1]`, so `data_factor` cannot desynchronize them
from the images the trainer loads; the trainer scales them back and converts
depth into gsplat's normalized world.

Three limits to know. gsplat's depth term is an L1 on *inverse* depth, so
near-field error dominates and far geometry is barely constrained. The term is
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
