# Gaussian Splatting server on a Jetson AGX Orin

`gsplat_server/` turns an Orin into a training appliance: clients upload a COLMAP
model over gRPC, the Orin trains it with gsplat, and they download the resulting
point cloud. It carries only the training half of the pipeline — clients run
COLMAP and `scripts/cubemap_convert.sh` themselves, since gsplat's COLMAP loader
accepts only perspective and fisheye cameras.

| File | |
| --- | --- |
| `jetson.dockerfile` | arm64 image: JetPack 6 (L4T r36.4, CUDA 12.6) plus gsplat. |
| `install_gsplat.sh` | Image installation script. |
| `server.py` | The gRPC service and its job queue. |
| `serve.sh` | Container: start the service. |
| `run_server.sh` | Host: start the service via Docker. |
| `client.py` | Command-line gRPC client. Requires `grpcio`. |
| `proto/build.sh` | Generates ignored Python bindings from `proto/gsplat.proto`. |

## Getting the image

CI publishes it on every push to `master`:

```
docker pull ghcr.io/mapmindai/gaussiansplatting-jetson:latest
```

Building it locally has to happen on arm64 — cross-building the CUDA extensions
under QEMU takes hours. As with the x86 image, gsplat comes from a separate
build context so the build doesn't have to send `data/`:

```
git submodule update --init third_party/gsplat
docker build -f artifacts/docker/jetson.dockerfile -t gaussiansplatting-jetson:dev \
  --build-context gsplatsrc=./third_party/gsplat \
  --build-context colmapsrc=./third_party/colmap \
  --build-context reposrc=. artifacts/docker
```

For a direct install, initialize the submodule and run the shared installer from the checkout:

```
git submodule update --init third_party/gsplat
GSPLAT_DIR="$PWD/third_party/gsplat" bash artifacts/docker/installers/install_gsplat.sh
```

This avoids Docker startup overhead, but Docker keeps the CUDA/Python environment reproducible and is easier to roll back.

Two pins are worth knowing about. The image builds COLMAP from the checked-out submodule with CUDA disabled; CUDA remains available to gsplat through JetPack. `install_gsplat.sh` fetches torch,
torchvision, and pycolmap as direct wheel URLs from NVIDIA's Jetson index,
because PyPI's aarch64 wheels are server-Grace builds that won't run on an Orin
(and it has no aarch64 pycolmap at all). And the CUDA extensions are compiled
for compute capability 8.7 only, which is the Orin's — the image will not run on
another GPU.

## Running the server

On the Orin, from a checkout of this repo:

```
gsplat_server/run_server.sh [port] [jobs_dir]
```

`port` defaults to 50051 and `jobs_dir` to `data/gsplat_server`, relative to the
checkout, which is where uploads, logs, and trained models land. The script runs
in the foreground; wrap it in a systemd unit to start it at boot. It passes
`--runtime nvidia`, which is how JetPack exposes the GPU; override
`DOCKER_GPU_FLAGS` if your daemon is configured for `--gpus all` instead.

Jobs run one at a time — a single training run already saturates the GPU — and
the queue lives in memory, so a restart fails whatever was queued or training.
Finished jobs survive it.

## Submitting a model

`client.py` zips a reconstruction, uploads it, follows the training log, and
downloads the point cloud. The client requires Python 3 and `grpcio` and loads
`gsplat_server/config/gsplat_train_defaults.proto.txt` for training parameters:
```
gsplat_server/client.py data/pano_mapping_cubemap --server orin:50051 \
  --output pano.ply
```
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
`doc/gsplat_parameter_sweep.md` for measured comparisons.

The directory must hold `images/` and `sparse/`; a `masks/` directory is
uploaded too if present, and training then skips the masked pixels (see
`scripts/segment_people.sh`). Nothing else is uploaded, so an earlier
`gsplat_output/` in the same directory costs nothing.

To mask people without segmenting them first, set `run_segmentation: true` and
upload no `masks/`. The server then segments the uploaded images before
training, writing its output into the job's own directory:

```
run_segmentation: true
```

An uploaded `masks/` always wins, so the flag is a fallback rather than an
override. Segmentation runs on the server's GPU and adds a few minutes, plus a
one-off download of the Mask R-CNN weights on the first such job.

## gRPC API

| Request | |
| --- | --- |
| `SubmitJob` (client stream) | First message has parameters; remaining messages carry zip chunks. |
| `ListJobs` | Every job, oldest first. |
| `GetJob` | One job and its state. |
| `StreamLog` | Training log chunks from a byte offset. |
| `DownloadResult` | Streams the trained point cloud once the job succeeds. |
| `DeleteJob` | Drop a finished job and its files. |
