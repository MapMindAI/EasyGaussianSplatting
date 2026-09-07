# Gaussian Splatting server on a Jetson AGX Orin

The same gRPC service as [doc/gsplat_server.md](gsplat_server.md), run on an
Orin. That doc covers the service itself — the client, the training parameters,
and the gRPC API; only the Orin-specific parts are here.

## Docker deployment

CI builds only the x86 image. Build the arm64 one on an Orin as below, or pull
it if it has already been pushed:

```
docker pull ghcr.io/mapmindai/gaussiansplatting-jetson:latest
```

It is JetPack 6 (L4T r36.4, CUDA 12.6) with gsplat installed into the system
Python rather than a conda env, plus COLMAP for inspection and repair.
`--runtime nvidia` is how JetPack exposes the GPU, rather than the `--gpus all`
the host scripts default to:

```
DOCKER_IMAGE=ghcr.io/mapmindai/gaussiansplatting-jetson:latest \
  DOCKER_GPU_FLAGS="--runtime nvidia" gsplat_server/run_server.sh [port] [jobs_dir]
```

Generate the proto bindings in the checkout first, and see
[doc/gsplat_server.md](gsplat_server.md) for the arguments and the systemd
unit.

<details>
<summary>Building the image locally</summary>

Build on arm64 — cross-building the CUDA extensions under QEMU takes hours. The
submodules come in as extra build contexts so the build doesn't have to send
`data/`:

```
git submodule update --init third_party/gsplat third_party/colmap
docker build -f artifacts/docker/jetson.dockerfile -t gaussiansplatting-jetson:dev \
  --build-context gsplatsrc=./third_party/gsplat \
  --build-context colmapsrc=./third_party/colmap \
  --build-context reposrc=. artifacts/docker
```

`install_gsplat_orin.sh` fetches torch,
torchvision, and pycolmap as direct wheel URLs from NVIDIA's Jetson index,
because PyPI's aarch64 wheels are server-Grace builds that won't run on an Orin
(and it has no aarch64 pycolmap at all). And the CUDA extensions are compiled
for compute capability 8.7 only, which is the Orin's — this image will not run
on another GPU. COLMAP is built from the checked-out submodule with CUDA
disabled; CUDA stays available to gsplat through JetPack.

</details>

<details>
<summary>Verifying GPU access</summary>

```
docker run --rm --runtime nvidia ghcr.io/mapmindai/gaussiansplatting-jetson:latest \
  python3 -c \
  "import torch, gsplat; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

The output must include CUDA `12.6` and `Orin`. An image built for another
board fails here with `no kernel image is available for execution`.

</details>

## Direct installation

This runbook records the direct installation tested on a Jetson AGX Orin running
JetPack 6 / L4T r36.5 and CUDA 12.6. It installs the training environment into
the user's Python environment; it does not install COLMAP.

### 1. Prepare the Orin

Install the CUDA compiler and development headers. The runtime included with
JetPack is not enough to compile gsplat's PyTorch extension:

```bash
sudo apt-get update
sudo apt-get install -y cuda-nvcc-12-6
sudo apt-get install -y libcusparse-dev-12-6 libcublas-dev-12-6
sudo apt-get install -y cuda-libraries-dev-12-6
```

Confirm that the compiler is available:

```bash
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH
nvcc --version
```

### 2. Copy the checkout

The Orin needs the repository, the gsplat submodule, and the training scripts.
From the development machine:

```bash
git submodule update --init third_party/gsplat
rsync -a --exclude data/ --exclude .git/ --exclude '__pycache__/' \
  ./ dm@192.168.19.119:/home/dm/Development/EasyGaussianSplatting/
```

On the Orin, set the checkout and gsplat paths:

```bash
cd /home/dm/Development/EasyGaussianSplatting
export GSPLAT_DIR="$PWD/third_party/gsplat"
```

### 3. Install Python and CUDA dependencies

The Orin uses NVIDIA's Jetson wheels. PyPI's aarch64 PyTorch wheels target
server Grace systems and are not suitable for Jetson.

```bash
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH
export CUDA_HOME=/usr/local/cuda
python3 -m pip install --user --no-cache-dir ninja

bash artifacts/docker/installers/install_gsplat_orin.sh
```

On a clean JetPack installation, install cuDSS if importing torch fails with
`ImportError: libcudss.so.0`:

```bash
python3 -m pip install --user --no-cache-dir nvidia-cudss-cu12==0.8.0.10
```

The package also installs the matching user-space cuBLAS libraries. Make them
visible when running Python:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.10/site-packages/nvidia/cu12/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/cublas/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
```

### 4. Build gsplat for Orin

The extension must be compiled for the AGX Orin's `sm_87` architecture. The
installer sets this automatically. The build is CPU-intensive and can take a
long time on the Orin, especially without Ninja.

```bash
export TORCH_CUDA_ARCH_LIST=8.7
export MAX_JOBS=4
export CMAKE_BUILD_PARALLEL_LEVEL=4
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.10/site-packages/nvidia/cu12/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/cublas/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

cd "$GSPLAT_DIR"
python3 -m pip install --user --no-cache-dir --no-build-isolation -e .
```

CUDA 12.6 does not provide the `cuda::ceil_div` helper expected by the pinned
gsplat sources. The checked-out source includes the equivalent integer ceiling
division implementation in `third_party/gsplat/gsplat/cuda/csrc/`.

The installer also installs grpcio and the supported gsplat example dependencies. If the installer is interrupted during compilation, rerun the editable install from the gsplat checkout with the same environment variables.

Verify the environment and GPU:

```bash
python3 -c 'import torch, gsplat; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))'
```

Expected output includes PyTorch `2.9.1`, CUDA `12.6`, and `Orin`.

### 5. Install server dependencies

Install the dependencies used by the gRPC server and the supported gsplat
examples. The Orin installer omits optional packages without a Jetson wheel.

```bash
cd /home/dm/Development/EasyGaussianSplatting
grep -vE '^(pycolmap|ppisp|.*fused-bilagrid)' \
  third_party/gsplat/examples/requirements.txt > /tmp/gsplat-requirements.txt
python3 -m pip install --user --no-cache-dir --no-build-isolation \
  -r /tmp/gsplat-requirements.txt grpcio
```

### 6. Start and test the server

The direct installation runs the server with Python rather than Docker. Keep
the CUDA library variables in the same shell used to start it:

```bash
cd /home/dm/Development/EasyGaussianSplatting
nohup env PATH="$PATH" CUDA_HOME=/usr/local/cuda \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  bash gsplat_server/serve.sh /home/dm/gsplat_server_test 50051 \
  >/tmp/gsplat-server.log 2>&1 &
```

Check that the server is listening:

```bash
ss -ltn | grep ':50051'
```

The client accepts a reconstruction directory containing `images/` and
`sparse/`, uploads it to the Orin, follows the training log, and downloads the
resulting PLY file:

```bash
python3 gsplat_server/client.py \
  /path/to/cubemap_reconstruction \
  --server 192.168.19.119:50051 \
  --parameters /tmp/orin-test.proto.txt \
  --output /tmp/orin-test.ply
```

The client loads training parameters from gsplat_server/config/gsplat_train_defaults.proto.txt by default. Pass --parameters with another text-format JobParameters file for a smoke test or custom run.
The server queue is in memory, while uploaded jobs, logs, and results are kept
under the jobs directory.

### Notes

- The direct path installs the training side only. Run COLMAP on the client or
  install it separately on the Orin before attempting a full reconstruction.
- The first gsplat build can consume tens of minutes and several gigabytes of temporary storage; later imports reuse the installed extension.
- After installation, reclaim download and package-manager cache space with python3 -m pip cache purge and sudo apt-get clean.
- Docker remains the reproducible deployment path. Use this runbook when
  avoiding image build and startup overhead is more important than isolation.
