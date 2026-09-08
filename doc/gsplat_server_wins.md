# Gaussian Splatting server on Windows Docker Desktop

This guide runs the x86 images in Docker Desktop on a Windows PC with an NVIDIA
GPU: the Triton inference server the mapping pipeline needs, the mapping
pipeline itself, and the gsplat training server on port `50051`. The project is
stored on `D:`. [doc/gsplat_server.md](gsplat_server.md) covers the training
service and [doc/panorama_mapping.md](panorama_mapping.md) the mapping stages;
only the Windows-specific parts are here.

## Prerequisites

- Windows 11 with Docker Desktop using the `desktop-linux` context.
- NVIDIA driver with Docker GPU support.
- SSH access to the Windows host, for example `dm@192.168.11.194`.
- A checkout containing `gsplat_server/`, `mapping/`, the generated proto
  bindings, and — for the mapping pipeline — the `third_party/EasyTensorRT`
  submodule, which carries the models and is around 800 MB.

The Jetson image is ARM64 and must not be used on this PC. Use the x86 image:

```
ghcr.io/mapmindai/gaussiansplatting:latest
```

## Copy the project to `D:`

Generate the ignored Python protobuf bindings and check out the models first;
neither is in a fresh clone, and the archive below has to carry both:

```
bash gsplat_server/proto/build.sh
git submodule update --init third_party/EasyTensorRT
```

Create an archive without local datasets or Git history, copy it to the host,
and extract it to `D:\EasyGaussianSplatting`:

```
tar --exclude=data --exclude=.git --exclude='__pycache__' \
  --exclude='*.pyc' -czf /tmp/easy-gsplat-server.tar.gz .
scp /tmp/easy-gsplat-server.tar.gz dm@192.168.11.194:C:/Users/49451/
```

On Windows:

```
mkdir D:\EasyGaussianSplatting
tar -xf C:\Users\49451\easy-gsplat-server.tar.gz \
  -C D:\EasyGaussianSplatting
```

## Verify Docker GPU access

Select Docker Desktop's Linux engine and run:

```
docker context use desktop-linux
docker run --rm --gpus all ghcr.io/mapmindai/gaussiansplatting:latest \
  conda run --no-capture-output -n gsplat python3 -c \
  "import torch, gsplat; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The expected result includes `True` and the NVIDIA GPU name.

## Start the TensorRT inference server

The mapping pipeline extracts and matches learned features against a Triton
server, which serves them out of `third_party/EasyTensorRT`. Start it first:

```
docker run -d --name tritonserver_trt --gpus all -p 8011:8001 \
  -v D:\EasyGaussianSplatting\third_party\EasyTensorRT:/repo \
  ghcr.io/mapmindai/tritonserver_amd64:latest \
  bash -c "bash /repo/model_trt/convert_models.sh && tritonserver --model-repository=/repo/model_repository_trt"
```

`convert_models.sh` builds a TensorRT plan per model for this GPU into
`model_repository_trt/<model>/1/`, so the first start takes several minutes and
is not listening until it finishes; later starts reuse the plans. Follow it
with `docker logs -f tritonserver_trt` and wait for:

```
Started GRPCInferenceService at 0.0.0.0:8001
```

Triton's own gRPC port is 8001, published here as 8011 because 8001 is so often
already taken; 8011 is what this repo defaults to.

Serving the ONNX models instead skips the plan build and runs slower: point
`--model-repository` at `/repo/model_repository` and leave out the conversion
step.

## Reconstruct a capture

With the inference server up, turn a stitched panorama video into the cube-map
model the trainer takes. `data/` is excluded from the archive above, so copy the
video to `D:\EasyGaussianSplatting\data\` first:

```
docker run --rm --gpus all -v D:\EasyGaussianSplatting:/workspace -w /workspace \
  ghcr.io/mapmindai/gaussiansplatting:latest \
  python3 -m mapping.mapping_pipeline \
    --video_path data/pano.mp4 --workspace_path data/pano_mapping \
    --triton-url host.docker.internal:8011
```

Docker Desktop maps `host.docker.internal` to the host itself, so unlike on
Linux no `--add-host` is needed. That is the only Windows-specific part;
[panorama_mapping.md](panorama_mapping.md) covers the stages and the options.

The workspace it writes — `images/<face>/`, `database.db`, `sparse/0/` — is what
the training service below takes.

## Start the training server

`run_server.sh` sets the GPU flags and a shared-memory size the dataloader
survives; `serve.sh` activates the image's conda environment and points the
trainer at the installed gsplat.
Run it from the checkout on `D:`:

```
gsplat_server/run_server.sh 50051 data/gsplat_server
```

To keep the server across reboots, run the same image detached instead:

```
docker run -d --name gsplat-server --gpus all --shm-size=8g -p 50051:50051 \
  -v D:\EasyGaussianSplatting:/workspace -w /workspace \
  ghcr.io/mapmindai/gaussiansplatting:latest \
  gsplat_server/serve.sh /workspace/data/gsplat_server 50051
```

`--shm-size` matters: below roughly 8g the trainer's dataloader workers die
partway through a run with a bus error, which surfaces as a failed job whose
log ends in `DataLoader worker ... killed by signal: Bus error`.

Check status and logs:

```
docker ps --filter name=gsplat-server
docker logs --tail=100 gsplat-server
```

## Test the service

The service can be checked inside the container:

```
docker exec gsplat-server conda run --no-capture-output -n gsplat python3 -c \
  "import grpc; from gsplat_server.proto import gsplat_pb2, gsplat_pb2_grpc; \
  print(len(gsplat_pb2_grpc.GsplatServiceStub(\
  grpc.insecure_channel('127.0.0.1:50051')).ListJobs(\
  gsplat_pb2.ListJobsRequest(), timeout=5).jobs))"
```

If Windows Firewall blocks port `50051`, use a temporary SSH tunnel instead of
adding a persistent inbound firewall rule:

```
ssh -N -L 55051:127.0.0.1:50051 dm@192.168.11.194
```

Then submit a COLMAP reconstruction from the client machine:

```
python3 gsplat_server/client.py data/panorama \
  --server 127.0.0.1:55051 \
  --parameters my-run.proto.txt \
  --output windows-panorama-test.ply
```

`--parameters` is layered over `gsplat_server/config/gsplat_train_defaults.proto.txt`,
so a file only names what it changes:

```
strategy: STRATEGY_MCMC
cap_max: 900000
```

The input directory must contain `images/` and `sparse/`; `masks/` is optional.
The client uploads only those directories and follows the job log until the PLY
result is downloaded.

## Stop and restart

```
docker stop gsplat-server
docker start gsplat-server
```

Jobs and results remain under `D:\EasyGaussianSplatting\data\gsplat_server`.
