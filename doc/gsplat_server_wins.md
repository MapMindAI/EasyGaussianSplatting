# Gaussian Splatting server on Windows Docker Desktop

This guide runs the x86 gsplat image in Docker Desktop on a Windows PC with an
NVIDIA GPU. The project is stored on `D:` and the server exposes gRPC on port
`50051`. [doc/gsplat_server.md](gsplat_server.md) covers the service itself —
the client, the training parameters, and the gRPC API; only the
Windows-specific parts are here.

## Prerequisites

- Windows 11 with Docker Desktop using the `desktop-linux` context.
- NVIDIA driver with Docker GPU support.
- SSH access to the Windows host, for example `dm@192.168.11.194`.
- A checkout containing `gsplat_server/`, `mapping/`, and the generated proto
  bindings.

The Jetson image is ARM64 and must not be used on this PC. Use the x86 image:

```
ghcr.io/mapmindai/gaussiansplatting:latest
```

## Copy the project to `D:`

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

Generate the ignored Python protobuf bindings before creating the archive:

```
bash gsplat_server/proto/build.sh
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

## Start the server

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
