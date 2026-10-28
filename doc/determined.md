## Run on determined

```
environment:
  add_capabilities: null
  drop_capabilities: null
  environment_variables:
    - PATH=/opt/miniconda3/envs/gsplat/bin:/opt/miniconda3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    - CONDA_PREFIX=/opt/miniconda3/envs/gsplat
    - CONDA_DEFAULT_ENV=gsplat
    - CUDA_HOME=/opt/miniconda3/envs/gsplat
  force_pull_image: true
  image:
    cpu: ghcr.io/mapmindai/gaussiansplatting:latest
    cuda: ghcr.io/mapmindai/gaussiansplatting:latest
    rocm: ghcr.io/mapmindai/gaussiansplatting:latest
  pod_spec:
```

```
cd /EasyGaussianSplatting
rm -f gsplat_server/proto/*_pb2*.py
source scripts/gsplat_env.sh
bash gsplat_server/proto/build.sh
```
