# Mapping a panorama capture

`mapping/mapping_pipeline.py` turns a stitched equirectangular video into a
cube-map COLMAP reconstruction that gsplat trains on directly. It runs
learned features against a Triton inference server and solves the scene with
global SfM (GLOMAP):

1. **Cube-map rig database** — each sampled panorama frame is reprojected into
   90-degree-FOV pinhole cube faces and registered as one COLMAP frame of a
   single rig. The front face is the rig's reference sensor; every other face
   is mounted on it by a pure rotation, since all faces share the panorama's
   optical centre. Global SfM then solves one pose per panorama instead of one
   per face, with the faces' relative orientations held at their exact values.
2. **Feature extraction** — SuperPoint keypoints and descriptors into the
   database's own tables, one SALAD place-recognition descriptor per image into
   a `global_features.npz` sidecar, which COLMAP's schema has nowhere to put.
   COLMAP types its descriptors table `uint8`, so the float descriptors are
   stored byte-reinterpreted under a learned-extractor tag. That is lossless
   and the mapper never reads the tag, but COLMAP's own matchers dispatch on
   it: this database is for this pipeline, not for `colmap matcher`. Each image
   keeps its 512 strongest detections, LightGlue's exported input size.
3. **Matching** — LightGlue over two sets of candidate pairs: each image
   against the next `--num-sequential` images of the same cube face, and
   against its `--num-retrieval` nearest images by SALAD descriptor. Retrieval
   skips same-face images within `--num-retrieval-excluded` frames, which
   sequential matching already covers, so what it contributes is revisits of a
   place from elsewhere in the capture. Faces are matched only against
   themselves: 90-degree faces of one panorama do not overlap, and the rig, not
   a match, is what ties them together.
4. **Global mapping** — `pycolmap.global_mapping` into `sparse/0/`. The cube
   faces' intrinsics and rig mounting come from the reprojection rather than a
   calibration, so bundle adjustment holds both fixed.

A re-run picks up where the last one stopped: the database is kept once it
holds images, extraction is skipped once the sidecar is written, matching skips
the pairs the database already holds a two-view geometry for, and global mapping
is skipped once `sparse/0/` exists. Stages 2 and 3 write into the same
`database.db` that stage 1 builds, so redoing stage 1 means deleting the
workspace.

## Serving the models

The pipeline needs `superpoint`, `lightglue`, and `salad` served over gRPC by
the Triton server in [`third_party/EasyTensorRT`](https://github.com/MapMindAI/EasyTensorRT),
which ships both the model repository and the clients this repo wraps:

```
git submodule update --init third_party/EasyTensorRT
cd third_party/EasyTensorRT
docker run --gpus all --rm --name tritonserver -p 8011:8001 \
  -v "$(pwd)":/repo ghcr.io/mapmindai/tritonserver_amd64:latest \
  tritonserver --model-repository=/repo/model_repository
```

The submodule carries the model weights, so expect the checkout to be around
800 MB.

The submodule's own `run_server_onnx.sh` publishes Triton's standard gRPC port
8001, which is often already taken; the command above is that script with the
host side moved to 8011, which is what this repo defaults to. `run_server_trt.sh`
serves TensorRT plans instead, which it builds for your GPU on the first run
(slow the first time, faster afterwards).

The host-side scripts default `TRITON_URL` to `host.docker.internal:8011` and
map that name to the Docker host, so a server started as above needs no
configuration. Point `TRITON_URL` at `host:port` to use a server elsewhere.

## Running it

```
docker run -it --rm --gpus all -v $(pwd):/workspace -w /workspace \
  --add-host host.docker.internal:host-gateway \
  ghcr.io/mapmindai/gaussiansplatting:latest \
  python3 -m mapping.mapping_pipeline \
    --video_path data/VID_20260422_153814_00_004_pano.mp4 \
    --workspace_path data/VID_20260422_153814_00_004_reconstruction \
    --triton-url host.docker.internal:8011
```

`--video_path` is the stitched video and `--workspace_path` the directory to
reconstruct into; both are required.
`--frame-rate` (frames/sec sampled from the video) defaults to 2, `--face-size`
(cube-face width/height in pixels) to a quarter of the video width — which
keeps the panorama's angular resolution — and `--faces` to
`front,right,back,left,up` — the nadir usually shows whoever is carrying the
rig, so pass all six explicitly to include it. The face list must contain
`front`, the rig's reference sensor. `--num-sequential`, `--num-retrieval`, and
`--num-retrieval-excluded` tune pair selection; `--help` lists everything.

The workspace ends up holding `images/<face>/`, `database.db`,
`global_features.npz`, and `sparse/0/`, which is what gsplat trains on.

`scripts/run_pipeline.sh` runs this stage between stitching and training as part
of the whole pipeline, passing `TRITON_URL` through to it.
