# Depth alignment and outlier correction

Design for aligning each VGGT-Omega depth map to the COLMAP sparse model and
removing inconsistent depth before `depth_points.ply` is merged and the depth
loss reads it. It adapts the World Composition stage of
[HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0)
([report](https://arxiv.org/abs/2604.14268), Sec. 7.1), with COLMAP standing in
for its panorama point cloud as the scale reference.

## Today

`mapping/depth/generate_depth.py` fits **one scale per covisibility group** (up
to 12 frames) to the COLMAP points the group observes and to the camera
baselines (`scale_alignment.py`). The merge (`point_cloud.py`) lifts every kept
pixel and averages per voxel. Nothing corrects a single frame within a group,
and nothing checks one frame's depth against another's.

Measured on `data/VID_20260902_113042_00_006` (618 images, 18.6k sparse
points, 409 frames with at least 10 sparse samples at confidence >= 0.7):

| Measurement | p50 | p95 | max |
| --- | --- | --- | --- |
| Sparse samples per frame | 73 | 165 | 249 |
| Frame's residual log scale against COLMAP (p5 / p95 for the spread) | -0.005 | -0.031 / 0.018 | -0.084 |
| Median relative error, group scale | 2.1% | 3.9% | 8.9% |
| Median relative error, per-frame disparity affine | 1.8% | 2.9% | 5.2% |

The worst frames sit in runs (`left/000078`–`000081` at -6% to -8%), which
is a per-frame drift a group scale cannot absorb. Reprojecting one frame's depth
into a later frame of the same face:

| Pair | Median abs log difference | Pixels > 5% off (p50 / p95) | In front of the other view's surface |
| --- | --- | --- | --- |
| Next frame | 0.012 | 8% / 39% | 1% |
| Three frames on | 0.023 | 25% / 62% | 3% |

The disagreement does not depend on depth (17–26% from 0–16 m). It is the
duplicated, offset surfaces that the merged cloud shows as thickened walls and
floaters.

## HY-World 2.0 World Composition

1. **Reference.** It renders the panorama point cloud into each keyframe as a
   sparse guidance depth `D_g`.
2. **Mask.** The fit uses only pixels in `M = confident ∩ guided ∩ normals agree
   (< 90°) ∩ percentile agree ∩ not sky`, with depth edges removed from both
   maps. The percentile test drops pixels whose depth rank differs between the
   two maps by more than 20 percentiles.
3. **Fit.** Each frame gets its own linear fit in disparity, `1/D_g ≈ k/D + b`,
   by RANSAC, with `k > 0` and a bounded `b`. The paper reports that non-linear
   refinement gives no gain.
4. **Coefficient outliers.** It maps Q = 9 anchor depths `A_q` through each
   frame's fit and takes the frame's maximum relative deviation from the median
   across frames. Frames above the 90th percentile take `(k, b)` from the
   nearest inlier frame in the same video. A video with no inliers is
   discarded.
5. **Fusion.** It back-projects the aligned depths, voxel-downsamples them with
   the panorama cloud, and applies statistical outlier removal (10 neighbours,
   2σ).

## Design

A new stage, `mapping/depth/depth_alignment.py`, runs after generation and
before the merge. It reads `depths/` and the COLMAP model and writes
`aligned_depths/` in the same `.npz` format. Inference is never rerun, so
alignment can be tuned on existing depths. The merge and the depth loss read
`aligned_depths/`.

### 1. Reference samples per frame

The frame's COLMAP observations (`observed_depths`) serve as the sparse guidance
in place of a rendered cloud. They are sampled at the map pixel that holds each
point (`sample_at`). Samples are kept where:

- confidence meets the floor, the map is non-zero (discontinuities are already
  zeroed), and the training mask allows the pixel;
- the point has a track of 3 or more images and depth <= `MAX_REFERENCE_DEPTH`;
- the percentile test passes: the sample's depth rank among the frame's samples
  differs by at most 20 percentiles between the two depths.

HY-World's normal test is dropped because sparse points carry no normals.

### 2. Per-frame fit in disparity

Fit `1/D_colmap = k/D + b` to each frame's samples with RANSAC:

- **Minimal sample:** 2 points.
- **Inliers:** relative depth error <= 5%.
- **Refinement:** least squares on the inliers.

The fit must satisfy `k > 0` and `k/D + b > 0` over the frame's kept depth
range. The group fit is `k = 1, b = 0`, which is also the starting point. A
frame falls back to scale only (`b = 0`) when:

- it has fewer than 30 inliers, or
- its sample disparities span less than a factor of 1.5, where the shift is
  unobservable.

A frame with no usable samples keeps its group scale. This covers up/down faces
with depth and frames with sparse coverage too thin to fit.

### 3. Coefficient outliers

As in HY-World, nine anchor depths are mapped through each frame's fit. The
anchors are the 5th–95th percentiles of all sample depths. The comparison
differs in two ways:

- **Neighbours, not the scene.** The reference is the median over the frame's
  neighbours: the same face within ±5 frames and the other faces of the same
  panorama frame. A scene-wide median assumes every frame shares one model bias,
  which fails over a long route.
- **Absolute floor.** A frame is an outlier when its maximum relative anchor
  deviation is above the 90th percentile **and** above 5%. A pure percentile
  always rejects 10% of frames, even when every fit is good.

An outlier takes `(k, b)` from the nearest inlier frame on the same face.

### 4. Cross-view consistency

Sparse points cover about 73 pixels per frame, so they cannot vet the dense map.
Each kept pixel is therefore checked against the aligned depths of its
neighbours from step 3, as in multi-view-stereo depth fusion:

- It is **confirmed** by a neighbour whose depth at the projected pixel is
  within 3% (log difference) of the projected depth.
- It **violates** a neighbour when it projects more than 5% in front of that
  neighbour's surface: free space that neighbour saw through.
- It is kept when it has at least one confirmation and more confirmations than
  violations.

A frame without neighbours keeps every pixel. Pixels that are behind a
neighbour's surface count as neither, since they are occluded there.

Rejected pixels are written as 0, so the merge and the depth loss both skip
them.

### 5. Fusion

The existing voxel-mean merge is unchanged. Statistical outlier removal is added
only if the merged cloud still shows isolated floaters after step 4. Step 4
removes floaters by evidence, while statistical removal also erodes thin
structures.

## Outputs

- `aligned_depths/<image_name>.npz`: the aligned depth, inconsistent pixels set
  to 0, and confidence and `scale` as in the input.
- `aligned_depths/alignment.json`: per frame `k`, `b`, the sample and inlier
  counts, the median relative error before and after, and the status
  (`fitted`, `scale_only`, `replaced`, or `group_scale`). It also records the
  pixel fraction step 4 kept.

## Validation on `VID_20260902_113042_00_006`

The alignment must improve on the tables above:

- Per-frame median relative error against COLMAP: p95 below 3%, max below 5%,
  checked on held-out sparse points (every fifth point excluded from the fit).
- Next frame and three-frames-on agreement: fewer pixels more than 5% off, and
  in front of the other view's surface under 1%.
- The merged cloud viewed against today's: thinner walls and fewer floaters,
  while keeping most points.

## Implementation order

1. Steps 1–3 with `alignment.json`, plus a test that recovers a synthetic `(k, b)`
   with outliers. Report validation numbers.
2. Step 4, with a two-view test where a floater is rejected and a surface
   seen by both is kept. Report the cross-view numbers and the point count.
3. Wire `gsplat_depth.sh` and the depth loss to `aligned_depths/`, and update
   `doc/gsplat_server.md`.
