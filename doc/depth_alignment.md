# Depth alignment and cross-view fusion

`mapping/depth/depth_alignment.py` aligns each VGGT-Omega depth map to the
COLMAP sparse model, then checks it against its neighbours' depths before
`depth_points.ply` is merged and the depth loss reads it. It adapts the World
Composition stage of [HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0)
([report](https://arxiv.org/abs/2604.14268), Sec. 7.1), with COLMAP standing in
for its panorama point cloud as the scale reference.

## Before

`mapping/depth/generate_depth.py` fits **one scale per covisibility group** (up
to 12 frames) to the COLMAP points the group observes and to the camera
baselines (`scale_alignment.py`). The merge (`point_cloud.py`) lifts every kept
pixel and averages per voxel. Nothing corrects one frame within its group, and
nothing checks one frame's depth against another's. Frames drift in runs: on
`VID_20260902_113042_00_006`, `left/000078`–`000081` sit 6–8% off COLMAP. Walls
merge into bands about 0.3 m thick, because each frame contributes its own
offset copy of the surface.

## HY-World 2.0 World Composition

1. **Reference.** The panorama point cloud is rendered into each keyframe as
   a sparse guidance depth `D_g`.
2. **Mask.** The fit uses only pixels in `confident ∩ guided ∩ normals agree ∩
   percentile agree ∩ not sky`, with depth edges removed.
3. **Fit.** Each frame gets `1/D_g ≈ k/D + b`, solved by RANSAC.
4. **Coefficient outliers.** Nine anchor depths are mapped through each fit.
   A frame above the 90th percentile of deviation from the median takes the
   nearest inlier frame's `(k, b)`.
5. **Fusion.** The aligned depths are back-projected, voxel-downsampled, and
   statistical outlier removal is applied.

## Stage

It runs inside `mapping/vggt_omega/generate_depth.py`, after prediction and
before the merge. It reads `depths/` and writes `aligned_depths/` in the same
`.npz` format, plus `aligned_depths/alignment.json`. That file records, per
frame: `scale` (k), `shift` (b), `status`, sample and inlier counts, the
neighbour count, and the fraction of checked pixels kept. The merge and the
depth loss read `aligned_depths/`. It can be rerun without inference:

```bash
python3 -m mapping.depth.depth_alignment <model_dir> --min-confidence 0.7 [--mask-dir <masks>]
```

### 1. Reference samples

The reference is the COLMAP points each frame observes, each read at the map
pixel that holds it. A sample is kept when it passes the confidence floor and
the training mask, has a track of at least 3 images and a depth of at most
20 m, and its depth rank among the frame's samples differs by at most 20
percentiles between the two depths. HY-World's normal test is dropped because
sparse points have no normals.

### 2. Per-frame fit in disparity

Each frame gets `1/D_colmap = k/D + b`, found by RANSAC:

- **Candidates:** 200 two-point fits.
- **Inliers:** relative depth error <= 5%.
- **Refinement:** twice, by least squares on relative error, over the inliers.

The fit needs at least 30 inliers. Its samples' disparities must span a
factor of 1.5, and it must keep the frame's farthest usable depth in front of
the camera. Otherwise the frame takes a scale-only fit (at least 10 inliers),
or failing that keeps its group scale.

### 3. Cross-view fusion

Each frame is checked against its 8 neighbours sharing the most sparse points
(at least 10). Each usable pixel is projected into each neighbour and compared
with the neighbour's depth there:

- **Confirm:** it is within 5% (log depth) of the neighbour's depth.
- **Violate:** it is more than 5% in front of the neighbour's surface, in free
  space that neighbour saw through.

A pixel is zeroed when its violations are at least its confirmations and at
least one. Otherwise its depth becomes the average of its own and the
confirming estimates, each weighted by inverse square depth, since depth
error grows with depth. A pixel that no neighbour sees, or that every neighbour
sees behind its own surface, is kept as it is, because occlusion is no evidence
against it.

### Departures from HY-World, measured

| Choice | Alternative tried | Why |
| --- | --- | --- |
| No coefficient-outlier step | HY-World's step, judged against covisible neighbours, with a 5% floor | It replaced 21 frames, including the drifting run `left/000078`–`000081`. There the frames' own fits scored 1.4–1.7% on held-out points against 6.5–7.4% for the replacements, because their neighbours came from the same bad group. With COLMAP as the reference, RANSAC alone is reliable. |
| Reject only on free-space evidence | Require a confirmation | Requiring one rejected 12–17% of pixels, mostly at viewpoints no neighbour shares. Coverage within 30 cm of the sparse points fell from 85% to 81%. |
| Confirm within 5% | 3% | The pixels left unconfirmed missed by a median 5.9%. Moving to 5% changed nothing else. |
| Average the confirming depths | Only reject | Rejection alone leaves walls as thick as before (35 mm). |
| One fusion pass | Two | The second pass gained 0.1 point of held-out error for 16 s more. |

Statistical outlier removal is not added, because the fusion already lowers the
share of isolated points.

## Results on `VID_20260902_113042_00_006`

618 images and 412 frames with depth, confidence >= 0.7, and a voxel edge of
0.001 of the scene scale (17 mm). Alignment takes 26 s on top of the 7 s
merge.

For the held-out rows, every fifth sparse point was held out of the fit. "In
front" is the share of a frame's pixels that land more than 5% in front of a
later frame's surface. Thickness is the RMS distance of the 256 nearest points
from their plane. Isolated points are those beyond 2σ of the mean 16-neighbour
distance.

| Measurement | Before | Aligned and fused |
| --- | --- | --- |
| Held-out sparse error, frame median (p50 / p95 / max) | 2.1% / 4.2% / 9.2% | 1.5% / 3.2% / 5.6% |
| In front, next frame (p50 / p95) | 0.8% / 18% | 0.1% / 3.8% |
| In front, three frames on (p50 / p95) | 2.0% / 27% | 0.3% / 7.5% |
| Wall thickness (p50 / p90) | 35 / 55 mm | 30 / 50 mm |
| Isolated points | 4.4% | 4.1% |
| Points | 28.6 M | 23.4 M |
| Sparse points with cloud within 10 cm / 30 cm / 1 m | 72% / 85% / 92% | 70% / 84% / 92% |

Status per frame: 360 fitted, 45 scale-only, 7 on their group scale. The
cross-view check kept a median 98% of checked pixels.

Fusion moves surfaces, so slightly fewer sparse points have a cloud point
within 3 cm (47% before, 41% after). Coverage at 30 cm and beyond is unchanged.
Horizontal slices show the near wall's band at about half its former
thickness, and the doubled curves of the far wall merged into one line. Walls
more than 10 m away stay layered: they are seen only from far, where few
neighbours confirm them.
