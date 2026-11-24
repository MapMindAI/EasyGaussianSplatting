# Cross-view fusion

`mapping/depth/cross_view_fusion.py` checks each written depth map against its
covisible neighbours' before `depth_points.ply` is merged and the depth loss
reads it. It adapts the fusion step of the World Composition stage of
[HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0)
([report](https://arxiv.org/abs/2604.14268), Sec. 7.1).

Without it, the merge (`point_cloud.py`) lifts every kept pixel and averages per
voxel, and nothing checks one frame's depth against another's: each frame adds
its own offset copy of a surface, so walls merge into thick or doubled bands.

## Stage

`generate_depths` fuses once every group is written, and the fused maps
replace the files in `depths/`. `depths/fusion.json` records each frame's
neighbour count and the fraction of checked pixels kept, and marks the depths
fused; writing any new depth deletes it, so a resumed run fuses again. Fusion
is not repeatable: a second pass averages already-fused depths. To fuse depths
written without it:

```bash
python3 -m mapping.depth.cross_view_fusion <model_dir> --min-confidence 0.7 [--mask-dir <masks>]
```

## Check

Each frame is checked against its 8 neighbours sharing the most sparse points
(at least 10). Each usable pixel (above the confidence floor and inside the
training mask) is projected into each neighbour and compared with the
neighbour's depth there:

- **Confirm:** it is within 5% (log depth) of the neighbour's depth.
- **Violate:** it is more than 5% in front of the neighbour's surface, in free
  space that neighbour saw through.

A pixel is zeroed when its violations are at least its confirmations and at
least one. Otherwise its depth becomes the average of its own and the
confirming estimates, each weighted by inverse square depth, since depth
error grows with depth. A pixel that no neighbour sees, or that every neighbour
sees behind its own surface, is kept as it is, because occlusion is no evidence
against it.

## Departures from HY-World

Measured on `VID_20260902_113042_00_006` with a per-group disparity fit in
place of the scale fit:

| Choice | Alternative tried | Why |
| --- | --- | --- |
| Reject only on free-space evidence | Require a confirmation | Requiring one rejected 12–17% of pixels, mostly at viewpoints no neighbour shares. Coverage within 30 cm of the sparse points fell from 85% to 81%. |
| Confirm within 5% | 3% | The pixels left unconfirmed missed by a median 5.9%. Moving to 5% changed nothing else. |
| Average the confirming depths | Only reject | Rejection alone leaves walls as thick as before (35 mm). |
| One fusion pass | Two | The second pass gained 0.1 point of held-out error for 16 s more. |

Statistical outlier removal is not added, because the fusion already lowers the
share of isolated points.
