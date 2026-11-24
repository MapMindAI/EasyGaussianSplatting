#!/usr/bin/env python3
"""Predict per-image depth for a COLMAP model with VGGT-Omega, run locally.

The same stage as mapping/depth/generate_depth.py -- covisibility grouping,
scale fitting to COLMAP, and the depth and confidence output -- with the
ONNX model at --model in place of the Triton DA3 one. Once new depths are
written, it fuses each with its neighbours' in place (see
mapping/depth/depth_alignment.py). It then merges every depth, filtered by
--min-confidence, the depth range, the masks in --mask-dir, and colour edges,
into <model_dir>/depth_points.ply, the cloud gsplat seeds its Gaussians from
(see mapping/depth/point_cloud.py), rebuilding it whenever the depths were
fused or the settings recorded beside it in depth_points.json changed.
"""

import argparse
import json
import time
from pathlib import Path

from mapping.depth.depth_alignment import fuse_depths
from mapping.depth.generate_depth import depth_filter, generate_depths, parse_depth_arguments
from mapping.depth.point_cloud import POINT_CLOUD_NAME, write_depth_point_cloud
from mapping.vggt_omega.depth_estimator import DEFAULT_MODEL_PATH, VggtOmegaDepthEstimator

GROUP_SIZE = 12
# Two views already give the model a baseline to triangulate.
MINIMUM_GROUP_SIZE = 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH, help="VGGT-Omega ONNX model"
    )
    parser.add_argument(
        "--voxel-factor",
        type=float,
        default=0.001,
        help="point cloud voxel edge as a fraction of the scene scale",
    )
    # Two workers share the one session, which runs one group at a time: one
    # infers while the other reads and filters. More would only queue on it.
    parser.set_defaults(num_threads=2)
    args = parse_depth_arguments(parser)
    generate_depths(
        args, lambda: VggtOmegaDepthEstimator(args.model), GROUP_SIZE, MINIMUM_GROUP_SIZE
    )
    pixel_filter = depth_filter(args)
    fused = fuse_depths(args.model_dir, args.depth_dir, pixel_filter, args.mask_dir)

    cloud_path = args.model_dir / POINT_CLOUD_NAME
    # What the cloud was built from, so a changed filter or voxel rebuilds it.
    settings_path = cloud_path.with_suffix(".json")
    settings = {
        "voxel_factor": args.voxel_factor,
        "depth_filter": pixel_filter._asdict(),
        "mask_dir": None if args.mask_dir is None else str(args.mask_dir),
    }
    if (
        not fused
        and cloud_path.exists()
        and settings_path.exists()
        and json.loads(settings_path.read_text()) == settings
    ):
        print(f"{cloud_path} is up to date; keeping it")
        return
    started = time.monotonic()
    count = write_depth_point_cloud(
        args.model_dir, args.depth_dir, cloud_path, args.voxel_factor, args.mask_dir, pixel_filter
    )
    settings_path.write_text(json.dumps(settings, indent=2))
    print(
        f"Merged depth point cloud ({count} points) written to {cloud_path} "
        f"({time.monotonic() - started:.1f} s)"
    )


if __name__ == "__main__":
    main()
