#!/usr/bin/env python3
"""Predict per-image depth for a COLMAP model with VGGT-Omega, run locally.

The same stage as mapping/depth/generate_depth.py -- covisibility grouping,
scale fitting to COLMAP, filtering, and the (x, y, depth) output -- with the
ONNX model at --model in place of the Triton DA3 one. It then merges every
written depth into <model_dir>/depth_points.ply, the cloud gsplat seeds its
Gaussians from (see mapping/depth/point_cloud.py).
"""

import argparse
from pathlib import Path

from mapping.depth.generate_depth import generate_depths, parse_depth_arguments
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
    # Two workers share the one session: one infers while the other reads and
    # filters. More would only queue on the GPU.
    parser.set_defaults(num_threads=2)
    args = parse_depth_arguments(parser)
    generate_depths(
        args, lambda: VggtOmegaDepthEstimator(args.model), GROUP_SIZE, MINIMUM_GROUP_SIZE
    )

    cloud_path = args.model_dir / POINT_CLOUD_NAME
    if cloud_path.exists() and not args.overwrite:
        print(f"{cloud_path} already exists; pass --overwrite to rebuild it")
        return
    count = write_depth_point_cloud(args.model_dir, args.depth_dir, cloud_path, args.voxel_factor)
    print(f"Merged depth point cloud ({count} points) written to {cloud_path}")


if __name__ == "__main__":
    main()
