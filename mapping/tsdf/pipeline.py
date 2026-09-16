#!/usr/bin/env python3
"""Render gsplat depths and fuse a TSDF mesh for a cube-map reconstruction."""
import argparse
from pathlib import Path

from .common import latest_point_cloud
from .fuse import fuse_tsdf
from .render_depth import render_depths


def run_pipeline(workspace_path, model_path=None, voxel_length=0.02,
                 sdf_truncation=0.08, maximum_depth=20.0, minimum_alpha=0.5,
                 overwrite_depth=False):
    """Write `<workspace>/tsdf/mesh.ply` in the cube-map COLMAP frame."""
    workspace_path = Path(workspace_path)
    model_path = Path(model_path) if model_path else latest_point_cloud(
        workspace_path / "gsplat_output"
    )
    reconstruction_path = workspace_path / "sparse" / "0"
    images_path = workspace_path / "images"
    depth_directory = workspace_path / "tsdf" / "depths"
    mesh_path = workspace_path / "tsdf" / "mesh.ply"
    rendered = render_depths(
        model_path, reconstruction_path, depth_directory,
        minimum_alpha, overwrite_depth,
    )
    integrated = fuse_tsdf(
        reconstruction_path, images_path, depth_directory, mesh_path,
        voxel_length, sdf_truncation, maximum_depth,
    )
    return mesh_path, rendered, integrated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace_path")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--voxel-length", type=float, default=0.02)
    parser.add_argument("--sdf-truncation", type=float, default=0.08)
    parser.add_argument("--maximum-depth", type=float, default=20.0)
    parser.add_argument("--minimum-alpha", type=float, default=0.5)
    parser.add_argument("--overwrite-depth", action="store_true")
    arguments = parser.parse_args()
    mesh_path, rendered, integrated = run_pipeline(**vars(arguments))
    print(f"Wrote {mesh_path}: rendered {rendered}, fused {integrated} cube faces")


if __name__ == "__main__":
    main()
