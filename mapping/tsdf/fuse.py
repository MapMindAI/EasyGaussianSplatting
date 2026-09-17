"""Fuse rendered cube-face depths into a triangle mesh."""
import argparse
import json
from pathlib import Path

import numpy as np

from .common import camera_manifest_path, depth_path, progress_milestones


def fuse_tsdf(images_path, depth_directory, mesh_path,
              voxel_length=0.02, sdf_truncation=0.08, maximum_depth=20.0):
    """Fuse registered cube-face depth maps and write their mesh in COLMAP space."""
    import open3d as open3d
    images_path = Path(images_path)
    depth_directory = Path(depth_directory)
    mesh_path = Path(mesh_path)
    volume = open3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_truncation,
        color_type=open3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    integrated = 0
    cameras = json.loads(camera_manifest_path(depth_directory).read_text())
    milestones = progress_milestones(len(cameras))
    milestone_index = 0
    for completed, camera in enumerate(cameras, start=1):
        depth_file = depth_path(depth_directory, camera["name"])
        if depth_file.exists():
            color_file = images_path / camera["name"]
            color = open3d.io.read_image(str(color_file))
            if color.is_empty():
                raise FileNotFoundError(f"Cannot read {color_file}")
            depth = open3d.geometry.Image(np.load(depth_file).astype(np.float32))
            rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1.0, depth_trunc=maximum_depth,
                convert_rgb_to_intensity=False,
            )
            focal_x, focal_y, principal_x, principal_y = camera["params"]
            intrinsic = open3d.camera.PinholeCameraIntrinsic(
                camera["width"],
                camera["height"],
                focal_x,
                focal_y,
                principal_x,
                principal_y,
            )
            volume.integrate(rgbd, intrinsic, np.asarray(camera["world_to_camera"]))
            integrated += 1
        while (milestone_index < len(milestones)
               and completed >= milestones[milestone_index][0]):
            _, percent = milestones[milestone_index]
            print(f"Fused TSDF: {percent}% ({completed}/{len(cameras)} cube faces)")
            milestone_index += 1
    if not integrated:
        raise RuntimeError(f"No rendered depths found under {depth_directory}")
    mesh = volume.extract_triangle_mesh()
    if not mesh.has_triangles():
        raise RuntimeError("TSDF extraction produced no triangles")
    mesh.compute_vertex_normals()
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    if not open3d.io.write_triangle_mesh(str(mesh_path), mesh):
        raise OSError(f"Could not write {mesh_path}")
    return integrated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace_path", type=Path)
    parser.add_argument("--voxel-length", type=float, default=0.02)
    parser.add_argument("--sdf-truncation", type=float, default=0.08)
    parser.add_argument("--maximum-depth", type=float, default=20.0)
    arguments = parser.parse_args()
    workspace_path = arguments.workspace_path
    integrated = fuse_tsdf(
        workspace_path / "images",
        workspace_path / "tsdf" / "depths",
        workspace_path / "tsdf" / "mesh.ply",
        arguments.voxel_length,
        arguments.sdf_truncation,
        arguments.maximum_depth,
    )
    print(f"Fused {integrated} cube faces into {workspace_path / 'tsdf' / 'mesh.ply'}")


if __name__ == "__main__":
    main()
