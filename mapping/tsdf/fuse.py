"""Fuse rendered cube-face depths into a triangle mesh."""
from pathlib import Path

import numpy as np

from .common import depth_path
from .render_depth import _camera_matrix, _world_to_camera


def fuse_tsdf(reconstruction_path, images_path, depth_directory, mesh_path,
              voxel_length=0.02, sdf_truncation=0.08, maximum_depth=20.0):
    """Fuse registered cube-face depth maps and write their mesh in COLMAP space."""
    import open3d as open3d
    import pycolmap

    reconstruction = pycolmap.Reconstruction(reconstruction_path)
    images_path = Path(images_path)
    depth_directory = Path(depth_directory)
    mesh_path = Path(mesh_path)
    volume = open3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_truncation,
        color_type=open3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    integrated = 0
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        depth_file = depth_path(depth_directory, image.name)
        if not depth_file.exists():
            continue
        color_file = images_path / image.name
        color = open3d.io.read_image(str(color_file))
        if color.is_empty():
            raise FileNotFoundError(f"Cannot read {color_file}")
        depth = open3d.geometry.Image(np.load(depth_file).astype(np.float32))
        rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1.0, depth_trunc=maximum_depth,
            convert_rgb_to_intensity=False,
        )
        camera = reconstruction.cameras[image.camera_id]
        focal_x, focal_y, principal_x, principal_y = camera.params
        intrinsic = open3d.camera.PinholeCameraIntrinsic(
            camera.width, camera.height, focal_x, focal_y, principal_x, principal_y
        )
        volume.integrate(rgbd, intrinsic, _world_to_camera(image))
        integrated += 1
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
