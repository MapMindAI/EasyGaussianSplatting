"""Render expected-hit-distance depth maps from a gsplat PLY."""
from pathlib import Path

import numpy as np

from .common import depth_path, valid_depth


def _camera_matrix(camera):
    if camera.model.name != "PINHOLE":
        raise ValueError(f"{camera}: TSDF rendering requires PINHOLE cameras")
    focal_x, focal_y, principal_x, principal_y = camera.params
    return np.array(
        [[focal_x, 0.0, principal_x], [0.0, focal_y, principal_y], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _world_to_camera(image):
    camera_from_world = image.cam_from_world
    if callable(camera_from_world):
        camera_from_world = camera_from_world()
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3] = np.asarray(camera_from_world.matrix(), dtype=np.float32)
    return matrix


def render_depths(model_path, reconstruction_path, depth_directory,
                  minimum_alpha=0.5, overwrite=False):
    """Render one depth map for every registered image in a cube-map model."""
    import pycolmap
    import torch
    from gsplat import rasterization
    from gsplat.exporter import load_ply_to_splats

    model_path = Path(model_path)
    depth_directory = Path(depth_directory)
    reconstruction = pycolmap.Reconstruction(reconstruction_path)
    splats = load_ply_to_splats(model_path)
    device = torch.device("cuda")
    splats = {name: value.to(device) for name, value in splats.items()}
    splats["quats"] = torch.nn.functional.normalize(splats["quats"], dim=-1)
    splats["scales"] = torch.exp(splats["scales"])
    splats["opacities"] = torch.sigmoid(splats["opacities"])

    written = 0
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        output_path = depth_path(depth_directory, image.name)
        if output_path.exists() and not overwrite:
            continue
        camera = reconstruction.cameras[image.camera_id]
        matrix = _camera_matrix(camera)
        render, alpha, _ = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=splats["scales"],
            opacities=splats["opacities"],
            colors=splats["sh0"],
            viewmats=torch.from_numpy(_world_to_camera(image))[None].to(device),
            Ks=torch.from_numpy(matrix)[None].to(device),
            width=camera.width,
            height=camera.height,
            render_mode="Ed",
        )
        depth = render[0, ..., 0].cpu().numpy()
        coverage = alpha[0, ..., 0].cpu().numpy()
        depth[~valid_depth(depth, coverage, minimum_alpha)] = 0.0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, depth.astype(np.float32))
        written += 1
    return written
