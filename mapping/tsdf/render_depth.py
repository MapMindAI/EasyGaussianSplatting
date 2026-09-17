"""Render expected-hit-distance depth maps from a gsplat PLY."""
import json
from pathlib import Path

import numpy as np

from .common import camera_manifest_path, depth_path, latest_point_cloud, valid_depth


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


def _read_ply(path):
    """Read gsplat's binary float PLY without adding a renderer dependency."""
    terminator = b"end_header\n"
    with Path(path).open("rb") as handle:
        header = handle.read(8192)
        while terminator not in header:
            chunk = handle.read(8192)
            if not chunk:
                raise ValueError(f"{path}: truncated PLY header")
            header += chunk
    header_end = header.index(terminator) + len(terminator)
    lines = header[:header_end].decode("ascii", "replace").splitlines()
    names = [line.split()[-1] for line in lines if line.startswith("property")]
    count = next(
        int(line.split()[-1]) for line in lines if line.startswith("element vertex")
    )
    return names, np.memmap(
        path, dtype="<f4", mode="r", offset=header_end, shape=(count, len(names))
    )


def _splat_parameters(path, torch):
    names, values = _read_ply(path)

    def columns(property_names):
        return np.array(
            values[:, [names.index(name) for name in property_names]],
            dtype=np.float32,
            copy=True,
        )

    return {
        "means": torch.from_numpy(columns(("x", "y", "z"))),
        "quats": torch.from_numpy(
            columns(tuple(f"rot_{index}" for index in range(4)))
        ),
        "scales": torch.from_numpy(
            columns(tuple(f"scale_{index}" for index in range(3)))
        ),
        "opacities": torch.from_numpy(
            np.array(values[:, names.index("opacity")], dtype=np.float32, copy=True)
        ),
    }


def _camera_record(image, camera):
    return {
        "name": image.name,
        "width": camera.width,
        "height": camera.height,
        "params": [float(value) for value in camera.params],
        "world_to_camera": _world_to_camera(image).tolist(),
    }


def render_depths(model_path, reconstruction_path, depth_directory,
                  minimum_alpha=0.5, overwrite=False):
    """Render one depth map for every registered image in a cube-map model."""
    import pycolmap
    import torch
    from gsplat import rasterization

    model_path = Path(model_path)
    depth_directory = Path(depth_directory)
    reconstruction = pycolmap.Reconstruction(reconstruction_path)
    splats = _splat_parameters(model_path, torch)
    device = torch.device("cuda")
    splats = {name: value.to(device) for name, value in splats.items()}
    splats["quats"] = torch.nn.functional.normalize(splats["quats"], dim=-1)
    splats["scales"] = torch.exp(splats["scales"])
    splats["opacities"] = torch.sigmoid(splats["opacities"])

    cameras = []
    written = 0
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        camera = reconstruction.cameras[image.camera_id]
        cameras.append(_camera_record(image, camera))
        output_path = depth_path(depth_directory, image.name)
        if output_path.exists() and not overwrite:
            continue
        matrix = _camera_matrix(camera)
        render, alpha, _ = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=splats["scales"],
            opacities=splats["opacities"],
            colors=torch.zeros((len(splats["means"]), 1, 3), device=device),
            viewmats=torch.from_numpy(_world_to_camera(image))[None].to(device),
            Ks=torch.from_numpy(matrix)[None].to(device),
            width=camera.width,
            height=camera.height,
            render_mode="Ed",
            packed=False,
            with_eval3d=True,
        )
        depth = render[0, ..., 0].cpu().numpy()
        coverage = alpha[0, ..., 0].cpu().numpy()
        depth[~valid_depth(depth, coverage, minimum_alpha)] = 0.0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, depth.astype(np.float32))
        written += 1
    depth_directory.mkdir(parents=True, exist_ok=True)
    camera_manifest_path(depth_directory).write_text(json.dumps(cameras))
    return written


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace_path", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--minimum-alpha", type=float, default=0.5)
    parser.add_argument("--overwrite-depth", action="store_true")
    arguments = parser.parse_args()
    model_path = arguments.model_path or latest_point_cloud(
        arguments.workspace_path / "gsplat_output"
    )
    written = render_depths(
        model_path,
        arguments.workspace_path / "sparse" / "0",
        arguments.workspace_path / "tsdf" / "depths",
        arguments.minimum_alpha,
        arguments.overwrite_depth,
    )
    print(f"Rendered {written} cube-face depth maps")


if __name__ == "__main__":
    main()
