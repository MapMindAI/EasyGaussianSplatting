"""The written depth rows merged into one coloured, grid-downsampled cloud.

Each image's rows are lifted through its COLMAP intrinsics and pose into the
COLMAP frame, coloured from the image, and averaged per voxel. The voxel edge is
a fraction of the scene scale -- the largest camera distance from the cameras'
mean, as gsplat measures it -- so one factor suits any capture size. gsplat
seeds its Gaussians from the result in place of the sparse SfM points.
"""

import numpy as np

POINT_CLOUD_NAME = "depth_points.ply"
_VERTEX = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])


def unproject(pixels, depths, intrinsics, camera_from_world):
    """World points at `depths` along the rays through (P, 2) `pixels`."""
    rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(intrinsics).T
    camera_points = rays * np.asarray(depths)[:, None]
    rotation, translation = camera_from_world[:, :3], camera_from_world[:, 3]
    return (camera_points - translation) @ rotation


def scene_scale(camera_centers):
    camera_centers = np.asarray(camera_centers, dtype=np.float64)
    return float(np.linalg.norm(camera_centers - camera_centers.mean(axis=0), axis=1).max())


def voxel_sums(points, colours, voxel_size):
    """(voxel keys, summed points, summed colours, counts) per occupied voxel."""
    keys, inverse, counts = np.unique(
        np.floor(points / voxel_size).astype(np.int64),
        axis=0, return_inverse=True, return_counts=True,
    )
    inverse = inverse.reshape(-1)
    point_sums = np.zeros((len(keys), 3))
    colour_sums = np.zeros((len(keys), 3))
    np.add.at(point_sums, inverse, points)
    np.add.at(colour_sums, inverse, colours)
    return keys, point_sums, colour_sums, counts


def merge_voxel_sums(parts):
    """Each voxel's mean point and colour over several voxel_sums results."""
    keys = np.concatenate([part[0] for part in parts])
    merged_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    sums = []
    for column in (1, 2):
        total = np.zeros((len(merged_keys), 3))
        np.add.at(total, inverse, np.concatenate([part[column] for part in parts]))
        sums.append(total)
    counts = np.bincount(inverse, weights=np.concatenate([part[3] for part in parts]))
    return sums[0] / counts[:, None], sums[1] / counts[:, None]


def write_point_cloud(path, points, colours_rgb):
    vertices = np.empty(len(points), dtype=_VERTEX)
    vertices["x"], vertices["y"], vertices["z"] = np.asarray(points).T
    vertices["red"], vertices["green"], vertices["blue"] = np.asarray(colours_rgb).T
    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(vertices)}",
        *(f"property float {axis}" for axis in "xyz"),
        *(f"property uchar {channel}" for channel in ("red", "green", "blue")),
        "end_header",
    ])
    with open(path, "wb") as file:
        file.write((header + "\n").encode("ascii"))
        file.write(vertices.tobytes())


def read_point_cloud(path):
    """(points, uint8 RGB colours) from a write_point_cloud file."""
    with open(path, "rb") as file:
        header = file.read().split(b"end_header\n", 1)
    count = int(next(
        line.split()[-1] for line in header[0].decode("ascii").splitlines()
        if line.startswith("element vertex")
    ))
    vertices = np.frombuffer(header[1], dtype=_VERTEX, count=count)
    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]])
    return points, np.column_stack([vertices["red"], vertices["green"], vertices["blue"]])


def depth_point_cloud(images, voxel_size):
    """Merged (points, uint8 RGB colours) over `images`, an iterable of
    (rows, image_bgr, intrinsics, camera_from_world)."""
    parts = []
    for rows, image, intrinsics, camera_from_world in images:
        if len(rows) == 0:
            continue
        height, width = image.shape[:2]
        columns = np.rint(rows[:, 0] * (width - 1)).astype(np.int64)
        lines = np.rint(rows[:, 1] * (height - 1)).astype(np.int64)
        # COLMAP puts pixel centres at half-integer coordinates.
        pixels = np.column_stack([columns, lines]) + 0.5
        points = unproject(pixels, rows[:, 2], intrinsics, camera_from_world)
        parts.append(voxel_sums(points, image[lines, columns, ::-1].astype(np.float64), voxel_size))
    if not parts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    points, colours = merge_voxel_sums(parts)
    return points, np.clip(np.rint(colours), 0, 255).astype(np.uint8)


def write_depth_point_cloud(model_dir, depth_dir, output_path, voxel_factor):
    """Merge every depth file under `depth_dir` for the COLMAP model in
    `model_dir` into `output_path`; returns the point count."""
    import cv2
    import pycolmap

    from mapping.depth.covisibility_groups import centre_and_rotation

    reconstruction = pycolmap.Reconstruction(model_dir / "sparse" / "0")
    poses = {}
    for image in reconstruction.images.values():
        camera_from_world = image.cam_from_world
        if callable(camera_from_world):
            camera_from_world = camera_from_world()
        poses[image.name] = np.asarray(camera_from_world.matrix(), dtype=np.float64)
    voxel_size = voxel_factor * scene_scale(
        [centre_and_rotation(pose)[0] for pose in poses.values()]
    )

    def images():
        for image in reconstruction.images.values():
            depth_path = depth_dir / f"{image.name}.npy"
            bgr = cv2.imread(str(model_dir / "images" / image.name))
            if not depth_path.exists() or bgr is None:
                continue
            yield (
                np.load(depth_path),
                bgr,
                np.asarray(reconstruction.cameras[image.camera_id].calibration_matrix()),
                poses[image.name],
            )

    points, colours = depth_point_cloud(images(), voxel_size)
    write_point_cloud(output_path, points, colours)
    return len(points)
