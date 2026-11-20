"""The written depth rows merged into one coloured, grid-downsampled cloud.

Each image's rows are lifted through its COLMAP intrinsics and pose into the
COLMAP frame, coloured from the image, and averaged per voxel. The voxel edge is
a fraction of the scene scale -- the largest camera distance from the cameras'
mean, as gsplat measures it -- so one factor suits any capture size. gsplat
seeds its Gaussians from the result alongside the sparse SfM points.
"""

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

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


# Voxel indices are packed into one int64, 21 bits per axis: a voxel more than
# a million voxel edges (a thousand scene scales) from the origin is dropped.
_AXIS_BITS = 21
_AXIS_OFFSET = 1 << (_AXIS_BITS - 1)
# Rows of unmerged voxel sums held before folding them into the running total.
_FOLD_ROWS = 8_000_000
_THREADS = 4


def _reduce_by_key(keys, counts, *columns):
    """(unique keys, counts, each (N, 3) column) summed per key; `counts` of
    None counts the rows."""
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    counts = np.bincount(inverse, weights=counts, minlength=len(unique_keys))
    sums = [
        np.stack([np.bincount(inverse, weights=column[:, axis], minlength=len(unique_keys))
                  for axis in range(3)], axis=1)
        for column in columns
    ]
    return unique_keys, counts, *sums


def voxel_sums(points, colours, voxel_size):
    """(packed voxel keys, counts, summed points, summed colours) per occupied voxel."""
    indices = np.floor(points / voxel_size).astype(np.int64) + _AXIS_OFFSET
    inside = np.all((indices >= 0) & (indices < 1 << _AXIS_BITS), axis=1)
    if not inside.all():
        indices, points, colours = indices[inside], points[inside], colours[inside]
    keys = (indices[:, 0] << 2 * _AXIS_BITS) | (indices[:, 1] << _AXIS_BITS) | indices[:, 2]
    return _reduce_by_key(keys, None, points, colours)


def merge_voxel_sums(parts):
    """Several voxel_sums results as one."""
    return _reduce_by_key(*(np.concatenate(column) for column in zip(*parts)))


def voxel_means(part):
    """Each voxel's mean point and colour."""
    keys, counts, point_sums, colour_sums = part
    return point_sums / counts[:, None], colour_sums / counts[:, None]


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
    # Through a temporary file, so an interrupted write never leaves a partial cloud.
    temporary_path = path.with_name(f"{path.name}.partial")
    with open(temporary_path, "wb") as file:
        file.write((header + "\n").encode("ascii"))
        file.write(vertices.tobytes())
    os.replace(temporary_path, path)


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


def image_points(rows, image, intrinsics, camera_from_world):
    """One image's depth rows lifted into the COLMAP frame, with their RGB colours."""
    height, width = image.shape[:2]
    columns = np.rint(rows[:, 0] * (width - 1)).astype(np.int64)
    lines = np.rint(rows[:, 1] * (height - 1)).astype(np.int64)
    # COLMAP puts pixel centres at half-integer coordinates.
    pixels = np.column_stack([columns, lines]) + 0.5
    points = unproject(pixels, rows[:, 2], intrinsics, camera_from_world)
    return points, image[lines, columns, ::-1].astype(np.float64)


def depth_point_cloud(images, voxel_size):
    """Merged (points, uint8 RGB colours) over `images`, an iterable of
    (rows, image_bgr, intrinsics, camera_from_world)."""
    pending = deque()
    parts, held_rows, folded_rows = [], 0, 0

    def collect():
        # Folding once the parts outweigh the total re-sorts it only as often as it doubles.
        nonlocal parts, held_rows, folded_rows
        parts.append(pending.popleft().result())
        held_rows += len(parts[-1][0])
        if held_rows > max(_FOLD_ROWS, folded_rows):
            parts = [merge_voxel_sums(parts)]
            held_rows, folded_rows = 0, len(parts[0][0])

    # Not map_with_progress: pool.map would read every image up front. The
    # unprojection stays on this thread, as concurrent BLAS calls returned wrong points.
    with ThreadPoolExecutor(max_workers=_THREADS) as pool:
        for rows, image, intrinsics, camera_from_world in images:
            if len(rows):
                points = image_points(rows, image, intrinsics, camera_from_world)
                pending.append(pool.submit(voxel_sums, *points, voxel_size))
            if len(pending) >= 2 * _THREADS:
                collect()
        while pending:
            collect()
    if not parts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    points, colours = voxel_means(merge_voxel_sums(parts))
    return points, np.clip(np.rint(colours), 0, 255).astype(np.uint8)


def write_depth_point_cloud(model_dir, depth_dir, output_path, voxel_factor):
    """Merge every depth file under `depth_dir` for the COLMAP model in
    `model_dir` into `output_path`; returns the point count."""
    import cv2
    import pycolmap
    from tqdm import tqdm

    from mapping.depth.covisibility_groups import centre_and_rotation

    reconstruction = pycolmap.Reconstruction(model_dir / "sparse" / "0")
    poses = {}
    for image in reconstruction.images.values():
        camera_from_world = image.cam_from_world
        if callable(camera_from_world):
            camera_from_world = camera_from_world()
        poses[image.name] = np.asarray(camera_from_world.matrix(), dtype=np.float64)
    scale = scene_scale([centre_and_rotation(pose)[0] for pose in poses.values()])
    voxel_size = voxel_factor * scale
    print(f"Voxel edge {voxel_size:.4g}: {voxel_factor} of the scene scale {scale:.4g}", flush=True)

    with_depth = [
        image for image in reconstruction.images.values()
        if (depth_dir / f"{image.name}.npy").exists()
    ]

    def images():
        for image in tqdm(with_depth, desc="Point cloud", unit="image", dynamic_ncols=True):
            rows = np.load(depth_dir / f"{image.name}.npy")
            if len(rows) == 0:
                continue
            bgr = cv2.imread(str(model_dir / "images" / image.name))
            if bgr is None:
                continue
            yield (
                rows,
                bgr,
                np.asarray(reconstruction.cameras[image.camera_id].calibration_matrix()),
                poses[image.name],
            )

    points, colours = depth_point_cloud(images(), voxel_size)
    write_point_cloud(output_path, points, colours)
    return len(points)
