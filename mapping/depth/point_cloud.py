"""The written depth rows merged into one coloured, grid-downsampled cloud.

Each image's rows are lifted through its COLMAP intrinsics and pose into the
COLMAP frame, coloured from the image, and averaged per voxel. The voxel edge is
a fraction of the scene scale -- the largest camera distance from the cameras'
mean, as gsplat measures it -- so one factor suits any capture size. gsplat
seeds its Gaussians from the result alongside the sparse SfM points.
"""

import os

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
_AXIS_MASK = (1 << _AXIS_BITS) - 1
# New voxels buffered before folding them into the running total.
_FOLD_ROWS = 8_000_000
# Voxels converted to points and written per slice.
_WRITE_ROWS = 4_000_000
# Images read ahead in parallel; reading the depth files dominates on network storage.
_LOAD_THREADS = 8


def _reduce_by_key(keys, values):
    """(sorted unique keys, the (N, C) `values` rows summed per key)."""
    import torch

    unique_keys, inverse = torch.unique(keys, return_inverse=True)
    sums = torch.zeros((len(unique_keys), values.shape[1]), dtype=values.dtype, device=values.device)
    return unique_keys, sums.index_add_(0, inverse, values)


def voxel_sums(points, colours, voxel_size):
    """(sorted packed voxel keys, (V, 7) float32 sums of count, position within
    the voxel in voxel edges, and colour) per occupied voxel, from (N, 3) tensors.

    Positions are summed relative to their voxel so float32 keeps them exact
    however far the voxel is from the origin.
    """
    import torch

    scaled = points / voxel_size
    corners = torch.floor(scaled)
    indices = corners.long() + _AXIS_OFFSET
    inside = ((indices >= 0) & (indices <= _AXIS_MASK)).all(dim=1)
    values = torch.cat([torch.ones_like(scaled[:, :1]), scaled - corners, colours], dim=1).float()
    if not inside.all():
        indices, values = indices[inside], values[inside]
    keys = (indices[:, 0] << 2 * _AXIS_BITS) | (indices[:, 1] << _AXIS_BITS) | indices[:, 2]
    return _reduce_by_key(keys, values)


def merge_voxel_sums(parts):
    """Several voxel_sums results as one."""
    import torch

    return _reduce_by_key(torch.cat([keys for keys, _ in parts]), torch.cat([sums for _, sums in parts]))


def _add_existing(total, part):
    """Adds `part`'s sums into the voxels `total` already has, in place, and
    returns the rest of `part`."""
    import torch

    keys, sums = total
    part_keys, part_sums = part
    position = torch.searchsorted(keys, part_keys).clamp_(max=len(keys) - 1)
    found = keys[position] == part_keys
    sums.index_add_(0, position[found], part_sums[found])
    return part_keys[~found], part_sums[~found]


def _insert_new(total, new):
    """`total` with the voxels of `new`, which it does not have, inserted in key order."""
    import torch

    keys, sums = total
    new_keys, new_sums = new
    destination = torch.searchsorted(keys, new_keys)
    destination += torch.arange(len(new_keys), device=keys.device)
    existing = torch.ones(len(keys) + len(new_keys), dtype=torch.bool, device=keys.device)
    existing[destination] = False
    merged_keys = torch.empty(len(existing), dtype=keys.dtype, device=keys.device)
    merged_keys[destination], merged_keys[existing] = new_keys, keys
    merged_sums = torch.empty((len(existing), sums.shape[1]), dtype=sums.dtype, device=sums.device)
    merged_sums[destination], merged_sums[existing] = new_sums, sums
    return merged_keys, merged_sums


def voxel_means(part, voxel_size):
    """Each voxel's mean point and uint8 colour, as NumPy arrays."""
    keys, sums = part[0].cpu().numpy(), part[1].cpu().numpy()
    indices = np.column_stack([
        (keys >> 2 * _AXIS_BITS) & _AXIS_MASK, (keys >> _AXIS_BITS) & _AXIS_MASK, keys & _AXIS_MASK,
    ]) - _AXIS_OFFSET
    counts = sums[:, :1].astype(np.float64)
    colours = np.clip(np.rint(sums[:, 4:] / counts), 0, 255).astype(np.uint8)
    return (indices + sums[:, 1:4] / counts) * voxel_size, colours


def _vertices(points, colours_rgb):
    vertices = np.empty(len(points), dtype=_VERTEX)
    vertices["x"], vertices["y"], vertices["z"] = np.asarray(points).T
    vertices["red"], vertices["green"], vertices["blue"] = np.asarray(colours_rgb).T
    return vertices


def _write_vertices(path, count, chunks):
    """A binary PLY of `count` vertices, written from an iterable of _VERTEX arrays."""
    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {count}",
        *(f"property float {axis}" for axis in "xyz"),
        *(f"property uchar {channel}" for channel in ("red", "green", "blue")),
        "end_header",
    ])
    # Through a temporary file, so an interrupted write never leaves a partial cloud.
    temporary_path = path.with_name(f"{path.name}.partial")
    with open(temporary_path, "wb") as file:
        file.write((header + "\n").encode("ascii"))
        for vertices in chunks:
            vertices.tofile(file)
    os.replace(temporary_path, path)


def write_point_cloud(path, points, colours_rgb):
    _write_vertices(path, len(points), [_vertices(points, colours_rgb)])


def read_point_cloud(path):
    """(points, uint8 RGB colours) from a write_point_cloud file."""
    with open(path, "rb") as file:
        count = None
        for line in iter(file.readline, b"end_header\n"):
            if line.startswith(b"element vertex"):
                count = int(line.split()[-1])
        vertices = np.fromfile(file, dtype=_VERTEX, count=count)
    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]])
    return points, np.column_stack([vertices["red"], vertices["green"], vertices["blue"]])


def image_points(rows, image, intrinsics, camera_from_world, device):
    """One image's depth rows lifted into the COLMAP frame, with their RGB
    colours, as (N, 3) float64 tensors on `device`."""
    import torch

    def tensor(array):
        return torch.as_tensor(np.asarray(array, dtype=np.float64), device=device)

    rows = tensor(rows)
    image = torch.as_tensor(image, device=device)
    height, width = image.shape[:2]
    columns = torch.round(rows[:, 0] * (width - 1)).long()
    lines = torch.round(rows[:, 1] * (height - 1)).long()
    # COLMAP puts pixel centres at half-integer coordinates.
    pixels = torch.stack([columns, lines, torch.zeros_like(lines)], dim=1) + tensor([0.5, 0.5, 1])
    camera_points = pixels @ tensor(np.linalg.inv(intrinsics)).T * rows[:, 2:]
    rotation, translation = tensor(camera_from_world[:, :3]), tensor(camera_from_world[:, 3])
    return (camera_points - translation) @ rotation, image[lines, columns].flip(-1).double()


def merge_images(images, voxel_size):
    """The voxel_sums total over `images`, an iterable of (rows, image_bgr,
    intrinsics, camera_from_world), on the GPU when there is one; None if empty.

    Like a voxel hash grid, each image adds into the voxels already seen in
    place; only voxels new to the total are buffered, and inserted once they
    reach a quarter of it, so memory stays near two copies of the total.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total, new_parts, new_rows = None, [], 0
    for rows, image, intrinsics, camera_from_world in images:
        if not len(rows):
            continue
        part = voxel_sums(*image_points(rows, image, intrinsics, camera_from_world, device), voxel_size)
        if total is None:
            total = part
            continue
        new_parts.append(_add_existing(total, part))
        new_rows += len(new_parts[-1][0])
        if new_rows > max(_FOLD_ROWS, len(total[0]) // 4):
            total = _insert_new(total, merge_voxel_sums(new_parts))
            new_parts, new_rows = [], 0
    if total is not None and new_parts:
        total = _insert_new(total, merge_voxel_sums(new_parts))
    return total


def write_voxel_means(path, total, voxel_size):
    """Writes a merge_images total's voxel means as a PLY, a slice at a time so
    host memory stays bounded however large the cloud."""
    count = 0 if total is None else len(total[0])
    starts = range(0, count, _WRITE_ROWS)
    _write_vertices(path, count, (
        _vertices(*voxel_means((total[0][start:start + _WRITE_ROWS], total[1][start:start + _WRITE_ROWS]), voxel_size))
        for start in starts
    ))
    return count


def prefetched(function, items, num_threads):
    """Yields `function(item)` for each of `items` in order, computed on
    `num_threads` threads at most twice that many items ahead, so reading from
    slow storage overlaps without holding every result in memory."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(num_threads) as pool:
        pending = deque()
        for item in items:
            pending.append(pool.submit(function, item))
            if len(pending) >= 2 * num_threads:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def write_depth_point_cloud(model_dir, depth_dir, output_path, voxel_factor):
    """Merge every depth file under `depth_dir` for the COLMAP model in
    `model_dir` into `output_path`; returns the point count."""
    import cv2
    import pycolmap
    from tqdm import tqdm

    from mapping.depth.covisibility_groups import centre_and_rotation
    from mapping.depth.depth_maps import depth_rows, load_depth_map
    from mapping.file_listing import files_under

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

    written = set(files_under(depth_dir))
    with_depth = [
        image for image in reconstruction.images.values()
        if depth_dir / f"{image.name}.npy" in written
    ]

    def load(image):
        depth = load_depth_map(depth_dir / f"{image.name}.npy")
        if depth is None:
            return None
        bgr = cv2.imread(str(model_dir / "images" / image.name))
        if bgr is None:
            return None
        return (
            depth_rows(depth, bgr.shape[1], bgr.shape[0]),
            bgr,
            np.asarray(reconstruction.cameras[image.camera_id].calibration_matrix()),
            poses[image.name],
        )

    def images():
        loaded = prefetched(load, with_depth, _LOAD_THREADS)
        for item in tqdm(loaded, total=len(with_depth), desc="Point cloud", unit="image", dynamic_ncols=True):
            if item is not None:
                yield item

    total = merge_images(images(), voxel_size)
    print(f"Writing {0 if total is None else len(total[0])} voxels to {output_path}", flush=True)
    return write_voxel_means(output_path, total, voxel_size)
