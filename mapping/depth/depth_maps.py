"""The per-image depth files the depth stage writes.

Each "<image_name>.npy" holds the filtered depth map at the depth model's own
resolution (416 x 416 for VGGT-Omega) as float16, in the frame of the input
model, with 0 where there is no depth to supervise on. An image without depth
gets an empty (0, 0) map. Readers interpolate the map to whatever resolution
they load the image at.
"""

import os

import cv2
import numpy as np


def save_depth_map(path, depth):
    """Writes `depth` as float16 through a temporary file, so an interrupted
    run leaves no partial file."""
    depth = np.asarray(depth)
    if not (np.abs(depth) <= np.finfo(np.float16).max).all():
        raise ValueError("depth beyond float16's range of 65504")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.partial")
    with open(temporary_path, "wb") as file:
        np.save(file, depth.astype(np.float16))
    os.replace(temporary_path, path)


def upsample(depth, width, height):
    """`depth` bilinearly interpolated to `height` x `width`, only between
    positive pixels: a pixel any 0 contributes to stays 0 rather than blending
    a surface into the background."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape == (height, width):
        return depth
    size = (width, height)
    valid = cv2.resize((depth > 0).astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    interpolated = cv2.resize(depth, size, interpolation=cv2.INTER_LINEAR)
    return np.where(valid > 1 - 1e-4, interpolated, 0).astype(np.float32)


def load_depth_map(path):
    """A save_depth_map file's map as float32, or None for an image without depth."""
    depth = np.load(path)
    if depth.ndim != 2:
        raise ValueError(f"{path} is not a depth map; regenerate the depths with --overwrite")
    return depth.astype(np.float32) if depth.size else None


def depth_rows(depth, width, height):
    """(N, 3) float32 (x, y, depth) rows of the positive pixels of `depth`
    interpolated to `width` x `height`, with x and y normalized to [0, 1]."""
    depth = upsample(depth, width, height)
    lines, columns = np.nonzero(depth > 0)
    return np.column_stack(
        [columns / (width - 1), lines / (height - 1), depth[lines, columns]]
    ).astype(np.float32)
