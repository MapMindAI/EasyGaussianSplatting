"""The per-image depth files the depth stage writes, and their filtering.

Each "<image_name>.npz" holds the depth model's full output for the image at
its own resolution (416 x 416 for VGGT-Omega): `depth` as float16, already
scaled into the frame of the input model, `confidence` as a uint8 count of
CONFIDENCE_STEP, and the group's `scale` from model units to that frame. An
image without depth gets empty maps. The only filtering on write zeroes depth
discontinuities, the floaters a model predicts between a surface and what lies
behind it. Readers choose a DepthFilter, apply it with supervised_depth, and
take one point per kept map pixel, at its centre in the image as they load
it, with depth_rows.
"""

import os
import zipfile
from typing import NamedTuple

import cv2
import numpy as np

DEPTH_SUFFIX = ".npz"
# Stored confidence resolution; anything above 255 steps (12.75) saturates.
CONFIDENCE_STEP = 0.05
DEPTH_EDGE_THRESHOLD = 0.15


class DepthMap(NamedTuple):
    depth: np.ndarray  # (H, W) float32 in the input model's frame.
    confidence: np.ndarray  # (H, W) float32.
    scale: float  # From the depth model's units to the input model's frame.


class DepthFilter(NamedTuple):
    """Which pixels of a DepthMap to supervise on. The depth range is in the
    depth model's units, before scaling."""
    min_confidence: float = 0.5
    min_depth: float = 0.1
    max_depth: float = 5.0


def depth_path(depth_dir, image_name):
    return depth_dir / f"{image_name}{DEPTH_SUFFIX}"


def save_depth_map(path, depth, confidence, scale):
    """Writes one image's `depth`, `confidence`, and `scale` through a
    temporary file, so an interrupted run leaves no partial file."""
    depth = np.asarray(depth)
    if not (np.abs(depth) <= np.finfo(np.float16).max).all():
        raise ValueError("depth beyond float16's range of 65504")
    confidence_steps = np.clip(np.rint(np.asarray(confidence) / CONFIDENCE_STEP), 0, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.partial")
    with open(temporary_path, "wb") as file:
        np.savez(
            file,
            depth=depth.astype(np.float16),
            confidence=confidence_steps.astype(np.uint8),
            scale=np.float64(scale),
        )
    os.replace(temporary_path, path)


def save_empty_depth_map(path):
    save_depth_map(path, np.zeros((0, 0)), np.zeros((0, 0)), np.nan)


def holds_depth(path):
    """Whether a save_depth_map file holds a depth, read from its header alone."""
    with zipfile.ZipFile(path) as archive, archive.open("depth.npy") as file:
        version = np.lib.format.read_magic(file)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(file)
        else:
            shape, _, _ = np.lib.format.read_array_header_2_0(file)
    return int(np.prod(shape)) > 0


def load_depth_map(path):
    """A save_depth_map file as a DepthMap, or None for an image without depth."""
    with np.load(path) as stored:
        if stored["depth"].size == 0:
            return None
        return DepthMap(
            stored["depth"].astype(np.float32),
            stored["confidence"] * np.float32(CONFIDENCE_STEP),
            float(stored["scale"]),
        )


def training_masks(mask_dir):
    """A function from an image name to its training mask, True where training
    looks, or None when it has none. Lists `mask_dir` (which may be None) once."""
    from mapping.file_listing import files_under

    listed = set() if mask_dir is None else set(files_under(mask_dir))

    def mask(image_name):
        path = None if mask_dir is None else mask_dir / f"{image_name}.png"
        # Checked first: OpenCV warns on every missing file it is asked to read.
        if path not in listed:
            return None
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return None if image is None else image > 127

    return mask


def without_depth_discontinuities(depth):
    """`depth` with 0 wherever its log-depth gradient jumps past
    DEPTH_EDGE_THRESHOLD: pixels straddling a surface and its background, which
    a model fills with depths in between. Scale-free, so it can run on depth in
    any units. At the map's resolution one pixel already spans about 5 source
    pixels of a 1920 image, so the edges are not grown further."""
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    log_depth = np.zeros(depth.shape, dtype=np.float32)
    log_depth[valid] = np.log(depth[valid])
    gradient_x = cv2.Sobel(log_depth, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(log_depth, cv2.CV_32F, 0, 1, ksize=3)
    return np.where(valid & (np.hypot(gradient_x, gradient_y) <= DEPTH_EDGE_THRESHOLD), depth, 0)


def supervised_depth(depth_map, mask, depth_filter):
    """`depth_map`'s depth at its own resolution, 0 wherever `depth_filter` or
    the training `mask` (True to keep, at any size, or None) rejects it."""
    depth = depth_map.depth
    usable = (
        np.isfinite(depth)
        & (depth > 0)
        & (depth >= depth_filter.min_depth * depth_map.scale)
        & (depth <= depth_filter.max_depth * depth_map.scale)
        & (depth_map.confidence >= depth_filter.min_confidence)
    )
    if mask is not None:
        usable &= mask_at(mask, depth.shape)
    return np.where(usable, depth, 0).astype(np.float32)


def mask_at(mask, shape):
    """The boolean `mask`, at any size, resized by nearest neighbour to `shape` (height, width)."""
    return cv2.resize(mask.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST) > 0


def upsample(depth, width, height):
    """`depth` bilinearly interpolated to `height` x `width` for display, only between
    positive pixels: a pixel any 0 contributes to stays 0 rather than blending
    a surface into the background."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape == (height, width):
        return depth
    size = (width, height)
    valid = cv2.resize((depth > 0).astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    interpolated = cv2.resize(depth, size, interpolation=cv2.INTER_LINEAR)
    return np.where(valid > 1 - 1e-4, interpolated, 0).astype(np.float32)


def depth_rows(depth, width, height):
    """(N, 3) float32 (x, y, depth) rows, one per positive pixel of `depth`,
    placed at that pixel's centre in a `width` x `height` image of the same
    view. x and y are pixel coordinates over (size - 1), clipped to [0, 1].

    The map holds every pixel the model predicted; interpolating it to the
    image's resolution would only add points carrying no new information."""
    map_height, map_width = depth.shape
    lines, columns = np.nonzero(depth > 0)
    rows = np.empty((len(lines), 3), dtype=np.float32)
    rows[:, 0] = ((columns + 0.5) * (width / map_width) - 0.5) / (width - 1)
    rows[:, 1] = ((lines + 0.5) * (height / map_height) - 0.5) / (height - 1)
    np.clip(rows[:, :2], 0, 1, out=rows[:, :2])
    rows[:, 2] = depth[lines, columns]
    return rows
