"""The per-image depth files the depth stage writes, and their filtering.

Each "<image_name>.npz" holds the depth model's full output for the image at
its own resolution (416 x 416 for VGGT-Omega): `depth` as float16, already
scaled into the frame of the input model, `confidence` as a uint8 count of
CONFIDENCE_STEP, and the group's `scale` from model units to that frame. An
image without depth gets empty maps. Nothing is filtered on write: readers
choose a DepthFilter, apply it with supervised_depth, and interpolate the
result to whatever resolution they load the image at with depth_rows.
"""

import os
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


def depth_edge_mask(grayscale, depth):
    """Pixels on colour or depth discontinuities, where predicted depth is
    unreliable. At the map's resolution one pixel already spans about 5 source
    pixels of a 1920 image, so the edges are not grown further."""
    colour_edges = cv2.Canny(grayscale, 100, 200) > 0
    valid = np.isfinite(depth) & (depth > 0)
    log_depth = np.zeros(depth.shape, dtype=np.float32)
    log_depth[valid] = np.log(depth[valid])
    gradient_x = cv2.Sobel(log_depth, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(log_depth, cv2.CV_32F, 0, 1, ksize=3)
    return colour_edges | (valid & (np.hypot(gradient_x, gradient_y) > DEPTH_EDGE_THRESHOLD))


def supervised_depth(depth_map, grayscale, mask, depth_filter):
    """`depth_map`'s depth at its own resolution, 0 wherever `depth_filter`,
    the training `mask` (True to keep, at any size, or None), or a colour or
    depth edge of the `grayscale` image (at any size) rejects it."""
    depth = depth_map.depth
    height, width = depth.shape
    usable = (
        np.isfinite(depth)
        & (depth > 0)
        & (depth >= depth_filter.min_depth * depth_map.scale)
        & (depth <= depth_filter.max_depth * depth_map.scale)
        & (depth_map.confidence >= depth_filter.min_confidence)
    )
    if mask is not None:
        usable &= cv2.resize(
            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        ) > 0
    small = cv2.resize(grayscale, (width, height), interpolation=cv2.INTER_AREA)
    usable &= ~depth_edge_mask(small.astype(np.uint8), depth)
    return np.where(usable, depth, 0).astype(np.float32)


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


def depth_rows(depth, width, height):
    """(N, 3) float32 (x, y, depth) rows of the positive pixels of `depth`
    interpolated to `width` x `height`, with x and y normalized to [0, 1]."""
    depth = upsample(depth, width, height)
    lines, columns = np.nonzero(depth > 0)
    rows = np.empty((len(lines), 3), dtype=np.float32)
    np.multiply(columns, np.float32(1 / (width - 1)), out=rows[:, 0], casting="unsafe")
    np.multiply(lines, np.float32(1 / (height - 1)), out=rows[:, 1], casting="unsafe")
    rows[:, 2] = depth[lines, columns]
    return rows
