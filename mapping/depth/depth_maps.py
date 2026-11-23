"""The per-image depth files the depth stage writes.

Each "<image_name>.npy" holds the filtered depth map at the depth model's own
resolution (416 x 416 for VGGT-Omega), 3 bytes a pixel: depth as float16 in
the frame of the input model, 0 where there is no depth to supervise on, and
the model's confidence as a uint8 count of CONFIDENCE_STEP, so a reader can
filter on confidence again without regenerating. An image without depth gets
an empty (0, 0) map. Readers interpolate the map to whatever resolution they
load the image at.
"""

import os

import cv2
import numpy as np

PIXEL = np.dtype([("depth", "<f2"), ("confidence", "u1")])
# Stored confidence resolution; anything above 255 steps (12.75) saturates.
CONFIDENCE_STEP = 0.05


def save_depth_map(path, depth, confidence):
    """Writes (H, W) `depth` and its `confidence` through a temporary file, so
    an interrupted run leaves no partial file."""
    depth = np.asarray(depth)
    if not (np.abs(depth) <= np.finfo(np.float16).max).all():
        raise ValueError("depth beyond float16's range of 65504")
    pixels = np.empty(depth.shape, dtype=PIXEL)
    pixels["depth"] = depth
    pixels["confidence"] = np.clip(np.rint(np.asarray(confidence) / CONFIDENCE_STEP), 0, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.partial")
    with open(temporary_path, "wb") as file:
        np.save(file, pixels)
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


def load_depth_map(path, min_confidence=None):
    """A save_depth_map file's depth as float32, 0 below `min_confidence` when
    given, or None for an image without depth."""
    pixels = np.load(path)
    if pixels.dtype != PIXEL or pixels.ndim != 2:
        raise ValueError(f"{path} is not a depth map; regenerate the depths with --overwrite")
    if pixels.size == 0:
        return None
    depth = pixels["depth"].astype(np.float32)
    if min_confidence is not None:
        depth[pixels["confidence"] < np.rint(min_confidence / CONFIDENCE_STEP)] = 0
    return depth


def depth_rows(depth, width, height):
    """(N, 3) float32 (x, y, depth) rows of the positive pixels of `depth`
    interpolated to `width` x `height`, with x and y normalized to [0, 1]."""
    depth = upsample(depth, width, height)
    lines, columns = np.nonzero(depth > 0)
    return np.column_stack(
        [columns / (width - 1), lines / (height - 1), depth[lines, columns]]
    ).astype(np.float32)
