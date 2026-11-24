"""Depth alignment to the COLMAP points, and cross-view fusion.

Adapts HY-World 2.0's World Composition (see doc/depth_alignment.md). Each
multi-view group's prediction gets one fit in disparity, 1/D_colmap = k/D + b,
to the sparse points its images observe. Once every group is written, each
pixel is checked against the depths of the frames sharing the most sparse
points with it: zeroed where it lies in space they saw through, averaged with
them where they agree. The fused depths replace the written ones, with a
per-frame summary, SUMMARY_NAME, beside them marking them fused.
"""

import json
import time
from typing import NamedTuple

import numpy as np

SUMMARY_NAME = "fusion.json"

MIN_TRACK_LENGTH = 3
# Sparse references beyond this depth, in map units, are too noisy to fit to.
MAX_REFERENCE_DEPTH = 20.0
# A sample is dropped when its depth rank differs this many percentiles between the maps.
PERCENTILE_TOLERANCE = 20.0
INLIER_RELATIVE_ERROR = 0.05
RANSAC_ITERATIONS = 200
MIN_AFFINE_INLIERS = 30
MIN_SCALE_INLIERS = 10
# Below this ratio of the samples' largest to smallest disparity, a shift is unobservable.
MIN_DISPARITY_SPAN = 1.5
NEIGHBOUR_COUNT = 8
MIN_SHARED_POINTS = 10
CONFIRM_TOLERANCE = 0.05  # Log depth.
VIOLATION_TOLERANCE = 0.05  # Log depth.

FITTED, SCALE_ONLY = "fitted", "scale_only"


class Fit(NamedTuple):
    scale: float  # k
    shift: float  # b, in inverse map units.
    status: str
    samples: int = 0
    inliers: int = 0


def aligned_depth(depth, fit):
    """`depth` through `fit`; 0 where it is 0 or the fit sends it behind the camera."""
    depth = np.asarray(depth, dtype=np.float32)
    disparity = np.zeros_like(depth)
    valid = depth > 0
    disparity[valid] = fit.scale / depth[valid] + fit.shift
    result = np.zeros_like(depth)
    positive = valid & (disparity > 0)
    result[positive] = 1 / disparity[positive]
    return result


def usable_depths(predicted, reference):
    return (
        np.isfinite(predicted) & np.isfinite(reference)
        & (predicted > 0) & (reference > 0) & (reference <= MAX_REFERENCE_DEPTH)
    )


def median_relative_error(fit, predicted, reference):
    """How far `fit` carries one frame's sampled depths from its sparse ones; inf if unmeasured."""
    predicted, reference = np.asarray(predicted), np.asarray(reference)
    aligned = aligned_depth(predicted, fit)
    usable = usable_depths(predicted, reference) & (aligned > 0)
    if not usable.any():
        return np.inf
    return float(np.median(np.abs(aligned[usable] / reference[usable] - 1)))


def percentile_agreement(predicted, reference):
    """True where a sample's rank among the samples differs by at most
    PERCENTILE_TOLERANCE percentiles between the two depths."""
    count = len(predicted)
    if count < 2:
        return np.ones(count, dtype=bool)
    ranks = lambda values: np.argsort(np.argsort(values)) * 100.0 / (count - 1)
    return np.abs(ranks(predicted) - ranks(reference)) <= PERCENTILE_TOLERANCE


def _relative_errors(scale, shift, disparity, reference_disparity):
    """|aligned / reference - 1| per sample for each (scale, shift) row."""
    aligned = np.outer(scale, disparity) + np.asarray(shift)[:, None]
    with np.errstate(divide="ignore"):
        return np.where(aligned > 0, np.abs(reference_disparity / aligned - 1), np.inf)


def _refine_affine(disparity, reference_disparity):
    """Least squares on relative disparity error, which is relative depth error to first order."""
    design = np.column_stack([disparity, np.ones_like(disparity)]) / reference_disparity[:, None]
    (scale, shift), *_ = np.linalg.lstsq(design, np.ones(len(disparity)), rcond=None)
    return scale, shift


def _refine_scale(disparity, reference_disparity):
    ratio = disparity / reference_disparity
    return (ratio.sum() / (ratio**2).sum()), 0.0


def _ransac(disparity, reference_disparity, candidates, refine):
    """The best of the (scale, shift) `candidates` by inlier count, refined twice on its inliers."""
    errors = _relative_errors(candidates[:, 0], candidates[:, 1], disparity, reference_disparity)
    best = np.argmax((errors <= INLIER_RELATIVE_ERROR).sum(axis=1))
    scale, shift = candidates[best]
    inliers = errors[best] <= INLIER_RELATIVE_ERROR
    for _ in range(2):
        if inliers.sum() < 2:
            break
        scale, shift = refine(disparity[inliers], reference_disparity[inliers])
        inliers = _relative_errors([scale], [shift], disparity, reference_disparity)[0] <= INLIER_RELATIVE_ERROR
    return float(scale), float(shift), inliers


def fit_depth(predicted, reference, largest_depth, seed=0):
    """The Fit of `predicted` depths onto `reference` ones, in disparity, or None.

    An affine fit when enough samples span enough disparity and it keeps every
    depth up to `largest_depth` in front of the camera; else a scale.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    disparity, reference_disparity = 1 / predicted, 1 / reference
    count = len(predicted)
    if count < MIN_SCALE_INLIERS:
        return None

    random = np.random.default_rng(seed)
    if count >= MIN_AFFINE_INLIERS and disparity.max() >= MIN_DISPARITY_SPAN * disparity.min():
        pairs = random.integers(0, count, size=(RANSAC_ITERATIONS, 2))
        pairs = pairs[disparity[pairs[:, 0]] != disparity[pairs[:, 1]]]
        first, second = disparity[pairs[:, 0]], disparity[pairs[:, 1]]
        scales = (reference_disparity[pairs[:, 0]] - reference_disparity[pairs[:, 1]]) / (first - second)
        candidates = np.column_stack([scales, reference_disparity[pairs[:, 0]] - scales * first])
        candidates = candidates[candidates[:, 0] > 0]
        if len(candidates):
            scale, shift, inliers = _ransac(disparity, reference_disparity, candidates, _refine_affine)
            if (
                inliers.sum() >= MIN_AFFINE_INLIERS
                and scale > 0
                and scale / largest_depth + shift > 0
            ):
                return Fit(scale, shift, FITTED, count, int(inliers.sum()))

    candidates = np.column_stack([reference_disparity / disparity, np.zeros(count)])
    scale, _, inliers = _ransac(disparity, reference_disparity, candidates, _refine_scale)
    if inliers.sum() >= MIN_SCALE_INLIERS and scale > 0:
        return Fit(scale, 0.0, SCALE_ONLY, count, int(inliers.sum()))
    return None


def covisible_neighbours(point_indices_per_frame, point_count):
    """Each frame's up to NEIGHBOUR_COUNT frames sharing the most sparse points,
    at least MIN_SHARED_POINTS, most shared first. Each frame's indices must be unique."""
    from mapping.depth.covisibility_groups import shared_point_counts

    shared = shared_point_counts(point_indices_per_frame, point_count).toarray()
    np.fill_diagonal(shared, 0)
    neighbours = []
    for row in shared:
        order = np.argsort(-row, kind="stable")[:NEIGHBOUR_COUNT]
        neighbours.append([int(frame) for frame in order if row[frame] >= MIN_SHARED_POINTS])
    return neighbours


class View(NamedTuple):
    depth: np.ndarray  # (h, w) aligned depth, 0 where unknown.
    usable: np.ndarray  # (h, w) pixels trusted as evidence and checked.
    intrinsics: np.ndarray  # (3, 3) for the map's resolution.
    camera_from_world: np.ndarray  # (3, 4)


def map_intrinsics(intrinsics, image_size, map_shape):
    """`intrinsics` of an `image_size` (width, height) image, for a `map_shape` map of it."""
    width, height = image_size
    return np.diag([map_shape[1] / width, map_shape[0] / height, 1.0]) @ np.asarray(intrinsics)


def fused_depth(view, neighbour_views):
    """`view`'s depth with each usable pixel checked against `neighbour_views`:
    0 where it lies in free space a neighbour saw through at least as often as
    a neighbour confirms it, else averaged with the confirming depths, each
    weighted by its inverse square since depth error grows with depth. A pixel
    no neighbour sees, or that each sees behind its own surface, is kept as it
    is: occlusion is no evidence against it."""
    depth = view.depth.copy()
    if not neighbour_views:
        return depth
    from mapping.depth.point_cloud import unproject

    lines, columns = np.nonzero(view.usable)
    own = depth[lines, columns].astype(np.float64)
    world = unproject(
        np.column_stack([columns + 0.5, lines + 0.5]), own, view.intrinsics, view.camera_from_world
    )
    confirmations = np.zeros(len(lines), dtype=np.int32)
    violations = np.zeros(len(lines), dtype=np.int32)
    weights = 1 / own**2
    weighted_depths = own * weights
    for neighbour in neighbour_views:
        local = world @ neighbour.camera_from_world[:, :3].T + neighbour.camera_from_world[:, 3]
        in_front = local[:, 2] > 1e-6
        projected = local[in_front] @ neighbour.intrinsics.T
        neighbour_columns = np.floor(projected[:, 0] / projected[:, 2]).astype(np.int64)
        neighbour_lines = np.floor(projected[:, 1] / projected[:, 2]).astype(np.int64)
        height, width = neighbour.depth.shape
        inside = (
            (neighbour_columns >= 0) & (neighbour_columns < width)
            & (neighbour_lines >= 0) & (neighbour_lines < height)
        )
        rows = np.flatnonzero(in_front)[inside]
        neighbour_lines, neighbour_columns = neighbour_lines[inside], neighbour_columns[inside]
        seen = neighbour.usable[neighbour_lines, neighbour_columns]
        rows, neighbour_lines, neighbour_columns = rows[seen], neighbour_lines[seen], neighbour_columns[seen]
        neighbour_depth = neighbour.depth[neighbour_lines, neighbour_columns].astype(np.float64)
        difference = np.log(local[rows, 2] / neighbour_depth)
        confirmed = np.abs(difference) <= CONFIRM_TOLERANCE
        confirmations[rows] += confirmed
        violations[rows] += difference < -VIOLATION_TOLERANCE
        # The neighbour's surface sits at this ratio of the projected depth, along this ray too.
        rows, neighbour_depth = rows[confirmed], neighbour_depth[confirmed]
        estimate = own[rows] * neighbour_depth / local[rows, 2]
        weights[rows] += 1 / neighbour_depth**2  # Rows are unique within one neighbour.
        weighted_depths[rows] += estimate / neighbour_depth**2
    kept = (violations == 0) | (confirmations > violations)
    depth[lines, columns] = np.where(kept, weighted_depths / weights, 0)
    return depth


def frame_samples(observation, depth, confidence, track_lengths, mask, image_shape, min_confidence):
    """(predicted, reference) depths at the sparse points `observation` holds
    that pass the confidence, mask, track, and range tests."""
    from mapping.depth.generate_depth import sample_at

    predicted = sample_at(depth, observation.pixels, image_shape)
    reference = observation.depths
    keep = (
        usable_depths(predicted, reference)
        & (sample_at(confidence, observation.pixels, image_shape) >= min_confidence)
        & (track_lengths[observation.point_indices] >= MIN_TRACK_LENGTH)
    )
    if mask is not None:
        keep &= sample_at(mask, observation.pixels, image_shape)
    return predicted[keep], reference[keep]


def fit_group(samples, largest_depth):
    """The one Fit of a group's per-frame (predicted, reference) `samples`, or None.
    Samples whose depth rank disagrees between the two are left out."""
    predicted = np.concatenate([predicted for predicted, _ in samples])
    reference = np.concatenate([reference for _, reference in samples])
    agree = percentile_agreement(predicted, reference)
    return fit_depth(predicted[agree], reference[agree], largest_depth)


def fuse_depths(model_dir, depth_dir, depth_filter, mask_dir=None):
    """Replaces each depth under `depth_dir` with it fused with its
    neighbours', and writes SUMMARY_NAME; returns whether it fused, which it
    skips when SUMMARY_NAME already exists. Pixels below the confidence floor
    or outside the masks in `mask_dir` (which may be None) are kept but neither
    checked nor used as evidence."""
    import pycolmap
    from tqdm import tqdm

    from mapping.depth.depth_maps import (
        depth_path,
        load_depth_map,
        mask_at,
        save_depth_map,
        training_masks,
    )
    from mapping.depth.generate_depth import observed_depths
    from mapping.depth.point_cloud import prefetched
    from mapping.file_listing import files_under

    summary_path = depth_dir / SUMMARY_NAME
    if summary_path.exists():
        print(f"{depth_dir} is already fused; keeping it")
        return False
    started = time.monotonic()
    reconstruction = pycolmap.Reconstruction(model_dir / "sparse" / "0")
    points, observations = observed_depths(reconstruction)
    images = {image.name: image for image in reconstruction.images.values()}
    written = set(files_under(depth_dir))
    training_mask = training_masks(mask_dir)

    def load(name):
        return name, load_depth_map(depth_path(depth_dir, name)), training_mask(name)

    frames = []  # (name, view, confidence, scale) per image with depth.
    with_files = sorted(name for name in images if depth_path(depth_dir, name) in written)
    for name, depth_map, mask in tqdm(
        prefetched(load, with_files, 8), total=len(with_files), desc="Load", unit="image",
        dynamic_ncols=True,
    ):
        if depth_map is None:
            continue
        camera = reconstruction.cameras[images[name].camera_id]
        usable = (depth_map.depth > 0) & (depth_map.confidence >= depth_filter.min_confidence)
        if mask is not None:
            usable &= mask_at(mask, usable.shape)
        frames.append((name, View(
            depth_map.depth,
            usable,
            map_intrinsics(camera.calibration_matrix(), (camera.width, camera.height), usable.shape),
            observations[name].camera_from_world,
        ), depth_map.confidence, depth_map.scale))

    neighbours = covisible_neighbours(
        [np.unique(observations[name].point_indices) for name, *_ in frames], len(points)
    )
    views = [view for _, view, *_ in frames]
    summary = {}
    for frame, (name, view, confidence, scale) in enumerate(tqdm(
        frames, desc="Cross-view check", unit="image", dynamic_ncols=True,
    )):
        depth = fused_depth(view, [views[neighbour] for neighbour in neighbours[frame]])
        save_depth_map(depth_path(depth_dir, name), depth, confidence, scale)
        checked = int(view.usable.sum())
        summary[name] = {
            "neighbours": len(neighbours[frame]),
            "kept_fraction": float((view.usable & (depth > 0)).sum() / checked) if checked else None,
        }
    summary_path.write_text(json.dumps(summary, indent=1))
    kept = [entry["kept_fraction"] for entry in summary.values() if entry["kept_fraction"] is not None]
    print(
        f"Fused {len(summary)} depths in {depth_dir} in {time.monotonic() - started:.1f} s; "
        f"the cross-view check kept a median {np.median(kept) if kept else float('nan'):.0%} of pixels",
        flush=True,
    )
    return True


def main():
    import argparse
    from pathlib import Path

    from mapping.depth.depth_maps import DepthFilter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="holds images/ and sparse/")
    parser.add_argument("--depth-dir", type=Path, help="defaults to <model_dir>/depths")
    parser.add_argument("--min-confidence", type=float, default=DepthFilter().min_confidence)
    parser.add_argument("--mask-dir", type=Path, default=None)
    args = parser.parse_args()
    fuse_depths(
        args.model_dir,
        args.depth_dir or args.model_dir / "depths",
        DepthFilter(min_confidence=args.min_confidence),
        args.mask_dir,
    )


if __name__ == "__main__":
    main()
