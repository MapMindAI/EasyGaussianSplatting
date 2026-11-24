"""Cross-view fusion of the written depths.

Adapts HY-World 2.0's World Composition (see doc/cross_view_fusion.md). Once
every depth is written, each pixel is checked against the depths of the frames
sharing the most sparse points with it: zeroed where it lies in space they saw
through, averaged with them where they agree. The fused depths replace the
written ones, with a per-frame summary, SUMMARY_NAME, beside them marking them
fused.
"""

import json
import os
import time
from typing import NamedTuple

import numpy as np

SUMMARY_NAME = "fusion.json"
FUSING_DIR_NAME = ".fusing"
# Views held while fusing, about 1.5 MB each at 416 x 416 with their confidence.
VIEW_CACHE_SIZE = 128
# Each check holds a few float64 copies of its pixels, so the pool is capped
# rather than one thread per core on a large server.
FUSION_THREADS = min(8, os.cpu_count() or 1)

NEIGHBOUR_COUNT = 8
MIN_SHARED_POINTS = 10
CONFIRM_TOLERANCE = 0.05  # Log depth.
VIOLATION_TOLERANCE = 0.05  # Log depth.


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
    depth: np.ndarray  # (h, w) depth in map units, 0 where unknown.
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


def fusion_order(names):
    """`names` by frame, then by face, so a frame's covisible neighbours -- the
    other faces of the same and nearby frames -- are fused close together."""
    from pathlib import PurePosixPath

    return sorted(names, key=lambda name: (PurePosixPath(name).stem, name))


def fuse_depths(model_dir, depth_dir, depth_filter, mask_dir=None):
    """Replaces each depth under `depth_dir` with it fused with its
    neighbours', and writes SUMMARY_NAME; returns whether it fused, which it
    skips when SUMMARY_NAME already exists. Pixels below the confidence floor
    or outside the masks in `mask_dir` (which may be None) are kept but neither
    checked nor used as evidence.

    Views are loaded on this thread as they are needed and at most
    VIEW_CACHE_SIZE are held; FUSION_THREADS workers check and save them. The
    fused maps go to FUSING_DIR_NAME first, so every neighbour is read unfused,
    and replace the originals once all are written."""
    import shutil
    from collections import OrderedDict

    import pycolmap
    from tqdm import tqdm

    from mapping.depth.depth_maps import (
        depth_path,
        holds_depth,
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
    by_name = sorted(
        name for name in images
        if depth_path(depth_dir, name) in written and holds_depth(depth_path(depth_dir, name))
    )
    # Picked in name order, so ties in shared points break as they always have.
    neighbours_by_name = covisible_neighbours(
        [np.unique(observations[name].point_indices) for name in by_name], len(points)
    )
    names = fusion_order(by_name)
    position = {name: index for index, name in enumerate(names)}
    neighbours = [None] * len(names)
    for name, named_neighbours in zip(by_name, neighbours_by_name):
        neighbours[position[name]] = [position[by_name[neighbour]] for neighbour in named_neighbours]

    cache = OrderedDict()  # Frame index to (View, confidence, scale), least recent first.

    def frame(index):
        if index in cache:
            cache.move_to_end(index)
            return cache[index]
        name = names[index]
        depth_map = load_depth_map(depth_path(depth_dir, name))
        mask = training_mask(name)
        camera = reconstruction.cameras[images[name].camera_id]
        usable = (depth_map.depth > 0) & (depth_map.confidence >= depth_filter.min_confidence)
        if mask is not None:
            usable &= mask_at(mask, usable.shape)
        cache[index] = (View(
            depth_map.depth,
            usable,
            map_intrinsics(camera.calibration_matrix(), (camera.width, camera.height), usable.shape),
            observations[name].camera_from_world,
        ), depth_map.confidence, depth_map.scale)
        if len(cache) > VIEW_CACHE_SIZE:
            cache.popitem(last=False)
        return cache[index]

    fusing_dir = depth_dir / FUSING_DIR_NAME
    shutil.rmtree(fusing_dir, ignore_errors=True)  # Left by an interrupted run.

    def check(item):
        name, (view, confidence, scale), neighbour_views = item
        depth = fused_depth(view, neighbour_views)
        save_depth_map(depth_path(fusing_dir, name), depth, confidence, scale)
        checked = int(view.usable.sum())
        return name, {
            "neighbours": len(neighbour_views),
            "kept_fraction": float((view.usable & (depth > 0)).sum() / checked) if checked else None,
        }

    # Generated on this thread, so only it touches the cache.
    items = (
        (name, frame(index), [frame(neighbour)[0] for neighbour in neighbours[index]])
        for index, name in enumerate(names)
    )
    summary = dict(tqdm(
        prefetched(check, items, FUSION_THREADS), total=len(names), desc="Cross-view check",
        unit="image", dynamic_ncols=True,
    ))
    for name in names:
        os.replace(depth_path(fusing_dir, name), depth_path(depth_dir, name))
    shutil.rmtree(fusing_dir)
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
