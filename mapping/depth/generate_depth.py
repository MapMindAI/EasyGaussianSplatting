#!/usr/bin/env python3
"""Predict per-image depth for a COLMAP model with a multi-view depth model.

The model's images are split into groups of frames that view the same part of
the scene (see mapping/depth/covisibility_groups.py), each predicted in one
request.

Each group is reconstructed in its own arbitrary scale, so the depth it predicts
is fitted to the COLMAP points the group's images observe and to the COLMAP
camera baselines against the predicted cameras (see
mapping/depth/scale_alignment.py). Groups without enough of that evidence are
skipped rather than written at a guessed scale, and are not retried by a
rerun. A frame in several groups keeps the prediction that best matches its own
sparse depths.

Every image in the model ends up with a file: one no group predicted -- an
up/down face, a frame in no covisible group, or one whose group failed -- gets
an empty one. Existing files are kept, so an interrupted run resumes by
predicting only the groups that still have an image without one.

The output is one "<image_name>.npz" per image under --depth-dir: the model's
full depth and confidence at its own resolution, scaled into the frame of the
input model (see mapping/depth/depth_maps.py). Nothing is filtered here; the
point cloud merge and training filter on confidence, masks, depth range, and
edges when they load it.

Run as a script, it predicts with DA3 on Triton, reached at --triton-url or
$TRITON_URL; mapping/vggt_omega/generate_depth.py runs VGGT-Omega instead.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from mapping.depth.covisibility_groups import Frame, centre_and_rotation, select_groups
from mapping.depth.depth_maps import (
    DepthFilter,
    DepthMap,
    depth_path,
    save_depth_map,
    save_empty_depth_map,
    supervised_depth,
    training_masks,
    upsample,
)
from mapping.depth.scale_alignment import align_sparse_depth, median_relative_error, pose_depth_scale
from mapping.file_listing import files_under
from mapping.triton.endpoint import add_endpoint_argument, resolve_endpoint

# Views per DA3 request. The served model's input shape is fixed, so this has
# to match the deployed model (see third_party/EasyTensorRT).
GROUP_SIZE = 5
SKIPPED_DEPTH_FACES = {"up", "down"}
# Why a group produced nothing.
UNREADABLE, UNSCALED = "unreadable", "unscaled"


def sample_at(values, points, image_shape):
    """`values`, a map of an image of `image_shape` resized to its own shape,
    read at the cell holding each COLMAP (x, y) in `points`, whose pixel
    centres sit at half-integer coordinates."""
    height, width = values.shape[:2]
    columns = np.floor(points[:, 0] * width / image_shape[1]).astype(np.int64)
    lines = np.floor(points[:, 1] * height / image_shape[0]).astype(np.int64)
    return values[np.clip(lines, 0, height - 1), np.clip(columns, 0, width - 1)]


def depth_overlay(image, depth, alpha=0.5):
    """A colourized `depth` map blended over its source BGR `image`."""
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        low, high = np.percentile(depth[valid], (2, 98))
        if high > low:
            normalized[valid] = np.clip(
                (depth[valid] - low) * 255 / (high - low), 0, 255
            ).astype(np.uint8)
    colours = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    overlay = image.copy()
    blended = cv2.addWeighted(image, 1.0 - alpha, colours, alpha, 0.0)
    overlay[valid] = blended[valid]
    return overlay


def depth_debug_path(debug_dir, name):
    path = Path(name)
    return debug_dir / path.with_name(f"{path.stem}_depth.png")


class Observations(NamedTuple):
    camera_from_world: np.ndarray  # (3, 4)
    pixels: np.ndarray  # (P, 2) xy of each observed point.
    depths: np.ndarray  # (P,) its depth along the camera axis.
    point_indices: np.ndarray  # (P,) its row in the point table.


def observed_depths(reconstruction):
    """(points, observations): the (N, 3) COLMAP point table, and each image's
    pose and observed points by image name.

    These are the same triangulated points gsplat's own depth option supervises
    on; here they group the images and pin down one scale per group.
    """
    point_ids = np.array(sorted(reconstruction.points3D), dtype=np.int64)
    points = np.array(
        [reconstruction.points3D[point_id].xyz for point_id in point_ids], dtype=np.float64
    ).reshape(-1, 3)
    observations = {}
    for image in reconstruction.images.values():
        camera_from_world = image.cam_from_world
        if callable(camera_from_world):
            camera_from_world = camera_from_world()
        camera_from_world = np.asarray(camera_from_world.matrix(), dtype=np.float64)
        observed = image.get_observation_points2D()
        point_indices = np.searchsorted(
            point_ids, np.array([point.point3D_id for point in observed], dtype=np.int64)
        )
        observations[image.name] = Observations(
            camera_from_world,
            np.array([point.xy for point in observed], dtype=np.float64).reshape(-1, 2),
            points[point_indices] @ camera_from_world[2, :3] + camera_from_world[2, 3],
            point_indices,
        )
    return points, observations


def depth_groups(points, observations, group_size, minimum_group_size):
    """(groups of image names, names no group could hold), skipping up/down faces."""
    names = sorted(
        name for name in observations if Path(name).parts[0] not in SKIPPED_DEPTH_FACES
    )
    frames = []
    for name in names:
        observation = observations[name]
        center, rotation = centre_and_rotation(observation.camera_from_world)
        in_front = observation.point_indices[observation.depths > 1e-6]
        frames.append(Frame(center, rotation, np.unique(in_front)))
    groups, rejected = select_groups(frames, points, group_size, minimum_group_size)
    return (
        [[names[index] for index in group] for group in groups],
        [names[index] for index in rejected],
    )


def pending_groups(groups, done):
    """The groups with at least one image not in `done`."""
    return [group for group in groups if not set(group) <= done]


def parse_depth_arguments(parser):
    """`parser`'s arguments, plus those every depth stage shares."""
    parser.add_argument("model_dir", type=Path, help="holds images/ and sparse/")
    parser.add_argument("depth_dir", type=Path)
    # The filter only shapes the debug overlays here; readers apply their own.
    defaults = DepthFilter()
    parser.add_argument(
        "--min-confidence", type=float, default=defaults.min_confidence,
        help="depth confidence floor",
    )
    parser.add_argument(
        "--min-depth", type=float, default=defaults.min_depth,
        help="minimum depth to keep, in the depth model's units before scaling",
    )
    parser.add_argument(
        "--max-depth", type=float, default=defaults.max_depth,
        help="maximum depth to keep, in the depth model's units before scaling",
    )
    parser.add_argument(
        "--mask-dir", type=Path, default=None, help="skip pixels these masks block"
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="write colourized overlays of the filtered depth over the source images here",
    )
    parser.add_argument(
        "--num-threads", type=int, default=4, help="worker threads running inference"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.min_depth > args.max_depth:
        parser.error("--min-depth cannot exceed --max-depth")
    return args


def depth_filter(args):
    """The DepthFilter parse_depth_arguments' flags describe."""
    return DepthFilter(args.min_confidence, args.min_depth, args.max_depth)


def generate_depths(args, make_estimator, group_size, minimum_group_size):
    """Write depth maps for `args.model_dir` with `make_estimator()`'s model,
    over groups of `minimum_group_size` to `group_size` images.

    The estimator's run(images_bgr) returns per-image `depth_list` and
    `depth_conf_list` maps and its predicted `camera_from_world_list`. Returns
    how many images it predicted, not counting the empty files.
    """
    # Deferred so the helpers above stay testable without pycolmap or tqdm installed.
    import pycolmap
    from tqdm import tqdm

    started = time.monotonic()
    images_dir = args.model_dir / "images"
    reconstruction = pycolmap.Reconstruction(args.model_dir / "sparse" / "0")
    names = [image.name for image in reconstruction.images.values()]
    written = set() if args.overwrite else set(files_under(args.depth_dir))
    done = {name for name in names if depth_path(args.depth_dir, name) in written}
    if len(done) == len(names):
        print(f"Depths already exist in {args.depth_dir} for all {len(done)} images; "
              "pass --overwrite to regenerate")
        return 0
    points, observations = observed_depths(reconstruction)
    print(
        f"Loaded {args.model_dir / 'sparse' / '0'}: {len(observations)} images, "
        f"{len(points)} points ({time.monotonic() - started:.1f} s)", flush=True
    )

    stage_started = time.monotonic()
    groups, ungrouped = depth_groups(points, observations, group_size, minimum_group_size)
    sizes = [len(group) for group in groups] or [0]
    print(
        f"Grouped {len({name for group in groups for name in group})} images "
        f"into {len(groups)} groups of {min(sizes)}-{max(sizes)} (mean {np.mean(sizes):.1f}); "
        f"{len(ungrouped)} images in no group ({time.monotonic() - stage_started:.1f} s)",
        flush=True,
    )
    pending = pending_groups(groups, done)
    if done:
        print(
            f"Resuming: {len(done)} of {len(observations)} images already have depth; "
            f"{len(pending)} of {len(groups)} groups still to predict", flush=True
        )
    groups = pending

    if groups:
        stage_started = time.monotonic()
        estimator = make_estimator()
        print(f"Loaded the depth model ({time.monotonic() - stage_started:.1f} s)", flush=True)

    def estimate(group):
        """One group's depth maps and sparse error per image, or (None, why)."""
        images = []
        for name in group:
            image = cv2.imread(str(images_dir / name))
            if image is None:
                tqdm.write(f"Skipping a group with unreadable image {name}")
                return None, UNREADABLE
            images.append(image)

        result = estimator.run(images)
        predicted = [
            sample_at(depth, observations[name].pixels, image.shape)
            for name, image, depth in zip(group, images, result["depth_list"])
        ]
        pose_scale = pose_depth_scale(
            result["camera_from_world_list"],
            [observations[name].camera_from_world for name in group],
        )
        scale = align_sparse_depth(
            np.repeat(np.arange(len(group)), [len(values) for values in predicted]),
            np.concatenate([observations[name].point_indices for name in group]),
            np.concatenate(predicted),
            np.concatenate([observations[name].depths for name in group]),
            pose_scale,
        )
        if scale is None:
            return None, UNSCALED
        group_scaled_by_cameras.append(pose_scale is not None)

        maps_by_name = []
        for name, image, depth, confidence, sparse_predicted in zip(
            group, images, result["depth_list"], result["depth_conf_list"], predicted
        ):
            error = median_relative_error(scale, sparse_predicted, observations[name].depths)
            depth_map = DepthMap(depth * scale, confidence, scale)
            overlay_image = image if args.debug_dir is not None else None
            maps_by_name.append((name, depth_map, error, overlay_image))
        return maps_by_name, None

    def write_best(maps_by_name):
        """Save each image's map unless an earlier group fitted it better."""
        for name, depth_map, error, image in maps_by_name:
            if name in best_error and error >= best_error[name]:
                continue
            best_error[name] = error
            save_depth_map(depth_path(args.depth_dir, name), *depth_map)
            if image is not None:
                grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                depth = supervised_depth(depth_map, grayscale, training_mask(name), overlay_filter)
                overlay_path = depth_debug_path(args.debug_dir, name)
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                full_depth = upsample(depth, image.shape[1], image.shape[0])
                cv2.imwrite(str(overlay_path), depth_overlay(image, full_depth))

    overlay_filter = depth_filter(args)
    training_mask = training_masks(args.mask_dir if args.debug_dir is not None else None)
    # Groups share frames, so the writes stay on this thread: two workers
    # saving the same image's file would race.
    stage_started = time.monotonic()
    # A resumed run keeps every file already written, whatever its error.
    best_error = dict.fromkeys(done, -np.inf)
    skipped = {UNREADABLE: 0, UNSCALED: 0}
    group_scaled_by_cameras = []
    with ThreadPoolExecutor(max_workers=args.num_threads) as pool:
        progress = tqdm(
            pool.map(estimate, groups), total=len(groups), desc="Depth", unit="group",
            dynamic_ncols=True,
        )
        for maps_by_name, failure in progress:
            if failure is not None:
                skipped[failure] += 1
            else:
                write_best(maps_by_name)
            progress.set_postfix(
                images=len(best_error), unscaled=skipped[UNSCALED],
                unreadable=skipped[UNREADABLE], refresh=False,
            )

    predicted = len(best_error) - len(done)
    without_depth = sorted(set(observations) - set(best_error))
    for name in without_depth:
        save_empty_depth_map(depth_path(args.depth_dir, name))

    errors = [error for error in best_error.values() if np.isfinite(error)]
    print(
        f"Depths written to {args.depth_dir} ({predicted} images from "
        f"{len(groups)} groups in {time.monotonic() - stage_started:.1f} s, "
        f"{len(without_depth)} left empty; "
        f"{sum(group_scaled_by_cameras)} groups scaled with camera baselines; "
        f"{skipped[UNSCALED]} skipped for lack of scale evidence, "
        f"{skipped[UNREADABLE]} for unreadable images; {len(ungrouped)} images in no group "
        f"of {minimum_group_size} or more; median sparse depth error "
        f"{np.median(errors) if errors else float('nan'):.3f})"
    )
    if args.debug_dir is not None:
        print(f"Depth overlays written to {args.debug_dir}")
    return predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_endpoint_argument(parser, "depth model")
    args = parse_depth_arguments(parser)
    triton_url = resolve_endpoint(args.triton_url)

    def make_estimator():
        # Deferred so the helpers above stay testable without the EasyTensorRT
        # submodule or the Triton client installed.
        from mapping.triton.clients import DepthEstimator

        return DepthEstimator(triton_url, expected_num_images=GROUP_SIZE)

    generate_depths(args, make_estimator, GROUP_SIZE, minimum_group_size=GROUP_SIZE)


if __name__ == "__main__":
    main()
