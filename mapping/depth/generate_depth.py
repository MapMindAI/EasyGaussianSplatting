#!/usr/bin/env python3
"""Predict per-image depth for a COLMAP model with a multi-view depth model.

The model's images are split into groups of frames that view the same part of
the scene (see mapping/depth/covisibility_groups.py), each predicted in one
request.

Each group is reconstructed in its own arbitrary scale, so the depth it predicts
is fitted to the COLMAP points the group's images observe and to the COLMAP
camera baselines against the predicted cameras (see
mapping/depth/scale_alignment.py). Groups without enough of that evidence are
skipped rather than written at a guessed scale. A frame in several groups keeps
the prediction that best matches its own sparse depths.

The output is one "<image_name>.npy" per image under --depth-dir, holding the
(M, 3) float32 columns (x, y, depth) that the trainer supervises on: depth in
the frame of the input model, and x and y normalized to [0, 1] so the file
survives whatever resolution the trainer loads the images at. Sampling only the
supervised pixels keeps the stage's output megabytes rather than gigabytes.

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
from mapping.depth.scale_alignment import align_sparse_depth, median_relative_error, pose_depth_scale
from mapping.triton.endpoint import add_endpoint_argument, resolve_endpoint

# Views per DA3 request. The served model's input shape is fixed, so this has
# to match the deployed model (see third_party/EasyTensorRT).
GROUP_SIZE = 5
SKIPPED_DEPTH_FACES = {"up", "down"}
EDGE_DILATION = 2
DEPTH_EDGE_THRESHOLD = 0.15
# Why a group produced nothing.
UNREADABLE, UNSCALED = "unreadable", "unscaled"


def sample_at(image_values, points):
    """`image_values` read at each integer-rounded (x, y) in `points`."""
    height, width = image_values.shape[:2]
    columns = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, width - 1)
    rows = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, height - 1)
    return image_values[rows, columns]


def supervision_mask(depth, confidence, mask, min_confidence, min_depth, max_depth):
    """Pixels whose depth is suitable for supervision."""
    usable = np.isfinite(depth) & (depth > 0)
    if min_depth is not None:
        usable &= depth >= min_depth
    if max_depth is not None:
        usable &= depth <= max_depth
    if confidence is not None:
        usable &= confidence >= min_confidence
    if mask is not None:
        usable &= mask
    return usable


def depth_edge_mask(image, depth):
    """Pixels near colour or depth discontinuities, where DA3 is unreliable."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    colour_edges = cv2.Canny(grayscale, 100, 200) > 0
    valid = np.isfinite(depth) & (depth > 0)
    log_depth = np.zeros(depth.shape, dtype=np.float32)
    log_depth[valid] = np.log(depth[valid])
    gradient_x = cv2.Sobel(log_depth, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(log_depth, cv2.CV_32F, 0, 1, ksize=3)
    depth_edges = valid & (np.hypot(gradient_x, gradient_y) > DEPTH_EDGE_THRESHOLD)
    edges = colour_edges | depth_edges
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * EDGE_DILATION + 1, 2 * EDGE_DILATION + 1)
    )
    return cv2.dilate(edges.astype(np.uint8), kernel).astype(bool)


def supervised_points(
    depth, confidence, mask, count, min_confidence, min_depth, max_depth, generator
):
    """Up to `count` (x, y, depth) rows worth supervising, as float32.

    Keeps confident, positive-depth pixels that the training mask also keeps,
    then subsamples: neighbouring pixels of a smooth depth map say almost the
    same thing, so a few thousand per image carry the geometry at a fraction of
    the size. x and y come back normalized to [0, 1].
    """
    usable = supervision_mask(
        depth, confidence, mask, min_confidence, min_depth, max_depth
    )
    rows, columns = np.nonzero(usable)
    if rows.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if count > 0 and rows.size > count:
        chosen = generator.choice(rows.size, size=count, replace=False)
        rows, columns = rows[chosen], columns[chosen]
    height, width = depth.shape[:2]
    return np.stack(
        [columns / (width - 1), rows / (height - 1), depth[rows, columns]], axis=1
    ).astype(np.float32)


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


def parse_depth_arguments(parser):
    """`parser`'s arguments, plus those every depth stage shares."""
    parser.add_argument("model_dir", type=Path, help="holds images/ and sparse/")
    parser.add_argument("depth_dir", type=Path)
    parser.add_argument(
        "--samples-per-image", type=int, default=4096, help="supervised pixels kept"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.5, help="depth confidence floor"
    )
    parser.add_argument(
        "--min-depth", type=float, default=0.1,
        help="minimum depth to keep, in the depth model's units before scaling"
    )
    parser.add_argument(
        "--max-depth", type=float, default=5.0,
        help="maximum depth to keep, in the depth model's units before scaling"
    )
    parser.add_argument(
        "--mask-dir", type=Path, default=None, help="skip pixels these masks block"
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="write colourized depth overlays over the source images here",
    )
    parser.add_argument(
        "--num-threads", type=int, default=4, help="worker threads running inference"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.min_depth > args.max_depth:
        parser.error("--min-depth cannot exceed --max-depth")
    return args


def generate_depths(args, make_estimator, group_size, minimum_group_size):
    """Write depth rows for `args.model_dir` with `make_estimator()`'s model,
    over groups of `minimum_group_size` to `group_size` images.

    The estimator's run(images_bgr) returns per-image `depth_list` and
    `depth_conf_list` maps and its predicted `camera_from_world_list`.
    """
    # Deferred so the helpers above stay testable without pycolmap or tqdm installed.
    import pycolmap
    from tqdm import tqdm

    # Groups share frames, so a partial run cannot be resumed group by group.
    if not args.overwrite and any(args.depth_dir.rglob("*.npy")):
        print(f"Depths already exist in {args.depth_dir}; pass --overwrite to regenerate")
        return

    started = time.monotonic()
    images_dir = args.model_dir / "images"
    points, observations = observed_depths(
        pycolmap.Reconstruction(args.model_dir / "sparse" / "0")
    )
    print(
        f"Loaded {args.model_dir / 'sparse' / '0'}: {len(observations)} images, "
        f"{len(points)} points ({time.monotonic() - started:.1f} s)", flush=True
    )

    stage_started = time.monotonic()
    groups, ungrouped = depth_groups(points, observations, group_size, minimum_group_size)
    if not groups:
        raise SystemExit(f"No covisible group of {minimum_group_size} images in {args.model_dir}")
    sizes = [len(group) for group in groups]
    print(
        f"Grouped {len({name for group in groups for name in group})} images "
        f"into {len(groups)} groups of {min(sizes)}-{max(sizes)} (mean {np.mean(sizes):.1f}); "
        f"{len(ungrouped)} images in no group ({time.monotonic() - stage_started:.1f} s)",
        flush=True,
    )

    stage_started = time.monotonic()
    args.depth_dir.mkdir(parents=True, exist_ok=True)
    estimator = make_estimator()
    print(f"Loaded the depth model ({time.monotonic() - stage_started:.1f} s)", flush=True)

    mask_dir = args.mask_dir
    if mask_dir is not None and not mask_dir.is_dir():
        print(f"No masks at {mask_dir}; sampling every image unmasked", flush=True)
        mask_dir = None

    def training_mask(name, shape):
        """The segmentation mask for `name` at `shape`, or None if there is none."""
        path = None if mask_dir is None else mask_dir / f"{name}.png"
        # Checked first: OpenCV warns on every missing file it is asked to read.
        if path is None or not path.is_file():
            return None
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if mask.shape[:2] != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask > 127

    def estimate(group):
        """One group's rows and sparse error per image, or (None, why)."""
        images = []
        for name in group:
            image = cv2.imread(str(images_dir / name))
            if image is None:
                tqdm.write(f"Skipping a group with unreadable image {name}")
                return None, UNREADABLE
            images.append(image)

        result = estimator.run(images)
        predicted = [
            sample_at(depth, observations[name].pixels)
            for name, depth in zip(group, result["depth_list"])
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

        # Seeded per group so a rerun samples the same pixels; the groups run
        # on worker threads, which would otherwise interleave the draws.
        generator = np.random.default_rng(0)
        rows_by_name = []
        for name, image, depth, confidence, sparse_predicted in zip(
            group, images, result["depth_list"], result["depth_conf_list"], predicted
        ):
            error = median_relative_error(scale, sparse_predicted, observations[name].depths)
            depth = depth * scale
            usable = supervision_mask(
                depth,
                confidence,
                training_mask(name, depth.shape[:2]),
                args.min_confidence,
                # The depth range is in the model's own units, before scaling.
                args.min_depth * scale,
                args.max_depth * scale,
            )
            filtered_depth = np.where(usable & ~depth_edge_mask(image, depth), depth, 0)
            rows = supervised_points(
                filtered_depth, None, None, args.samples_per_image, 0, None, None, generator
            )
            # Only the debug overlay needs the full-resolution maps back.
            overlay = (image, filtered_depth) if args.debug_dir is not None else None
            rows_by_name.append((name, rows, error, overlay))
        return rows_by_name, None

    def write_best(rows_by_name):
        """Save each image's rows unless an earlier group fitted it better."""
        for name, rows, error, overlay in rows_by_name:
            if name in best_error and error >= best_error[name]:
                continue
            best_error[name] = error
            output_path = args.depth_dir / f"{name}.npy"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, rows)
            if overlay is not None:
                overlay_path = depth_debug_path(args.debug_dir, name)
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(overlay_path), depth_overlay(*overlay))

    # Groups share frames, so the writes stay on this thread: two workers
    # saving the same image's file would race.
    stage_started = time.monotonic()
    best_error = {}
    skipped = {UNREADABLE: 0, UNSCALED: 0}
    group_scaled_by_cameras = []
    with ThreadPoolExecutor(max_workers=args.num_threads) as pool:
        progress = tqdm(
            pool.map(estimate, groups), total=len(groups), desc="Depth", unit="group",
            dynamic_ncols=True,
        )
        for rows_by_name, failure in progress:
            if failure is not None:
                skipped[failure] += 1
            else:
                write_best(rows_by_name)
            progress.set_postfix(
                images=len(best_error), unscaled=skipped[UNSCALED],
                unreadable=skipped[UNREADABLE], refresh=False,
            )

    errors = [error for error in best_error.values() if np.isfinite(error)]
    print(
        f"Depths written to {args.depth_dir} ({len(best_error)} images from "
        f"{len(groups)} groups in {time.monotonic() - stage_started:.1f} s; "
        f"{sum(group_scaled_by_cameras)} groups scaled with camera baselines; "
        f"{skipped[UNSCALED]} skipped for lack of scale evidence, "
        f"{skipped[UNREADABLE]} for unreadable images; {len(ungrouped)} images in no group "
        f"of {minimum_group_size} or more; median sparse depth error "
        f"{np.median(errors) if errors else float('nan'):.3f})"
    )
    if args.debug_dir is not None:
        print(f"Depth overlays written to {args.debug_dir}")


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
