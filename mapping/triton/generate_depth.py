#!/usr/bin/env python3
"""Predict per-image depth for a COLMAP model with Depth Anything 3.

DA3 takes a fixed number of views at once, so the model's images are grouped by
camera and split into consecutive groups of GROUP_SIZE, which for the cube-map
rig means neighbouring frames of one face.

Each group is reconstructed in its own arbitrary scale, so the depth it predicts
is fitted to the scale of the COLMAP points the group's images already observe
before anything is written. Groups whose images carry too few of those points
are skipped rather than written at a guessed scale.

The output is one "<image_name>.npy" per image under --depth-dir, holding the
(M, 3) float32 columns (x, y, depth) that the trainer supervises on: depth in
the frame of the input model, and x and y normalized to [0, 1] so the file
survives whatever resolution the trainer loads the images at. Sampling only the
supervised pixels keeps the stage's output megabytes rather than gigabytes.

The model is reached at --triton-url, or $TRITON_URL when the flag is left out.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from mapping.triton.endpoint import add_endpoint_argument, resolve_endpoint

# Views per DA3 request. The served model's input shape is fixed, so this has
# to match the deployed model (see third_party/EasyTensorRT).
GROUP_SIZE = 5
# Groups per progress log line.
_LOG_EVERY = 20
# A group's scale is only as trustworthy as the COLMAP depths behind it. The
# count is pooled over the group's images, since they share the one scale.
MIN_SCALE_POINTS = 20
# Why a group produced nothing.
UNREADABLE, UNSCALED = "unreadable", "unscaled"


def group_images(names, group_size):
    """Consecutive groups of `group_size` names.

    A trailing short group is grown backwards into the previous one instead of
    being padded: the model's view count is fixed, and repeating an image to
    fill the request would have it reconstruct a view it has already seen.
    """
    if len(names) < group_size:
        return []
    groups = [names[index:index + group_size] for index in range(0, len(names), group_size)]
    if len(groups[-1]) < group_size:
        groups[-1] = names[-group_size:]
    return groups


def fit_depth_scale(predicted, reference):
    """The scale carrying `predicted` onto `reference`, or None if unsupported.

    The median ratio rather than a least-squares fit: a learned depth map is
    wrong by orders of magnitude at object edges, and those outliers would
    otherwise drag the scale with them.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    usable = np.isfinite(predicted) & np.isfinite(reference) & (predicted > 0) & (reference > 0)
    if int(usable.sum()) < MIN_SCALE_POINTS:
        return None
    return float(np.median(reference[usable] / predicted[usable]))


def sample_at(image_values, points):
    """`image_values` read at each integer-rounded (x, y) in `points`."""
    height, width = image_values.shape[:2]
    columns = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, width - 1)
    rows = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, height - 1)
    return image_values[rows, columns]


def supervised_points(depth, confidence, mask, count, min_confidence, generator):
    """Up to `count` (x, y, depth) rows worth supervising, as float32.

    Keeps confident, positive-depth pixels that the training mask also keeps,
    then subsamples: neighbouring pixels of a smooth depth map say almost the
    same thing, so a few thousand per image carry the geometry at a fraction of
    the size. x and y come back normalized to [0, 1].
    """
    usable = np.isfinite(depth) & (depth > 0)
    if confidence is not None:
        usable &= confidence >= min_confidence
    if mask is not None:
        usable &= mask
    rows, columns = np.nonzero(usable)
    if rows.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if rows.size > count:
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
    return cv2.addWeighted(image, 1.0 - alpha, colours, alpha, 0.0)


def depth_debug_path(debug_dir, name):
    path = Path(name)
    return debug_dir / path.with_name(f"{path.stem}_depth.png")


def observed_depths(reconstruction):
    """Each image's observed COLMAP points, as name -> ((P, 2) xy, (P,) depth).

    These are the same triangulated points gsplat's own depth option supervises
    on; here they only have to pin down one scale per group.
    """
    observations = {}
    for image in reconstruction.images.values():
        camera_from_world = image.cam_from_world
        if callable(camera_from_world):
            camera_from_world = camera_from_world()
        camera_from_world = np.asarray(camera_from_world.matrix(), dtype=np.float64)
        coordinates = []
        depths = []
        for point2D in image.get_observation_points2D():
            xyz = np.asarray(reconstruction.points3D[point2D.point3D_id].xyz, dtype=np.float64)
            coordinates.append(point2D.xy)
            depths.append(camera_from_world[2, :3] @ xyz + camera_from_world[2, 3])
        observations[image.name] = (
            np.asarray(coordinates, dtype=np.float64).reshape(-1, 2),
            np.asarray(depths, dtype=np.float64).reshape(-1),
        )
    return observations


def camera_groups(reconstruction, group_size):
    """Groups of `group_size` image names, consecutive within one camera."""
    by_camera = {}
    for image in reconstruction.images.values():
        by_camera.setdefault(image.camera_id, []).append(image.name)
    groups = []
    for camera_id in sorted(by_camera):
        groups.extend(group_images(sorted(by_camera[camera_id]), group_size))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="holds images/ and sparse/")
    parser.add_argument("depth_dir", type=Path)
    add_endpoint_argument(parser, "depth model")
    parser.add_argument(
        "--samples-per-image", type=int, default=4096, help="supervised pixels kept"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=2.0, help="DA3 confidence floor"
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

    triton_url = resolve_endpoint(args.triton_url)

    # Deferred so the helpers above stay testable without pycolmap, the
    # EasyTensorRT submodule, or the Triton client installed.
    import pycolmap

    from mapping.triton.progress import map_with_progress
    from mapping.triton.clients import DepthEstimator

    images_dir = args.model_dir / "images"
    reconstruction = pycolmap.Reconstruction(args.model_dir / "sparse" / "0")
    groups = camera_groups(reconstruction, GROUP_SIZE)
    if not groups:
        raise SystemExit(
            f"No camera in {args.model_dir} has {GROUP_SIZE} images to group"
        )

    args.depth_dir.mkdir(parents=True, exist_ok=True)
    pending_groups = [
        group
        for group in groups
        if args.overwrite
        or not all(
            (args.depth_dir / f"{name}.npy").exists()
            and (
                args.debug_dir is None
                or depth_debug_path(args.debug_dir, name).exists()
            )
            for name in group
        )
    ]
    if not pending_groups:
        print(f"Depths already exist in {args.depth_dir}; skipping depth generation")
        return

    observations = observed_depths(reconstruction)
    estimator = DepthEstimator(triton_url, expected_num_images=GROUP_SIZE)

    def training_mask(name, shape):
        """The segmentation mask for `name` at `shape`, or None if there is none."""
        if args.mask_dir is None:
            return None
        mask = cv2.imread(str(args.mask_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if mask.shape[:2] != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask > 127

    def estimate(group):
        """One group's rows per image, or (None, why) when it cannot be written."""
        images = []
        for name in group:
            image = cv2.imread(str(images_dir / name))
            if image is None:
                print(f"Skipping unreadable image: {name}", flush=True)
                return None, UNREADABLE
            images.append(image)

        result = estimator.run(images)
        predicted = [
            sample_at(depth, observations[name][0])
            for name, depth in zip(group, result["depth_list"])
        ]
        scale = fit_depth_scale(
            np.concatenate(predicted),
            np.concatenate([observations[name][1] for name in group]),
        )
        if scale is None:
            return None, UNSCALED

        # Seeded per group so a rerun samples the same pixels; the groups run
        # on worker threads, which would otherwise interleave the draws.
        generator = np.random.default_rng(0)
        return [
            (
                name,
                image,
                depth * scale,
                supervised_points(
                    depth * scale,
                    confidence,
                    training_mask(name, depth.shape[:2]),
                    args.samples_per_image,
                    args.min_confidence,
                    generator,
                ),
            )
            for name, image, depth, confidence in zip(
                group, images, result["depth_list"], result["depth_conf_list"]
            )
        ], None

    # Consecutive groups overlap wherever a camera's image count is not a
    # multiple of GROUP_SIZE, so the writes stay on this thread: two workers
    # saving the same image's file would race.
    written = set()
    skipped = {UNREADABLE: 0, UNSCALED: 0}
    for rows_by_name, failure in map_with_progress(
        estimate, pending_groups, "Depth for group", args.num_threads, _LOG_EVERY
    ):
        if failure is not None:
            skipped[failure] += 1
            continue
        for name, image, depth, rows in rows_by_name:
            output_path = args.depth_dir / f"{name}.npy"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, rows)
            if args.debug_dir is not None:
                overlay_path = depth_debug_path(args.debug_dir, name)
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(overlay_path), depth_overlay(image, depth))
            written.add(name)

    print(
        f"Depths written to {args.depth_dir} ({len(written)} images; "
        f"{skipped[UNSCALED]} groups skipped for lack of points, "
        f"{skipped[UNREADABLE]} for unreadable images)"
    )
    if args.debug_dir is not None:
        print(f"Depth overlays written to {args.debug_dir}")


if __name__ == "__main__":
    main()
