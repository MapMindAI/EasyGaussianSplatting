#!/usr/bin/env python3
"""Predict VGGT-Omega depth for a few images and write maps, overlays, and a cloud.

    python3 -m mapping.vggt_omega.example output/ a.jpg b.jpg c.jpg

Writes "<stem>_depth.npy" (the raw float32 depth at the model's resolution, in
its arbitrary scale), "<stem>_confidence.npy" (its float32 confidence, at least
0.2), and "<stem>_depth.png" (a colourized overlay at that size) per image, and
"points.ply": every image's confident pixels lifted through its predicted
camera into the model's shared world frame, coloured from the image.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from mapping.depth.generate_depth import depth_overlay
from mapping.depth.point_cloud import unproject, write_point_cloud
from mapping.vggt_omega.depth_estimator import DEFAULT_MODEL_PATH, VggtOmegaDepthEstimator

# Pixels below this percentile of their image's confidence stay out of the
# cloud, as in VGGT's own demo.
CONFIDENCE_PERCENTILE = 50
# Pixels sampled along an image's longer side: about the model's own
# resolution, where full-size images would give millions of points each.
POINTS_PER_SIDE = 416


def cloud_pixels(confidence):
    """(P, 2) integer (x, y) pixels on a grid, kept where confidence is high."""
    height, width = confidence.shape
    stride = max(1, max(height, width) // POINTS_PER_SIDE)
    rows, columns = np.mgrid[0:height:stride, 0:width:stride].reshape(2, -1)
    sampled = confidence[rows, columns]
    keep = sampled >= np.percentile(sampled, CONFIDENCE_PERCENTILE)
    return np.stack([columns[keep], rows[keep]], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("images", type=Path, nargs="+", help="views of one scene")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    images = []
    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"Cannot read {path}")
        images.append(image)

    result = VggtOmegaDepthEstimator(args.model).run(images)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_points, all_colours = [], []
    for path, image, depth, confidence, camera_from_world, intrinsics in zip(
        args.images,
        images,
        result["depth_list"],
        result["depth_conf_list"],
        result["camera_from_world_list"],
        result["intrinsics_list"],
    ):
        # The maps are at the model's resolution; colour them from the image at that size.
        image = cv2.resize(image, depth.shape[::-1], interpolation=cv2.INTER_AREA)
        np.save(args.output_dir / f"{path.stem}_depth.npy", depth)
        np.save(args.output_dir / f"{path.stem}_confidence.npy", confidence)
        cv2.imwrite(str(args.output_dir / f"{path.stem}_depth.png"), depth_overlay(image, depth))
        print(
            f"{path.name}: depth {np.percentile(depth, 2):.3f}..{np.percentile(depth, 98):.3f}, "
            "confidence p5/p50/p95 "
            + "/".join(f"{value:.2f}" for value in np.percentile(confidence, (5, 50, 95)))
        )
        pixels = cloud_pixels(confidence)
        depths = depth[pixels[:, 1], pixels[:, 0]]
        all_points.append(unproject(pixels, depths, intrinsics, camera_from_world))
        all_colours.append(image[pixels[:, 1], pixels[:, 0], ::-1])

    cloud_path = args.output_dir / "points.ply"
    points = np.concatenate(all_points)
    write_point_cloud(cloud_path, points, np.concatenate(all_colours))
    print(f"Wrote {len(images)} depth maps and {len(points)} points ({cloud_path}) to {args.output_dir}")


if __name__ == "__main__":
    main()
