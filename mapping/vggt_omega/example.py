#!/usr/bin/env python3
"""Predict VGGT-Omega depth for a few images and write maps and overlays.

    python3 -m mapping.vggt_omega.example output/ a.jpg b.jpg c.jpg

Writes "<stem>_depth.npy" (the raw (H, W) float32 depth, in the model's
arbitrary scale) and "<stem>_depth.png" (a colourized overlay) per image.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from mapping.vggt_omega.depth_estimator import DEFAULT_MODEL_PATH, VggtOmegaDepthEstimator
from mapping.triton.generate_depth import depth_overlay


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
    for path, image, depth, confidence in zip(
        args.images, images, result["depth_list"], result["depth_conf_list"]
    ):
        np.save(args.output_dir / f"{path.stem}_depth.npy", depth)
        cv2.imwrite(str(args.output_dir / f"{path.stem}_depth.png"), depth_overlay(image, depth))
        print(
            f"{path.name}: depth {np.percentile(depth, 2):.3f}..{np.percentile(depth, 98):.3f}, "
            f"median confidence {np.median(confidence):.2f}"
        )
    print(f"Wrote {len(images)} depth maps to {args.output_dir}")


if __name__ == "__main__":
    main()
