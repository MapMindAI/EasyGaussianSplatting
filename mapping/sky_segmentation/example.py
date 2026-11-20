#!/usr/bin/env python3
"""Segment sky in a few images and write probability maps and overlays.

    python3 -m mapping.sky_segmentation.example output/ a.jpg b.jpg

Writes "<stem>_sky.png" (the sky probability as 0-255) and "<stem>_overlay.jpg"
(the image with pixels above --threshold tinted magenta) per image, where <stem>
prefixes the parent folder, such as a cube face.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from mapping.sky_segmentation.sky_segmenter import DEFAULT_MODEL_PATH, SkySegmenter

SKY_TINT = np.array([255, 0, 255], dtype=np.float32)


def sky_overlay(image, sky, alpha=0.5):
    """`image` with the `sky` pixels washed in magenta."""
    overlay = image.astype(np.float32)
    overlay[sky] = (1 - alpha) * overlay[sky] + alpha * SKY_TINT
    return overlay.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=0.5, help="sky probability above which a pixel is sky")
    args = parser.parse_args()

    segmenter = SkySegmenter(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"Cannot read {path}")
        start = time.perf_counter()
        probability = segmenter.probability(image)
        seconds = time.perf_counter() - start
        sky = probability > args.threshold
        stem = f"{path.parent.name}_{path.stem}"
        cv2.imwrite(str(args.output_dir / f"{stem}_sky.png"), np.round(probability * 255).astype(np.uint8))
        cv2.imwrite(str(args.output_dir / f"{stem}_overlay.jpg"), sky_overlay(image, sky))
        print(f"{path}: {sky.mean():.1%} sky, {seconds:.2f} s")


if __name__ == "__main__":
    main()
