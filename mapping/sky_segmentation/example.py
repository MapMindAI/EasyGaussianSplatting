#!/usr/bin/env python3
"""Segment sky in a few images and write probability maps and overlays.

    python3 -m mapping.sky_segmentation.example output/ a.jpg b.jpg

Writes "<stem>_sky.png" (the sky probability as 0-255) and "<stem>_overlay.jpg"
(segment_people.py's overlay of the pixels above --threshold) per image, where
<stem> prefixes the parent folder, such as a cube face.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from mapping.sky_segmentation.sky_segmenter import DEFAULT_MODEL_PATH, SkySegmenter
from mapping.triton.segment_people import SKY_THRESHOLD, mask_overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=SKY_THRESHOLD, help="sky probability above which a pixel is sky")
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
        cv2.imwrite(str(args.output_dir / f"{stem}_overlay.jpg"), mask_overlay(image, np.zeros_like(sky), sky))
        print(f"{path}: {sky.mean():.1%} sky, {seconds:.2f} s")


if __name__ == "__main__":
    main()
