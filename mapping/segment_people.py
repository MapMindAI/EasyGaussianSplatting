#!/usr/bin/env python3
"""Segment people and sky in a COLMAP image folder and write gsplat training masks.

Masks are white where gsplat should supervise training and black over people and
sky, both read from the ADE20K SegFormer model in third_party/EasyTensorRT over
gRPC. Named "<image_name>.png" after the image they belong to (COLMAP's mask
naming convention). Nested image folders are mirrored under the mask directory.

The model is reached at --triton-url, or $TRITON_URL when the flag is left out.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
LOG_EVERY = 200


def dilate(mask, radius):
    """Grow a bool mask by `radius` pixels."""
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def training_mask(blocked):
    """255 where training should look, 0 over the blocked (person or sky) pixels."""
    return np.where(blocked, np.uint8(0), np.uint8(255))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("mask_dir", type=Path)
    parser.add_argument(
        "--triton-url",
        default=None,
        help="Triton gRPC endpoint for the segmentation model; defaults to $TRITON_URL",
    )
    parser.add_argument(
        "--dilation", type=int, default=1, help="pixels to grow each person mask by"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    triton_url = args.triton_url or os.environ.get("TRITON_URL")
    if not triton_url:
        raise SystemExit("No Triton URL given; pass --triton-url or set TRITON_URL")
    # Imported here so the mask helpers stay testable without the EasyTensorRT
    # submodule or the Triton client installed.
    from mapping.features.triton_models import (
        ADE20K_PERSON_CLASS,
        ADE20K_SKY_CLASS,
        SemanticSegmenter,
    )

    segmenter = SemanticSegmenter(triton_url)

    image_paths = sorted(
        path
        for path in args.image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise SystemExit(f"No images found in {args.image_dir}")

    args.mask_dir.mkdir(parents=True, exist_ok=True)
    pending_image_paths = [
        image_path
        for image_path in image_paths
        if args.overwrite
        or not (
            args.mask_dir
            / f"{image_path.relative_to(args.image_dir)}.png"
        ).exists()
    ]
    if not pending_image_paths:
        print(f"Masks already exist in {args.mask_dir}; skipping segmentation")
        return

    masked_images = 0
    sky_images = 0
    for count, image_path in enumerate(pending_image_paths, start=1):
        mask_path = args.mask_dir / f"{image_path.relative_to(args.image_dir)}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        classes = segmenter.classes(image)
        sky = classes == ADE20K_SKY_CLASS
        # The semantic person class leaves soft edges (motion blur, hair, the
        # contact shadow) the way the instance masks did, so grow it a little.
        people = dilate(classes == ADE20K_PERSON_CLASS, args.dilation)
        cv2.imwrite(str(mask_path), training_mask(sky | people))
        sky_images += int(sky.any())
        masked_images += int((sky | people).any())
        if count % LOG_EVERY == 0 or count == len(pending_image_paths):
            print(f"Segmented {count}/{len(pending_image_paths)} images", flush=True)

    print(
        f"Masks written to {args.mask_dir} "
        f"({masked_images} images masked, {sky_images} contain sky)"
    )


if __name__ == "__main__":
    main()
