#!/usr/bin/env python3
"""Segment people and sky in a COLMAP image folder and write gsplat training masks.

Masks are white where gsplat should supervise training and black over enabled
classes. --mask-people and --mask-sky independently enable torchvision's
COCO-trained Mask R-CNN and the U²-Net sky model at models/skyseg.onnx (see
mapping/sky_segmentation/), both run locally. Both default to enabled. Named
"<image_name>.png" after the image they belong to (COLMAP's mask naming
convention). Nested image folders are mirrored under the mask directory.

--debug-dir additionally writes, per image, people in red and sky in magenta
washed over the image they came from.

Inference runs on --num-threads worker threads.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
# Sky probability above which a pixel is masked.
SKY_THRESHOLD = 0.5
# Images per progress log line.
_LOG_EVERY = 200


def dilate(mask, radius):
    """Grow a bool mask by `radius` pixels."""
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def training_mask(blocked):
    """255 where training should look, 0 over the blocked (person or sky) pixels."""
    return np.where(blocked, np.uint8(0), np.uint8(255))


def mask_overlay(image, people, sky, alpha=0.5):
    """`image` with detected people washed red and sky magenta."""
    colours = np.zeros_like(image)
    colours[sky] = (255, 0, 255)
    colours[people] = (0, 0, 255)
    return cv2.addWeighted(image, 1.0 - alpha, colours, alpha, 0.0)


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("mask_dir", type=Path)
    parser.add_argument(
        "--dilation", type=int, default=1, help="pixels to grow each person mask by"
    )
    parser.add_argument(
        "--mask-people",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mask people with Mask R-CNN",
    )
    parser.add_argument(
        "--mask-sky",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mask sky with U²-Net",
    )
    parser.add_argument(
        "--num-threads", type=int, default=4, help="worker threads running inference"
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="also write the masked people and sky overlaid on each image here",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(arguments)

    if not (args.mask_people or args.mask_sky):
        print("People and sky masking disabled; skipping segmentation")
        return

    image_paths = sorted(
        path
        for path in args.image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise SystemExit(f"No images found in {args.image_dir}")

    def output_paths(image_path):
        """The mask and, when debugging, the overlay this image writes."""
        relative = f"{image_path.relative_to(args.image_dir)}.png"
        debug_path = None if args.debug_dir is None else args.debug_dir / relative
        return args.mask_dir / relative, debug_path

    args.mask_dir.mkdir(parents=True, exist_ok=True)
    if args.debug_dir is not None:
        args.debug_dir.mkdir(parents=True, exist_ok=True)
    pending_image_paths = [
        image_path
        for image_path in image_paths
        if args.overwrite
        or not all(path.exists() for path in output_paths(image_path) if path)
    ]
    if not pending_image_paths:
        print(f"Masks already exist in {args.mask_dir}; skipping segmentation")
        return

    # Deferred so a run with nothing to do never loads either model.
    from mapping.triton.progress import map_with_progress

    person_segmenter = None
    if args.mask_people:
        from mapping.triton.people import PersonSegmenter

        person_segmenter = PersonSegmenter()
    sky_segmenter = None
    if args.mask_sky:
        from mapping.sky_segmentation.sky_segmenter import SkySegmenter

        sky_segmenter = SkySegmenter()

    def segment(image_path):
        """Writes one image's mask; returns whether it blocked anything and had sky."""
        mask_path, debug_path = output_paths(image_path)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}", flush=True)
            return False, False

        people = np.zeros(image.shape[:2], dtype=bool)
        if person_segmenter is not None:
            people = dilate(person_segmenter.mask(image), args.dilation)
        sky = np.zeros(image.shape[:2], dtype=bool)
        if sky_segmenter is not None:
            sky = sky_segmenter.probability(image) > SKY_THRESHOLD
        blocked = people | sky
        cv2.imwrite(str(mask_path), training_mask(blocked))
        if debug_path is not None:
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_path), mask_overlay(image, people, sky))
        return bool(blocked.any()), bool(sky.any())

    masked_images = 0
    sky_images = 0
    for blocked_any, sky_any in map_with_progress(
        segment, pending_image_paths, "Segmented", args.num_threads, _LOG_EVERY
    ):
        masked_images += blocked_any
        sky_images += sky_any

    sky_note = "" if args.mask_sky else ", sky skipped"
    print(
        f"Masks written to {args.mask_dir} "
        f"({masked_images} images masked, {sky_images} contain sky{sky_note})"
    )
    if args.debug_dir is not None:
        print(f"Overlays written to {args.debug_dir}")


if __name__ == "__main__":
    main()
