#!/usr/bin/env python3
"""Segment people and sky in a COLMAP image folder and write gsplat training masks.

Masks are white where gsplat should supervise training and black over people
and, unless --no-mask-sky is given, the sky; both are read from the ADE20K
SegFormer model in third_party/EasyTensorRT over gRPC. Named "<image_name>.png"
after the image they belong to (COLMAP's mask naming convention). Nested image
folders are mirrored under the mask directory.

--debug-dir additionally writes, per image, the model's whole label map washed
over the image it came from, which is what shows why a pixel was masked or
missed.

The model is reached at --triton-url, or $TRITON_URL when the flag is left out,
with inference on --num-threads worker threads.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from mapping.triton.endpoint import add_endpoint_argument, resolve_endpoint

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
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


def _class_colours():
    """One vivid, fixed colour per class index.

    The hue strides by a step coprime with OpenCV's 180-step circle, so labels
    that show up side by side -- sky 2, tree 4, person 12 -- land far apart on
    it instead of in one wash of green. Saturation and value alternate to keep
    the indices that share a hue apart once the circle wraps.
    """
    indices = np.arange(256)
    hue = (indices * 97 % 180).astype(np.uint8)
    saturation = np.where(indices % 2, 165, 255).astype(np.uint8)
    value = np.where(indices % 4 < 2, 255, 195).astype(np.uint8)
    hsv = np.stack([hue, saturation, value], axis=-1)[np.newaxis]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]


CLASS_COLOURS = _class_colours()


def label_overlay(image, classes, alpha=0.5):
    """`image` with every segmented label washed over it in its own colour."""
    return cv2.addWeighted(image, 1.0 - alpha, CLASS_COLOURS[classes], alpha, 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("mask_dir", type=Path)
    add_endpoint_argument(parser, "segmentation model")
    parser.add_argument(
        "--dilation", type=int, default=1, help="pixels to grow each person mask by"
    )
    parser.add_argument(
        "--mask-sky",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mask the sky as well as the people",
    )
    parser.add_argument(
        "--num-threads", type=int, default=4, help="worker threads running inference"
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="also write the whole label map overlaid on each image here",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    triton_url = resolve_endpoint(args.triton_url)
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
    pending_image_paths = [
        image_path
        for image_path in image_paths
        if args.overwrite
        or not all(path.exists() for path in output_paths(image_path) if path)
    ]
    if not pending_image_paths:
        print(f"Masks already exist in {args.mask_dir}; skipping segmentation")
        return

    # Deferred so the mask helpers stay testable without the EasyTensorRT
    # submodule, the Triton client, or pycolmap installed, and so a run with
    # nothing to do never reaches for Triton.
    from mapping.triton.progress import map_with_progress
    from mapping.triton.clients import (
        ADE20K_PERSON_CLASS,
        ADE20K_SKY_CLASS,
        SemanticSegmenter,
    )

    segmenter = SemanticSegmenter(triton_url)

    def segment(image_path):
        """Writes one image's mask; returns whether it blocked anything and had sky."""
        mask_path, debug_path = output_paths(image_path)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}", flush=True)
            return False, False

        classes = segmenter.classes(image)
        # The semantic person class leaves soft edges (motion blur, hair, the
        # contact shadow) the way the instance masks did, so grow it a little.
        blocked = dilate(classes == ADE20K_PERSON_CLASS, args.dilation)
        sky = classes == ADE20K_SKY_CLASS
        if args.mask_sky:
            blocked |= sky
        cv2.imwrite(str(mask_path), training_mask(blocked))
        if debug_path is not None:
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_path), label_overlay(image, classes))
        return bool(blocked.any()), bool(sky.any())

    masked_images = 0
    sky_images = 0
    for blocked_any, sky_any in map_with_progress(
        segment, pending_image_paths, "Segmented", args.num_threads, _LOG_EVERY
    ):
        masked_images += blocked_any
        sky_images += sky_any

    sky_note = "" if args.mask_sky else ", left unmasked"
    print(
        f"Masks written to {args.mask_dir} "
        f"({masked_images} images masked, {sky_images} contain sky{sky_note})"
    )
    if args.debug_dir is not None:
        print(f"Label overlays written to {args.debug_dir}")


if __name__ == "__main__":
    main()
