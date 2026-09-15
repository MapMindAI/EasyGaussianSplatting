#!/usr/bin/env python3
"""Segment people and sky in a COLMAP image folder and write gsplat training masks.

Masks are white where gsplat should supervise training and black over people
and, unless --no-mask-sky is given, the sky. People come from torchvision's
COCO-trained Mask R-CNN. Sky comes from the ADE20K SegFormer model in
third_party/EasyTensorRT over gRPC only when sky masking is enabled. Named
"<image_name>.png" after the image they belong to (COLMAP's mask naming
convention). Nested image folders are mirrored under the mask directory.

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
ADE20K_NUM_CLASSES = 150
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


def label_colour_legend(columns=10):
    """An ADE20K label-index palette matching `label_overlay`."""
    cell_width, cell_height, header_height = 100, 28, 30
    rows = (ADE20K_NUM_CLASSES + columns - 1) // columns
    legend = np.full((header_height + rows * cell_height, columns * cell_width, 3), 255, np.uint8)
    cv2.putText(
        legend,
        "ADE20K label colours",
        (4, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    for label in range(ADE20K_NUM_CLASSES):
        row, column = divmod(label, columns)
        x, y = column * cell_width, header_height + row * cell_height
        cv2.rectangle(legend, (x + 4, y + 4), (x + 24, y + 24), CLASS_COLOURS[label].tolist(), -1)
        cv2.putText(
            legend,
            str(label),
            (x + 30, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return legend


def person_overlay(image, people, alpha=0.5):
    """`image` with detected people washed red."""
    colours = np.zeros_like(image)
    colours[people] = (0, 0, 255)
    return cv2.addWeighted(image, 1.0 - alpha, colours, alpha, 0.0)


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

    triton_url = resolve_endpoint(args.triton_url) if args.mask_sky else None
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
        if args.mask_sky:
            cv2.imwrite(str(args.debug_dir / "label_colours.png"), label_colour_legend())
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
    from mapping.triton.people import PersonSegmenter

    person_segmenter = PersonSegmenter()
    segmenter = None
    if args.mask_sky:
        from mapping.triton.clients import ADE20K_SKY_CLASS, SemanticSegmenter

        segmenter = SemanticSegmenter(triton_url)

    def segment(image_path):
        """Writes one image's mask; returns whether it blocked anything and had sky."""
        mask_path, debug_path = output_paths(image_path)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}", flush=True)
            return False, False

        people = dilate(person_segmenter.mask(image), args.dilation)
        classes = None
        sky = np.zeros(image.shape[:2], dtype=bool)
        if segmenter is not None:
            classes = segmenter.classes(image)
            sky = classes == ADE20K_SKY_CLASS
        blocked = people | sky
        cv2.imwrite(str(mask_path), training_mask(blocked))
        if debug_path is not None:
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_image = person_overlay(image, people)
            if classes is not None:
                debug_image = person_overlay(label_overlay(image, classes), people)
            cv2.imwrite(str(debug_path), debug_image)
        return bool(blocked.any()), bool(sky.any())

    masked_images = 0
    sky_images = 0
    for blocked_any, sky_any in map_with_progress(
        segment, pending_image_paths, "Segmented", args.num_threads, _LOG_EVERY
    ):
        masked_images += blocked_any
        sky_images += sky_any

    sky_note = "" if args.mask_sky else ", SegFormer skipped"
    print(
        f"Masks written to {args.mask_dir} "
        f"({masked_images} images masked, {sky_images} contain sky{sky_note})"
    )
    if args.debug_dir is not None:
        palette_note = " and palette" if args.mask_sky else ""
        print(f"Label overlays{palette_note} written to {args.debug_dir}")


if __name__ == "__main__":
    main()
