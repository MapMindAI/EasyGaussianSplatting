#!/usr/bin/env python3
"""Segment people in a COLMAP image folder and write gsplat training masks.

Masks are white where gsplat should supervise training and black over people,
named "<image_name>.png" after the image they belong to (COLMAP's mask naming
convention). Nested image folders are mirrored under the mask directory.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.transforms.functional import to_tensor

PERSON_LABEL = 1  # COCO
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def person_mask(model, image, device, score_threshold, mask_threshold, dilation):
    """Return a uint8 mask: 255 where training should look, 0 over people."""
    tensor = to_tensor(image).to(device)
    with torch.inference_mode():
        prediction = model([tensor])[0]

    people = (prediction["labels"] == PERSON_LABEL) & (
        prediction["scores"] >= score_threshold
    )
    height, width = tensor.shape[1:]
    if not bool(people.any()):
        return np.full((height, width), 255, dtype=np.uint8)

    person = (prediction["masks"][people, 0].amax(0) >= mask_threshold).float()
    if dilation > 0:
        # Grows the mask over the soft edges the segmenter leaves behind:
        # motion blur, hair, and the contact shadow under the feet.
        person = torch.nn.functional.max_pool2d(
            person[None], kernel_size=2 * dilation + 1, stride=1, padding=dilation
        )[0]
    return ((person < 0.5).to(torch.uint8) * 255).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("mask_dir", type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--dilation", type=int, default=8, help="pixels to grow each person mask by"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights).eval().to(device)

    masked_images = 0
    for count, image_path in enumerate(pending_image_paths, start=1):
        mask_path = args.mask_dir / f"{image_path.relative_to(args.image_dir)}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(image_path).convert("RGB")
        mask = person_mask(
            model,
            image,
            device,
            args.score_threshold,
            args.mask_threshold,
            args.dilation,
        )
        Image.fromarray(mask).save(mask_path)
        masked_images += int((mask == 0).any())
        print(f"[{count}/{len(pending_image_paths)}] {mask_path}", flush=True)

    print(f"Masks written to {args.mask_dir} ({masked_images} images contain people)")


if __name__ == "__main__":
    main()
