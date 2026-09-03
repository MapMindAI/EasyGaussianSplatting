#!/usr/bin/env python3
"""Run gsplat's simple_trainer with the per-image masks segment_people.py writes.

gsplat's COLMAP dataset only carries one mask per camera (the undistortion
ROI), so per-image masks are attached here instead: its __getitem__ is wrapped
to load "<data_dir>/masks/<image_name>.png" alongside each image. Takes
simple_trainer's own arguments.
"""
import os
import runpy
import sys

import numpy as np
import torch
from PIL import Image

EXAMPLES_DIR = "/opt/gsplat/examples"


def attach_masks(dataset_class):
    original_getitem = dataset_class.__getitem__

    def __getitem__(self, item):
        data = original_getitem(self, item)
        image_name = self.parser.image_names[self.indices[item]]
        mask_path = os.path.join(self.parser.data_dir, "masks", f"{image_name}.png")
        if not os.path.exists(mask_path):
            return data

        assert self.patch_size is None, "masks are not cropped along with patches"
        height, width = data["image"].shape[:2]
        mask = Image.open(mask_path).resize((width, height), Image.NEAREST)
        mask = torch.from_numpy(np.asarray(mask) > 127)
        # An all-black mask leaves the losses averaging over no pixels at all,
        # which trains on NaN from there on.
        if mask.any():
            data["mask"] = mask
        return data

    dataset_class.__getitem__ = __getitem__


sys.path.insert(0, EXAMPLES_DIR)
from datasets.colmap import Dataset  # noqa: E402

attach_masks(Dataset)
runpy.run_path(os.path.join(EXAMPLES_DIR, "simple_trainer.py"), run_name="__main__")
