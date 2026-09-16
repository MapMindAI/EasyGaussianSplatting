"""Paths and validation shared by the TSDF stages."""
from pathlib import Path
import re

import numpy as np


def latest_point_cloud(model_path):
    """Return the most recently named gsplat PLY under a model directory."""
    def step(path):
        numbers = re.findall(r"\d+", path.stem)
        return int(numbers[-1]) if numbers else -1

    candidates = sorted((Path(model_path) / "ply").glob("*.ply"), key=step)
    if not candidates:
        raise FileNotFoundError(f"No gsplat PLY exists under {model_path / 'ply'}")
    return candidates[-1]


def depth_path(depth_directory, image_name):
    """Return the depth file preserving an image's nested cube-face path."""
    return Path(depth_directory) / Path(image_name).with_suffix(".npy")


def valid_depth(depth, alpha, minimum_alpha):
    """Keep finite, positive expected-hit distances with enough coverage."""
    return np.isfinite(depth) & (depth > 0) & (alpha >= minimum_alpha)
