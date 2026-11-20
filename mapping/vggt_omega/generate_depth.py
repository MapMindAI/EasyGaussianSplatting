#!/usr/bin/env python3
"""Predict per-image depth for a COLMAP model with VGGT-Omega, run locally.

The same stage as mapping/depth/generate_depth.py -- covisibility grouping,
scale fitting to COLMAP, filtering, and the (x, y, depth) output -- with the
ONNX model at --model in place of the Triton DA3 one.
"""

import argparse
from pathlib import Path

from mapping.depth.generate_depth import generate_depths, parse_depth_arguments
from mapping.vggt_omega.depth_estimator import DEFAULT_MODEL_PATH, VggtOmegaDepthEstimator

GROUP_SIZE = 12
# Two views already give the model a baseline to triangulate.
MINIMUM_GROUP_SIZE = 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH, help="VGGT-Omega ONNX model"
    )
    # Two workers share the one session: one infers while the other reads and
    # filters. More would only queue on the GPU.
    parser.set_defaults(num_threads=2)
    args = parse_depth_arguments(parser)
    generate_depths(
        args, lambda: VggtOmegaDepthEstimator(args.model), GROUP_SIZE, MINIMUM_GROUP_SIZE
    )


if __name__ == "__main__":
    main()
