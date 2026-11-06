#!/usr/bin/env python3
"""Split coarse Gaussians among training blocks, and merge the trained blocks.

A Gaussian belongs to the block owning the SfM point nearest it in XY, from
the `<method>_points.npz` block_partition.py writes beside its manifest. A
block trains the coarse Gaussians in its expanded region and keeps only the
ones it owns at merge, where the blocks are concatenated.

Usage: python3 -m mapping.block_gaussians <manifest.json> <blocks_dir> <output_dir>
"""
import argparse
import json
import os
import re

import numpy as np
from scipy import spatial

# Share of a block's extent its trainable region reaches past it on each side.
EXPANSION = 0.1


def load_block(manifest_path, name):
    """The block's manifest entry and index, and every SfM point with its owner."""
    with open(manifest_path) as file:
        manifest = json.load(file)
    index = [block["name"] for block in manifest["blocks"]].index(name)
    owned = np.load(os.path.join(os.path.dirname(manifest_path), manifest["points"]))
    return manifest["blocks"][index], index, owned["points"], owned["owners"]


def to_colmap(transform, means):
    """Means from gsplat's normalized frame to the COLMAP frame."""
    inverse = np.linalg.inv(transform)
    return means @ inverse[:3, :3].T + inverse[:3, 3]


def nearest_owners(positions, points, owners):
    _, nearest = spatial.cKDTree(points[:, :2]).query(positions[:, :2])
    return owners[nearest]


def expanded_region(positions, points, owners, block):
    """Positions the block owns, or within EXPANSION of its middle-90% extent of its points."""
    own = points[owners == block, :2]
    low, high = np.percentile(own, [5, 95], axis=0)
    distance, _ = spatial.cKDTree(own).query(positions[:, :2])
    return (distance <= EXPANSION * np.max(high - low)) | (nearest_owners(positions, points, owners) == block)


def latest_checkpoint(result_dir):
    directory = os.path.join(result_dir, "ckpts")
    steps = {int(re.fullmatch(r"ckpt_(\d+)_rank0\.pt", name).group(1)): name
             for name in os.listdir(directory) if re.fullmatch(r"ckpt_(\d+)_rank0\.pt", name)}
    return os.path.join(directory, steps[max(steps)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", help="block_partition.py's <method>.json")
    parser.add_argument("blocks_dir", help="One trained result directory per block, by name.")
    parser.add_argument("output_dir")
    arguments = parser.parse_args()

    import torch
    from gsplat import export_splats

    from mapping.gsplat_world_frame import restore_colmap_coordinates

    with open(arguments.manifest) as file:
        names = [block["name"] for block in json.load(file)["blocks"]]
    merged, step, transform = {}, 0, None
    for index, name in enumerate(names):
        _, _, points, owners = load_block(arguments.manifest, name)
        result_dir = os.path.join(arguments.blocks_dir, name)
        checkpoint = torch.load(latest_checkpoint(result_dir), map_location="cpu", weights_only=True)
        transform = np.load(os.path.join(result_dir, "scene_transform.npy"))
        means = to_colmap(transform, checkpoint["splats"]["means"].numpy())
        keep = torch.from_numpy(nearest_owners(means, points, owners) == index)
        print(f"{name}: keeps {int(keep.sum())} of {len(keep)} Gaussians", flush=True)
        for key, value in checkpoint["splats"].items():
            merged.setdefault(key, []).append(value[keep])
        step = max(step, checkpoint["step"])
    merged = {key: torch.cat(values) for key, values in merged.items()}

    os.makedirs(os.path.join(arguments.output_dir, "ckpts"), exist_ok=True)
    os.makedirs(os.path.join(arguments.output_dir, "ply"), exist_ok=True)
    torch.save({"step": step, "splats": merged}, os.path.join(arguments.output_dir, "ckpts", "ckpt_merged.pt"))
    means, scales, quats = restore_colmap_coordinates(
        transform, merged["means"].numpy(), merged["scales"].numpy(), merged["quats"].numpy()
    )
    export_splats(
        means=torch.from_numpy(means).float(),
        scales=torch.from_numpy(scales).float(),
        quats=torch.from_numpy(quats).float(),
        opacities=merged["opacities"],
        sh0=merged["sh0"],
        shN=merged["shN"],
        format="ply",
        save_to=os.path.join(arguments.output_dir, "ply", "point_cloud_merged.ply"),
    )
    print(f"merged {len(merged['means'])} Gaussians from {len(names)} blocks", flush=True)


if __name__ == "__main__":
    main()
