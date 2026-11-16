"""Scheduled significance pruning during single-GPU MCMC training."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from gsplat_server.proto import gsplat_pb2

MAXIMUM_REGION_HITS = 1_000_000


def validate_pruning(parameters):
    steps = list(parameters.mcmc_prune_steps)
    if not steps:
        return
    if parameters.strategy != gsplat_pb2.STRATEGY_MCMC:
        raise ValueError("mcmc_prune_steps requires STRATEGY_MCMC")
    if steps != sorted(set(steps)) or not all(0 < step < parameters.iterations for step in steps):
        raise ValueError("mcmc_prune_steps must be increasing, unique, and before the final iteration")
    if not 0 < parameters.mcmc_prune_keep_fraction < 1:
        raise ValueError("mcmc_prune_keep_fraction must be between 0 and 1")
    if parameters.mcmc_prune_scoring_views < 1 or parameters.mcmc_prune_resolution < 1:
        raise ValueError("pruning scoring views and resolution must be positive")


def removal_mask(scores, keep_fraction):
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("pruning requires finite, nonnegative scores for a nonempty model")
    if not (scores > 0).any():
        raise ValueError("all pruning scores are zero; check scoring cameras and model visibility")
    retained = max(1, round(len(scores) * keep_fraction))
    mask = np.ones(len(scores), dtype=bool)
    mask[np.argsort(-scores, kind="stable")[:retained]] = False
    return mask


def validate_pruning_configuration(configuration, steps, world_size):
    if world_size != 1 or configuration.patch_size is not None:
        raise ValueError("significance pruning requires one GPU and full training images")
    if any(step >= configuration.max_steps for step in steps):
        raise ValueError("pruning steps must leave an optimization step before max_steps")
    if configuration.with_eval3d or configuration.with_ut or configuration.camera_model != "pinhole":
        raise ValueError("significance pruning currently requires standard pinhole rasterization")


def load_training_mask(data_directory, image_name, size):
    path = Path(data_directory) / "masks" / f"{image_name}.png"
    if not path.exists():
        return None
    with Image.open(path) as image:
        mask = np.asarray(image.resize(size, Image.Resampling.NEAREST)) > 127
    # Empty per-image masks would make the training loss NaN.
    return mask if mask.any() else None


def score_gaussians(runner, parameters):
    import torch
    from gsplat.cuda._wrapper import (
        rasterize_num_contributing_gaussians,
        rasterize_to_indices_in_range,
    )
    from gsplat.rendering import rasterization

    splats, configuration = runner.splats, runner.cfg
    device = splats["means"].device
    count = len(splats["means"])
    hits = torch.zeros(count, dtype=torch.int64, device=device)
    with torch.no_grad():
        opacity = splats["opacities"].sigmoid().flatten()
        scales = splats["scales"].exp()

    def score_region(view_matrix, intrinsics, width, height, mask):
        _, _, metadata = rasterization(
            means=splats["means"], quats=splats["quats"], scales=scales,
            opacities=opacity, colors=None, render_mode="D",
            viewmats=view_matrix, Ks=intrinsics,
            width=width, height=height, packed=configuration.packed,
            near_plane=configuration.near_plane, far_plane=configuration.far_plane,
            camera_model=configuration.camera_model,
            rasterize_mode="antialiased" if configuration.antialiased else "classic",
        )
        counts, alphas = rasterize_num_contributing_gaussians(
            metadata["means2d"], metadata["conics"], metadata["opacities"],
            metadata["isect_offsets"], metadata["flatten_ids"],
            width, height, metadata["tile_size"],
        )
        # Bound the three returned index arrays to about 24 MB per region.
        if counts.sum().item() > MAXIMUM_REGION_HITS:
            if width == 1 and height == 1:
                raise ValueError("too many contributors at one scoring pixel; reduce the Gaussian budget")
            del metadata, counts, alphas
            for top, bottom in ((0, height // 2), (height // 2, height)):
                for left, right in ((0, width // 2), (width // 2, width)):
                    if bottom == top or right == left:
                        continue
                    cropped = intrinsics.clone()
                    cropped[:, 0, 2] -= left
                    cropped[:, 1, 2] -= top
                    score_region(view_matrix, cropped, right - left, bottom - top,
                                 None if mask is None else mask[top:bottom, left:right])
            return
        # With one camera, packed projected IDs form a dense one-camera array.
        gaussian_ids, pixel_ids, _ = rasterize_to_indices_in_range(
            0, 2**30, torch.ones_like(alphas),
            metadata["means2d"].reshape(1, -1, 2),
            metadata["conics"].reshape(1, -1, 3),
            metadata["opacities"].reshape(1, -1),
            width, height, metadata["tile_size"],
            metadata["isect_offsets"], metadata["flatten_ids"],
        )
        if mask is not None:
            gaussian_ids = gaussian_ids[mask.reshape(-1)[pixel_ids]]
        if configuration.packed:
            gaussian_ids = metadata["gaussian_ids"][gaussian_ids]
        hits.add_(torch.bincount(gaussian_ids.long(), minlength=count))

    dataset = runner.trainset
    parser = dataset.parser
    selected = np.linspace(0, len(dataset) - 1,
                           min(parameters.mcmc_prune_scoring_views, len(dataset)), dtype=int)
    with torch.no_grad():
        for index in selected:
            image_index = dataset.indices[index]
            camera_id = parser.camera_ids[image_index]
            width, height = parser.imsize_dict[camera_id]
            factor = min(1.0, parameters.mcmc_prune_resolution / max(width, height))
            resized_width, resized_height = max(1, round(width * factor)), max(1, round(height * factor))
            intrinsics = torch.tensor(parser.Ks_dict[camera_id], dtype=torch.float32, device=device)[None]
            intrinsics[:, 0] *= resized_width / width
            intrinsics[:, 1] *= resized_height / height
            view = torch.tensor(parser.camtoworlds[image_index], dtype=torch.float32, device=device)[None]
            if configuration.pose_opt:
                view = runner.pose_adjust(view, torch.tensor([int(index)], device=device))
            mask = load_training_mask(parser.data_dir, parser.image_names[image_index], (width, height))
            if mask is None:
                mask = parser.mask_dict[camera_id]
            if mask is not None:
                mask = torch.nn.functional.interpolate(
                    torch.as_tensor(mask, device=device).float()[None, None],
                    size=(resized_height, resized_width), mode="nearest",
                )[0, 0].bool()
            score_region(torch.linalg.inv(view), intrinsics, resized_width, resized_height, mask)
        # The volume reference is common to every score and cannot change ranking.
        log_volume = splats["scales"].sum(dim=-1)
        volume_weight = torch.exp(0.1 * (log_volume - log_volume.max()))
        return (hits * opacity * volume_weight).cpu().numpy(), len(selected)


def attach_pruning(trainer, parameters):
    import torch
    from gsplat.strategy.ops import remove

    runner_class = trainer["Runner"]
    original_init = runner_class.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        configuration = self.cfg
        if not len(self.trainset):
            raise ValueError("significance pruning requires training views")
        original_step = configuration.strategy.step_post_backward
        pruning_steps = frozenset(parameters.mcmc_prune_steps)

        def step_post_backward(params, optimizers, state, step, info, lr, scene=None):
            original_step(params, optimizers, state, step, info, lr, scene=scene)
            if step + 1 not in pruning_steps:
                return
            before = len(params["means"])
            scores, views = score_gaussians(self, parameters)
            mask = torch.from_numpy(removal_mask(scores, parameters.mcmc_prune_keep_fraction)).to(self.device)
            # MCMC's state contains a shared binomial table, not per-Gaussian rows.
            remove(params, optimizers, {}, mask, scene=scene)
            after = len(params["means"])
            configuration.strategy.cap_max = min(configuration.strategy.cap_max, after)
            record = {"iteration": step + 1, "before": before, "after": after,
                      "cap_max": configuration.strategy.cap_max, "scoring_views": views,
                      "resolution": parameters.mcmc_prune_resolution,
                      "score": "accepted_pixel_hits * opacity * volume**0.1"}
            with (Path(configuration.result_dir) / "pruning.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(f"Significance pruning: {json.dumps(record)}", flush=True)

        configuration.strategy.step_post_backward = step_post_backward

    runner_class.__init__ = __init__
