#!/usr/bin/env python3
"""Run gsplat's simple_trainer with the per-image masks segment_people.py writes.

gsplat's COLMAP dataset only carries one mask per camera (the undistortion
ROI), so per-image masks are attached here instead: its __getitem__ is wrapped
to load "<data_dir>/masks/<image_name>.png" alongside each image. Takes
simple_trainer's own arguments.
"""
import os
import pathlib
import runpy
import sys
import numpy as np
import torch
from PIL import Image

GSPLAT_ROOT = pathlib.Path(os.environ.get("GSPLAT_DIR", pathlib.Path(__file__).resolve().parents[1] / "third_party" / "gsplat"))
EXAMPLES_DIR = str(GSPLAT_ROOT / "examples")


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


def attach_depths(dataset_class):
    """Supply the depth supervision simple_trainer's --depth_loss consumes.

    Its own source is the sparse COLMAP points, which it looks up in
    parser.point_indices -- a dict with no entry at all for a cube face SfM
    barely covered. That path is switched off and answered here with what
    generate_depth.py wrote, which is denser and covers the textureless surfaces
    SfM never triangulated. The file stores x and y normalized and depth in the
    COLMAP frame, so both are put into the frame the renderer works in here:
    pixels of the image as loaded, and the parser's normalized world. The scale
    comes off the parser rather than a module global so it survives a dataloader
    worker that re-imports instead of forking.
    """
    original_getitem = dataset_class.__getitem__

    def __getitem__(self, item):
        self.load_depths = False
        data = original_getitem(self, item)
        image_name = self.parser.image_names[self.indices[item]]
        depth_path = os.path.join(self.parser.data_dir, "depths", f"{image_name}.npy")
        rows = (
            np.load(depth_path)
            if os.path.exists(depth_path)
            else np.zeros((0, 3), dtype=np.float32)
        )

        assert self.patch_size is None, "depths are not cropped along with patches"
        height, width = data["image"].shape[:2]
        points = rows[:, :2] * np.array([width - 1, height - 1], dtype=np.float32)
        data["points"] = torch.from_numpy(points).float()
        scale_inv = 1.0 / similarity_scale(self.parser.transform)
        data["depths"] = torch.from_numpy(rows[:, 2] * scale_inv).float()
        return data

    dataset_class.__getitem__ = __getitem__


# Rendered depth below this fraction of the scene scale means the ray hit
# nothing, not that the surface is that close.
UNRENDERED_DEPTH = 1e-3


def guard_depth_loss(losses_module):
    """Supervise depth only where the render has geometry to supervise.

    The term is an L1 on inverse depth, and simple_trainer adds it for every
    sample once --depth_loss is on. A pixel no Gaussian covers yet renders a
    depth near zero, and its disparity buries every other term; an image
    generate_depth.py skipped averages over no points at all, which is NaN.
    Both reach the weights on the next step and stay there.
    """
    original_loss = losses_module.depth_l1_loss

    def depth_l1_loss(pred_depth, gt_depth, scene_scale=1.0, **kwargs):
        rendered = pred_depth > UNRENDERED_DEPTH * scene_scale
        if not rendered.any():
            return pred_depth.new_zeros(())
        return original_loss(
            pred_depth[rendered], gt_depth[rendered], scene_scale=scene_scale, **kwargs
        )

    losses_module.depth_l1_loss = depth_l1_loss


def composite_over_background(rendering, color):
    """Rasterize onto a fixed background colour rather than onto black.

    simple_trainer binds rasterization by name when it is imported, so the
    module has to be patched before runpy loads it.
    """
    channels = (color.red, color.green, color.blue)
    rasterization = rendering.rasterization
    background = None

    def rasterization_over_background(*args, **kwargs):
        # Built once: rebuilding it per call copies host to device, which
        # synchronizes the stream before every forward pass.
        nonlocal background
        viewmats = kwargs["viewmats"]
        if background is None:
            background = torch.tensor(channels, dtype=torch.float32, device=viewmats.device)
        # viewmats is [..., C, 4, 4] and backgrounds [..., C, 3], one colour
        # per camera in the batch. The viewer passes its own.
        kwargs.setdefault("backgrounds", background.expand(viewmats.shape[:-2] + (3,)))
        return rasterization(*args, **kwargs)

    rendering.rasterization = rasterization_over_background


sys.path.insert(0, EXAMPLES_DIR)
import gsplat  # noqa: E402
import gsplat.losses  # noqa: E402
import gsplat.rendering  # noqa: E402
from datasets.colmap import Dataset, Parser  # noqa: E402
from gsplat_server.parameters import load_parameters  # noqa: E402
from gsplat_server.proto import gsplat_pb2  # noqa: E402
from mapping.gsplat_world_frame import (  # noqa: E402
    capture_scene_transform,
    export_in_colmap_frame,
    similarity_scale,
)


def number(value):
    """Recover the decimal the parameter file was written with.

    float32 widens to values like 0.000600000028, which would otherwise reach
    the trainer's log and saved config verbatim.
    """
    return f"{value:.7g}"


SUBCOMMANDS = {
    gsplat_pb2.STRATEGY_UNSPECIFIED: "default",
    gsplat_pb2.STRATEGY_DEFAULT: "default",
    gsplat_pb2.STRATEGY_MCMC: "mcmc",
}


def strategy_flags(parameters):
    """simple_trainer flags for the chosen strategy.

    The two strategies expose different flags: DefaultStrategy has no cap_max,
    and MCMC has none of the grow/prune/reset knobs, so passing the wrong set
    makes the trainer exit before it starts.
    """
    flags = [
        "--strategy.refine-start-iter", str(parameters.refine_start_iter),
        "--strategy.refine-stop-iter", str(parameters.refine_stop_iter),
        "--strategy.refine-every", str(parameters.refine_every),
    ]
    if parameters.strategy == gsplat_pb2.STRATEGY_MCMC:
        return flags + ["--strategy.cap-max", str(parameters.cap_max)]

    flags += [
        "--strategy.grow-grad2d", number(parameters.grow_grad2d),
        "--strategy.prune-opa", number(parameters.prune_opa),
        "--strategy.grow-scale3d", number(parameters.grow_scale3d),
        "--strategy.prune-scale3d", number(parameters.prune_scale3d),
        "--strategy.prune-scale2d", number(parameters.prune_scale2d),
        "--strategy.reset-every", str(parameters.reset_every),
    ]
    if parameters.absgrad:
        flags.append("--strategy.absgrad")
    return flags


def load_job_parameters():
    try:
        parameter_index = sys.argv.index("--job_parameters")
    except ValueError:
        return None
    parameters = load_parameters(sys.argv[parameter_index + 1])
    flags = [
        "--data_factor", str(parameters.data_factor),
        "--max_steps", str(parameters.iterations),
        "--eval_steps", str(parameters.iterations),
        "--save_steps", str(parameters.iterations),
        "--ply_steps", str(parameters.iterations),
        "--opacity_reg", number(parameters.floater_reg_weight),
        "--scale_reg", number(parameters.floater_reg_weight),
        "--sh_degree", str(parameters.sh_degree),
        "--ssim_lambda", number(parameters.ssim_lambda),
        "--means_lr", number(parameters.means_lr),
        "--scales_lr", number(parameters.scales_lr),
        "--opacities_lr", number(parameters.opacities_lr),
        "--quats_lr", number(parameters.quats_lr),
        "--sh0_lr", number(parameters.sh0_lr),
        "--shN_lr", number(parameters.shN_lr),
        "--pose_opt" if parameters.pose_opt else "--no-pose_opt",
        "--pose_opt_lr", number(parameters.pose_opt_lr),
    ]
    if parameters.run_depth:
        flags += ["--depth_loss", "--depth_lambda", number(parameters.depth_lambda)]
    if parameters.packed:
        flags.append("--packed")
    flags += strategy_flags(parameters)
    sys.argv = sys.argv[:parameter_index] + sys.argv[parameter_index + 2:]
    # argv[1] is the strategy subcommand, which the parameters decide.
    sys.argv[1] = SUBCOMMANDS[parameters.strategy]
    sys.argv[2:2] = flags
    return parameters


parameters = load_job_parameters()
if parameters is not None and parameters.HasField("background_color"):
    composite_over_background(gsplat.rendering, parameters.background_color)
attach_masks(Dataset)
if parameters is not None and parameters.run_depth:
    attach_depths(Dataset)
    guard_depth_loss(gsplat.losses)
capture_scene_transform(Parser)
export_in_colmap_frame(gsplat)
runpy.run_path(os.path.join(EXAMPLES_DIR, "simple_trainer.py"), run_name="__main__")
