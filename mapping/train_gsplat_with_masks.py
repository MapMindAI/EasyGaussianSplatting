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


sys.path.insert(0, EXAMPLES_DIR)
import gsplat  # noqa: E402
from datasets.colmap import Dataset, Parser  # noqa: E402
from gsplat_server.parameters import load_parameters  # noqa: E402
from gsplat_server.proto import gsplat_pb2  # noqa: E402
from mapping.gsplat_world_frame import (  # noqa: E402
    capture_scene_transform,
    export_in_colmap_frame,
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
        return
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
        "--pose_opt" if parameters.pose_opt else "--no-pose_opt",
    ]
    if parameters.packed:
        flags.append("--packed")
    flags += strategy_flags(parameters)
    sys.argv = sys.argv[:parameter_index] + sys.argv[parameter_index + 2:]
    # argv[1] is the strategy subcommand, which the parameters decide.
    sys.argv[1] = SUBCOMMANDS[parameters.strategy]
    sys.argv[2:2] = flags


load_job_parameters()
attach_masks(Dataset)
capture_scene_transform(Parser)
export_in_colmap_frame(gsplat)
runpy.run_path(os.path.join(EXAMPLES_DIR, "simple_trainer.py"), run_name="__main__")
