#!/usr/bin/env python3
"""Run gsplat's simple_trainer with the per-image masks segment_people.py writes.

gsplat's COLMAP dataset only carries one mask per camera (the undistortion
ROI), so per-image masks are attached here instead: its __getitem__ is wrapped
to load "<data_dir>/masks/<image_name>.png" alongside each image. Takes
simple_trainer's own arguments, plus --job_parameters, and --block with
--coarse_checkpoint to train one block (doc/block_training.md).
"""
import concurrent.futures
import os
import pathlib
import pprint
import runpy
import sys

# Running this file puts mapping/ on sys.path, where mapping/triton/ shadows the
# triton package torch probes for on import.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [path for path in sys.path if os.path.abspath(path or ".") != SCRIPT_DIR]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import tqdm  # noqa: E402
from PIL import Image  # noqa: E402

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


def resize_image(source_path, resized_path, factor):
    """Write one image downscaled by `factor` as PNG, as gsplat's Parser expects.

    `draft` lets libjpeg decode straight at 1/2, 1/4, or 1/8 scale, which is
    most of the cost of decoding full size and filtering down.
    """
    with Image.open(source_path) as image:
        size = (round(image.width / factor), round(image.height / factor))
        image.draft("RGB", size)
        image = image.convert("RGB")
        if image.size != size:
            image = image.resize(size, Image.BICUBIC)
        # Renamed into place, so an interrupted write never leaves a truncated
        # PNG that a resumed run would take as done.
        image.save(resized_path + ".partial", format="PNG", compress_level=1)
    os.replace(resized_path + ".partial", resized_path)


def resize_images(image_dir, resized_dir, factor):
    """Resize every image under `image_dir` in parallel, skipping ones already written.

    Output paths mirror gsplat's _resize_image_folder: the same relative path
    with a .png extension, so the Parser pairs them with the COLMAP names.
    """
    jobs = []
    for directory, _, files in os.walk(image_dir):
        output_directory = os.path.join(resized_dir, os.path.relpath(directory, image_dir))
        os.makedirs(output_directory, exist_ok=True)
        for name in files:
            resized_path = os.path.join(output_directory, os.path.splitext(name)[0] + ".png")
            if not os.path.isfile(resized_path):
                jobs.append((os.path.join(directory, name), resized_path))
    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as pool:
        futures = [pool.submit(resize_image, source_path, resized_path, factor)
                   for source_path, resized_path in jobs]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                                desc=f"Downscaling images by {factor}x", unit="image"):
            future.result()


def downsample_images(parser_class):
    """Write the images_<factor>/ folder gsplat's Parser requires but never creates.

    Staged under another name because the Parser takes any images_<factor>/ as
    complete; resize_images skips files already written, so an interrupted
    resize resumes rather than training on part of the images.
    """
    original_init = parser_class.__init__

    def __init__(self, data_dir, factor=1, **kwargs):
        resized_dir = os.path.join(data_dir, f"images_{factor}")
        if factor > 1 and not os.path.isdir(resized_dir):
            staging_dir = resized_dir + "_partial"
            resize_images(os.path.join(data_dir, "images"), staging_dir, factor)
            os.replace(staging_dir, resized_dir)
        original_init(self, data_dir=data_dir, factor=factor, **kwargs)

    parser_class.__init__ = __init__


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
        points, depths = depth_supervision_in_normalized_frame(
            self.parser.transform, rows, width, height
        )
        data["points"] = torch.from_numpy(points).float()
        data["depths"] = torch.from_numpy(depths).float()
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


def restrict_to_images(dataset_class, image_names):
    """Keep only a block's images in both splits.

    The split is still taken over the full model's indices, so an image held
    out of the coarse run stays held out of every block.
    """
    original_init = dataset_class.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.indices = np.array([index for index in self.indices
                                 if self.parser.image_names[index] in image_names], dtype=int)

    dataset_class.__init__ = __init__


def train_block(trainer, coarse_checkpoint, points, owners, block, result_dir):
    """Start from the coarse Gaussians around the block and render the rest frozen.

    `trainer` is simple_trainer's namespace. The coarse Gaussians in the
    block's expanded region become the trainable splats; the rest are
    appended to every render with no gradient or optimizer state, so the
    block's images never show content it would have to explain with floaters.
    MCMC only: it reads nothing per Gaussian from the render, while
    DefaultStrategy's densification would see the context in it.
    """
    coarse = torch.load(coarse_checkpoint, map_location="cpu", weights_only=True)["splats"]
    context = {}
    original_create = trainer["create_splats_with_optimizers"]

    def create_splats_with_optimizers(parser, *args, **kwargs):
        splats, optimizers = original_create(parser, *args, **kwargs)
        np.save(os.path.join(result_dir, "scene_transform.npy"), parser.transform)
        means = block_gaussians.to_colmap(parser.transform, coarse["means"].numpy())
        inside = torch.from_numpy(block_gaussians.expanded_region(means, points, owners, block))
        print(f"Block: {int(inside.sum())} trainable and {int((~inside).sum())} frozen coarse Gaussians", flush=True)
        for name, optimizer in optimizers.items():
            device = splats[name].device
            assert coarse[name].shape[1:] == splats[name].shape[1:], f"coarse {name} has another shape"
            splats[name] = torch.nn.Parameter(coarse[name][inside].to(device))
            optimizer.param_groups[0]["params"] = [splats[name]]
            context[name] = coarse[name][~inside].to(device)
        return splats, optimizers

    runner_class = trainer["Runner"]
    original_rasterize = runner_class.rasterize_splats

    def rasterize_splats(self, *args, splats=None, **kwargs):
        splats = self.splats if splats is None else splats
        combined = {name: torch.cat([splats[name], context[name]]) for name in context}
        return original_rasterize(self, *args, splats=combined, **kwargs)

    trainer["create_splats_with_optimizers"] = create_splats_with_optimizers
    runner_class.rasterize_splats = rasterize_splats


def take_option(name):
    """Remove `name` and its value from the command line, returning the value."""
    if name not in sys.argv:
        return None
    index = sys.argv.index(name)
    value = sys.argv[index + 1]
    del sys.argv[index:index + 2]
    return value


sys.path.insert(0, EXAMPLES_DIR)
import gsplat  # noqa: E402
import gsplat.distributed  # noqa: E402
import gsplat.losses  # noqa: E402
import gsplat.rendering  # noqa: E402
from datasets.colmap import Dataset, Parser  # noqa: E402
from gsplat_server.parameters import load_parameters  # noqa: E402
from mapping import block_gaussians  # noqa: E402
from gsplat_server.proto import gsplat_pb2  # noqa: E402
from mapping.gsplat_world_frame import (  # noqa: E402
    capture_scene_transform,
    depth_supervision_in_normalized_frame,
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
        return flags + [
            "--strategy.cap-max", str(parameters.cap_max),
            "--strategy.min-opacity", number(parameters.min_opacity),
        ]

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
        "--test_every", str(parameters.test_every),
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


original_cli = gsplat.distributed.cli


block_training = None


def cli_with_configuration(function, configuration, verbose=False):
    print("Training configuration:", flush=True)
    pprint.pprint(vars(configuration), sort_dicts=False, width=100)
    sys.stdout.flush()
    if block_training is not None:
        # simple_trainer's functions and classes are only reachable once
        # runpy has defined them, through the globals of its main.
        train_block(function.__globals__, *block_training, configuration.result_dir)
    return original_cli(function, configuration, verbose=verbose)


gsplat.distributed.cli = cli_with_configuration
# --block <manifest.json>:<name> trains one block of block_partition.py's
# manifest, from --coarse_checkpoint.
block_option = take_option("--block")
coarse_checkpoint = take_option("--coarse_checkpoint")
parameters = load_job_parameters()
if block_option is not None:
    assert coarse_checkpoint is not None, "--block needs --coarse_checkpoint"
    assert parameters is not None and parameters.strategy == gsplat_pb2.STRATEGY_MCMC, \
        "block training needs STRATEGY_MCMC"
    manifest_path, block_name = block_option.rsplit(":", 1)
    block, block_index, block_points, block_owners = block_gaussians.load_block(manifest_path, block_name)
    restrict_to_images(Dataset, set(block["training_images"]))
    block_training = (coarse_checkpoint, block_points, block_owners, block_index)
if parameters is not None and parameters.HasField("background_color"):
    composite_over_background(gsplat.rendering, parameters.background_color)
downsample_images(Parser)
attach_masks(Dataset)
if parameters is not None and parameters.run_depth:
    attach_depths(Dataset)
    guard_depth_loss(gsplat.losses)
capture_scene_transform(Parser)
export_in_colmap_frame(gsplat)
runpy.run_path(os.path.join(EXAMPLES_DIR, "simple_trainer.py"), run_name="__main__")
