"""MaskGaussian (arXiv 2412.20522) for simple_trainer's DefaultStrategy.

Each Gaussian gets two logits, keep and drop. Every training render samples a
hard keep/drop mask from them, and gsplat's masked rasterization
(`rasterization(gaussian_masks=...)`) renders the dropped Gaussians as nothing
while still giving them the gradient that would bring them back. A loss on the
squared mean mask pushes Gaussians to drop, and those dropped in every one of
PRUNE_DRAWS samples are pruned. Evaluation, the viewer, and the export keep a
Gaussian when its keep logit is the larger.

Settings follow the authors' code (github.com/kaikai23/MaskGaussian).
"""

MASK_KEY = "mask_scores"
INITIAL_SCORES = (10.0, 1.0)  # Keep and drop logits: keeps with probability 0.9999.
LEARNING_RATE = 0.01
PRUNE_DRAWS = 10
# After refinement stops, prune this often; before, at every refinement step.
PRUNE_EVERY_AFTER_REFINEMENT = 1000


def prune_due(step, refine_start_iter, refine_stop_iter, refine_every):
    """Whether the masks prune at `step`: with each of DefaultStrategy's
    refinement steps, then every PRUNE_EVERY_AFTER_REFINEMENT steps."""
    if step < refine_stop_iter:
        return step > refine_start_iter and step % refine_every == 0
    return step % PRUNE_EVERY_AFTER_REFINEMENT == 0


def sample_masks(scores):
    """One hard keep (1) or drop (0) per Gaussian, straight through to `scores`."""
    import torch.nn.functional as functional

    return functional.gumbel_softmax(functional.log_softmax(scores, dim=-1), hard=True)[:, 0]


def kept(scores):
    return scores[:, 0] > scores[:, 1]


def always_dropped(scores):
    """Gaussians dropped in each of PRUNE_DRAWS samples."""
    import torch

    with torch.no_grad():
        return sum(sample_masks(scores) for _ in range(PRUNE_DRAWS)) == 0


def add_mask_loss(masks, weight):
    """Add weight * mean(masks)^2 to the loss through `masks`' gradient, since
    simple_trainer builds its loss with no hook for another term."""
    count = masks.numel()
    mean = masks.detach().mean()
    masks.register_hook(lambda gradient: gradient + weight * 2 * mean / count)


def enable_mask_gaussian(trainer, strategy_class, weight):
    """Train with masks. `trainer` is simple_trainer's namespace; the strategy
    class is patched in place."""
    import torch

    from gsplat.strategy.ops import remove

    original_create = trainer["create_splats_with_optimizers"]
    splats_holder = {}

    def create_splats_with_optimizers(*args, **kwargs):
        splats, optimizers = original_create(*args, **kwargs)
        means = splats["means"]
        splats[MASK_KEY] = torch.nn.Parameter(
            torch.tensor(INITIAL_SCORES, device=means.device).repeat(len(means), 1)
        )
        optimizers[MASK_KEY] = torch.optim.Adam(
            [{"params": splats[MASK_KEY], "lr": LEARNING_RATE, "name": MASK_KEY}], eps=1e-15
        )
        splats_holder["splats"] = splats
        return splats, optimizers

    runner_class = trainer["Runner"]
    original_rasterize = runner_class.rasterize_splats

    def rasterize_splats(self, *args, splats=None, **kwargs):
        scores = (self.splats if splats is None else splats)[MASK_KEY]
        if torch.is_grad_enabled():
            masks = sample_masks(scores)
            add_mask_loss(masks, weight)
        else:
            masks = kept(scores).float()
        return original_rasterize(self, *args, splats=splats, gaussian_masks=masks, **kwargs)

    original_step = strategy_class.step_post_backward

    def step_post_backward(self, params, optimizers, state, step, info, **kwargs):
        result = original_step(self, params, optimizers, state, step, info, **kwargs)
        if prune_due(step, self.refine_start_iter, self.refine_stop_iter, self.refine_every):
            dropped = always_dropped(params[MASK_KEY])
            if dropped.any():
                remove(params=params, optimizers=optimizers, state=state, mask=dropped,
                       scene=kwargs.get("scene"))
                if self.verbose:
                    print(f"Step {step}: {int(dropped.sum())} GSs pruned by their masks. "
                          f"Now having {len(params['means'])} GSs.")
        return result

    original_export = trainer["export_splats"]

    def export_splats(**kwargs):
        keep = kept(splats_holder["splats"][MASK_KEY])
        return original_export(**{
            name: value[keep] if isinstance(value, torch.Tensor) and len(value) == len(keep) else value
            for name, value in kwargs.items()
        })

    trainer["create_splats_with_optimizers"] = create_splats_with_optimizers
    runner_class.rasterize_splats = rasterize_splats
    strategy_class.step_post_backward = step_post_backward
    trainer["export_splats"] = export_splats
