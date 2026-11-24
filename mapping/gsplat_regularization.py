"""Regularization terms simple_trainer lacks or cannot stop."""


def stop_regularization_after_refinement(losses_module, strategy_classes, refine_stop_iter):
    """Make opacity_reg_loss and scale_reg_loss zero from refine_stop_iter on.

    The opacity term exists to sink unused Gaussians below min_opacity so MCMC
    relocates them; after refine_stop_iter nothing relocates, so it only dims
    every Gaussian. The losses carry no step, so the step is read from the
    strategy's step_post_backward, which runs after every backward pass.
    simple_trainer binds the losses by name on import, so patch before runpy.
    """
    next_step = 0

    for strategy_class in strategy_classes:
        def step_post_backward(self, *args, original=strategy_class.step_post_backward, **kwargs):
            nonlocal next_step
            next_step = kwargs["step"] + 1
            return original(self, *args, **kwargs)

        strategy_class.step_post_backward = step_post_backward

    def gate(loss):
        def regularization(values):
            if next_step >= refine_stop_iter:
                return values.new_zeros(())
            return loss(values)
        return regularization

    losses_module.opacity_reg_loss = gate(losses_module.opacity_reg_loss)
    losses_module.scale_reg_loss = gate(losses_module.scale_reg_loss)


# How often PhysGaussian's public implementation applies its anisotropy penalty.
ANISOTROPY_STEP_INTERVAL = 10


def anisotropy_loss(log_scales, max_ratio):
    """Mean excess of each Gaussian's largest over smallest scale past `max_ratio`."""
    ratios = (log_scales.amax(dim=-1) - log_scales.amin(dim=-1)).exp()
    return (ratios - max_ratio).clamp(min=0).mean()


def regularize_anisotropy(strategy_classes, weight, max_ratio):
    """simple_trainer's loss has no hook for another term, so the penalty is
    back-propagated on its own before the main backward; both accumulate into
    the scales' gradient."""
    for strategy_class in strategy_classes:
        def step_pre_backward(self, *args, original=strategy_class.step_pre_backward, **kwargs):
            original(self, *args, **kwargs)
            if kwargs["step"] % ANISOTROPY_STEP_INTERVAL == 0:
                (weight * anisotropy_loss(kwargs["params"]["scales"], max_ratio)).backward()

        strategy_class.step_pre_backward = step_pre_backward
