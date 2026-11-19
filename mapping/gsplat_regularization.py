"""Turn off simple_trainer's opacity and scale regularization once refinement stops."""


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


# Keeps log() finite; a Gaussian this close to 0 or 1 opacity has nothing left to gain.
ENTROPY_EPSILON = 1e-6


def binary_entropy(logits):
    """Mean binary entropy of the sigmoid opacities: zero at 0 and 1, largest at 0.5."""
    opacity = logits.sigmoid().clamp(ENTROPY_EPSILON, 1 - ENTROPY_EPSILON)
    return -(opacity * opacity.log() + (1 - opacity) * (1 - opacity).log()).mean()


def add_opacity_entropy(strategy_classes, weight):
    """Add weight * binary_entropy(opacities) to every training step's gradient.

    The term pushes each opacity to 0 or 1: visible Gaussians turn opaque, and
    unused ones still sink below min_opacity so MCMC relocates them, which a
    negative opacity_reg_weight would prevent. simple_trainer has no hook into
    its loss, so the term is backpropagated on its own in step_pre_backward;
    its gradient accumulates with the main loss's.
    """
    for strategy_class in strategy_classes:
        def step_pre_backward(self, params, *args, original=strategy_class.step_pre_backward, **kwargs):
            (weight * binary_entropy(params["opacities"])).backward()
            return original(self, params, *args, **kwargs)

        strategy_class.step_pre_backward = step_pre_backward
