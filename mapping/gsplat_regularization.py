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
