"""Decrease depth_lambda over refinement, as the public HY-World trainer does."""


def depth_lambda_at(step, initial, final, end_step):
    """`initial` decreased linearly to `final` at `end_step`, then held."""
    progress = min(step / max(end_step, 1), 1.0)
    return initial + (final - initial) * progress


def schedule_depth_loss(losses_module, strategy_classes, initial, final, end_step):
    """Scale depth_l1_loss by depth_lambda_at the current step.

    simple_trainer multiplies the term by a fixed --depth_lambda, which is then
    passed as 1. The loss carries no step, so it is read from the strategy's
    step_post_backward, which runs after every backward pass. simple_trainer
    binds the loss by name on import, so patch before runpy.
    """
    next_step = 0

    for strategy_class in strategy_classes:
        def step_post_backward(self, *args, original=strategy_class.step_post_backward, **kwargs):
            nonlocal next_step
            next_step = kwargs["step"] + 1
            return original(self, *args, **kwargs)

        strategy_class.step_post_backward = step_post_backward

    loss = losses_module.depth_l1_loss

    def depth_l1_loss(*args, **kwargs):
        return depth_lambda_at(next_step, initial, final, end_step) * loss(*args, **kwargs)

    losses_module.depth_l1_loss = depth_l1_loss
