"""Tests for stopping regularization after refinement."""

import types

from mapping.gsplat_regularization import stop_regularization_after_refinement


class Values:
    def new_zeros(self, shape):
        return 0.0


class Strategy:
    def step_post_backward(self, params, optimizers, state, step, info, **kwargs):
        return step


def patched(refine_stop_iter):
    losses = types.SimpleNamespace(opacity_reg_loss=lambda values: 1.0, scale_reg_loss=lambda values: 2.0)
    strategy_class = type("Strategy", (Strategy,), {})
    stop_regularization_after_refinement(losses, [strategy_class], refine_stop_iter)
    return losses, strategy_class()


def run_step(strategy, step):
    return strategy.step_post_backward(params={}, optimizers={}, state={}, step=step, info={})


def test_regularization_applies_before_the_first_step():
    losses, _ = patched(refine_stop_iter=3)

    assert losses.opacity_reg_loss(Values()) == 1.0
    assert losses.scale_reg_loss(Values()) == 2.0


def test_regularization_stops_at_refine_stop_iter():
    losses, strategy = patched(refine_stop_iter=3)

    applied = []
    for step in range(5):
        applied.append(losses.opacity_reg_loss(Values()) != 0.0)
        assert run_step(strategy, step) == step

    assert applied == [True, True, True, False, False]


def test_scale_regularization_stops_with_opacity():
    losses, strategy = patched(refine_stop_iter=1)

    run_step(strategy, 0)

    assert losses.scale_reg_loss(Values()) == 0.0
