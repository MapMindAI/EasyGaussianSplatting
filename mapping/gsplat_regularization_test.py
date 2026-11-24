"""Tests for stopping regularization after refinement, and the anisotropy penalty."""

import types

import pytest

from mapping.gsplat_regularization import (
    ANISOTROPY_STEP_INTERVAL,
    anisotropy_loss,
    regularize_anisotropy,
    stop_regularization_after_refinement,
)


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


def test_anisotropy_penalizes_only_the_ratio_past_the_bound():
    torch = pytest.importorskip("torch")
    log_scales = torch.log(torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 8.0]]))

    assert anisotropy_loss(log_scales, max_ratio=5.0).item() == pytest.approx(1.5)


def test_anisotropy_adds_its_gradient_every_tenth_step():
    torch = pytest.importorskip("torch")
    strategy_class = type("Strategy", (), {"step_pre_backward": lambda self, *args, **kwargs: None})
    regularize_anisotropy([strategy_class], weight=0.1, max_ratio=5.0)
    scales = torch.log(torch.tensor([[1.0, 1.0, 8.0]])).requires_grad_()

    gradients = []
    for step in range(ANISOTROPY_STEP_INTERVAL + 1):
        scales.grad = None
        strategy_class().step_pre_backward(params={"scales": scales}, optimizers={}, state={}, step=step, info={})
        gradients.append(scales.grad is not None)

    assert gradients == [True] + [False] * (ANISOTROPY_STEP_INTERVAL - 1) + [True]
