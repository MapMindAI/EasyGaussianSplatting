"""Tests for the regularization stop and the opacity entropy term."""

import types

import numpy as np
import pytest

from mapping import gsplat_regularization
from mapping.gsplat_regularization import (
    add_opacity_entropy,
    binary_entropy,
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


class Logits(np.ndarray):
    """The tensor methods binary_entropy uses, on numpy."""

    def sigmoid(self):
        return 1 / (1 + np.exp(-self))

    def clamp(self, low, high):
        return np.clip(self, low, high)

    def log(self):
        return np.log(self)


def logits(*values):
    return np.array(values, dtype=np.float64).view(Logits)


def test_binary_entropy_peaks_at_half_opacity():
    assert binary_entropy(logits(0.0)) == pytest.approx(np.log(2))
    assert binary_entropy(logits(-20.0, 20.0)) < 1e-4


class Loss:
    def __init__(self):
        self.weight = None
        self.backpropagated = False

    def __rmul__(self, weight):
        self.weight = weight
        return self

    def backward(self):
        self.backpropagated = True


class PreBackwardStrategy:
    def step_pre_backward(self, params, optimizers, state, step, info):
        return params


def test_opacity_entropy_backpropagates_before_the_strategy(monkeypatch):
    loss = Loss()
    monkeypatch.setattr(gsplat_regularization, "binary_entropy", lambda values: loss)
    strategy_class = type("Strategy", (PreBackwardStrategy,), {})
    add_opacity_entropy([strategy_class], 0.02)
    params = {"opacities": None}

    result = strategy_class().step_pre_backward(params=params, optimizers={}, state={}, step=0, info={})

    assert result is params
    assert loss.weight == 0.02 and loss.backpropagated
