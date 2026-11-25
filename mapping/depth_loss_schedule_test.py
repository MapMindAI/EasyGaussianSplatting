"""Tests for the depth_lambda decay."""

import types

import pytest

from mapping.depth_loss_schedule import depth_lambda_at, schedule_depth_loss


def test_depth_lambda_decreases_linearly_then_holds():
    assert depth_lambda_at(0, 0.2, 0.01, 100) == pytest.approx(0.2)
    assert depth_lambda_at(50, 0.2, 0.01, 100) == pytest.approx(0.105)
    assert depth_lambda_at(100, 0.2, 0.01, 100) == pytest.approx(0.01)
    assert depth_lambda_at(500, 0.2, 0.01, 100) == pytest.approx(0.01)


def test_depth_loss_is_weighted_at_the_next_step():
    class Strategy:
        def step_post_backward(self, params, optimizers, state, step, info, **kwargs):
            return step

    losses = types.SimpleNamespace(depth_l1_loss=lambda predicted, truth: 1.0)
    schedule_depth_loss(losses, [Strategy], initial=0.2, final=0.01, end_step=10)

    assert losses.depth_l1_loss(None, None) == pytest.approx(0.2)
    Strategy().step_post_backward(params={}, optimizers={}, state={}, step=9, info={})
    assert losses.depth_l1_loss(None, None) == pytest.approx(0.01)
