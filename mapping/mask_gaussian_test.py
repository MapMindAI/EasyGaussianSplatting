"""Tests for MaskGaussian's schedule, sampling, and loss."""

import pytest

from mapping.mask_gaussian import PRUNE_EVERY_AFTER_REFINEMENT, prune_due


def test_masks_prune_with_each_refinement_then_on_their_own_schedule():
    due = [step for step in range(0, 4001) if prune_due(step, 500, 2000, 100)]

    assert due[:3] == [600, 700, 800] and 1900 in due and 2000 in due
    assert [step for step in due if step >= 2000] == [2000, 2000 + PRUNE_EVERY_AFTER_REFINEMENT, 4000]


def test_samples_are_hard_and_follow_the_logits():
    torch = pytest.importorskip("torch")
    from mapping.mask_gaussian import INITIAL_SCORES, always_dropped, kept, sample_masks

    scores = torch.tensor([INITIAL_SCORES, (-10.0, 10.0)] * 50)

    masks = sample_masks(scores)

    assert set(masks.tolist()) <= {0.0, 1.0}
    assert kept(scores).tolist() == [True, False] * 50
    assert always_dropped(scores).tolist() == [False, True] * 50


def test_mask_loss_adds_the_gradient_of_the_squared_mean():
    torch = pytest.importorskip("torch")
    from mapping.mask_gaussian import add_mask_loss

    masks = torch.tensor([1.0, 0.0, 1.0, 1.0], requires_grad=True)
    reference = masks.detach().clone().requires_grad_()

    add_mask_loss(masks, weight=0.5)
    (masks * 0).sum().backward()
    (0.5 * reference.mean() ** 2).backward()

    torch.testing.assert_close(masks.grad, reference.grad)
