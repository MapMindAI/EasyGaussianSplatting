import numpy as np

from mapping.depth.depth_alignment import (
    FITTED,
    GROUP_SCALE,
    SCALE_ONLY,
    Fit,
    View,
    aligned_depth,
    covisible_neighbours,
    fit_frame,
    fused_depth,
    percentile_agreement,
)


def test_fit_recovers_a_disparity_affine_despite_outliers():
    random = np.random.default_rng(1)
    reference = random.uniform(2.0, 12.0, 200)
    predicted = 1 / ((1 / reference - 0.02) / 0.9)  # 1/reference = 0.9/predicted + 0.02
    predicted[:30] *= random.uniform(1.3, 2.0, 30)

    fit = fit_frame(predicted, reference, largest_depth=50.0)

    assert fit.status == FITTED and fit.inliers == 170
    np.testing.assert_allclose([fit.scale, fit.shift], [0.9, 0.02], atol=1e-6)
    np.testing.assert_allclose(aligned_depth(predicted[30:], fit), reference[30:], rtol=1e-5)


def test_fit_falls_back_to_scale_when_the_depths_span_too_little():
    reference = np.linspace(5.0, 6.0, 50)

    fit = fit_frame(reference / 1.1, reference, largest_depth=10.0)

    assert fit.status == SCALE_ONLY and fit.shift == 0
    np.testing.assert_allclose(fit.scale, 1 / 1.1)


def test_fit_rejects_a_shift_that_sends_far_depths_behind_the_camera():
    reference = np.linspace(2.0, 8.0, 60)
    predicted = 1 / ((1 / reference + 0.05) / 1.2)  # A shift of -0.05 inverse units.

    fit = fit_frame(predicted, reference, largest_depth=100.0)

    assert fit.status != FITTED


def test_too_few_samples_keep_the_group_scale():
    assert fit_frame([1.0] * 5, [1.0] * 5, 10.0).status == GROUP_SCALE


def test_aligned_depth_keeps_holes_and_drops_depths_sent_behind_the_camera():
    depth = np.array([0.0, 1.0, 100.0])

    np.testing.assert_allclose(aligned_depth(depth, Fit(1.0, -0.05, FITTED)), [0.0, 1 / 0.95, 0.0])


def test_percentile_agreement_drops_samples_out_of_order():
    reference = np.arange(1.0, 11.0)
    predicted = reference.copy()
    predicted[0] = 20.0  # Nearest by reference, farthest by prediction.

    agree = percentile_agreement(predicted, reference)

    assert not agree[0] and agree[1:].all()


def test_neighbours_are_the_frames_sharing_the_most_points():
    neighbours = covisible_neighbours(
        [np.arange(0, 30), np.arange(5, 30), np.arange(20, 50), np.arange(100, 120)], 200
    )

    assert neighbours == [[1, 2], [0, 2], [0, 1], []]


def view_at(camera_x, depth):
    """A 40 x 40, 90 degree view from (camera_x, 0, 0) looking along +z."""
    intrinsics = np.array([[20.0, 0, 20], [0, 20.0, 20], [0, 0, 1]])
    camera_from_world = np.column_stack([np.eye(3), [-camera_x, 0, 0]])
    depth = np.asarray(depth, dtype=np.float32)
    return View(depth, depth > 0, intrinsics, camera_from_world)


def test_fusion_averages_agreeing_views_and_zeroes_free_space():
    wall = np.full((40, 40), 10.0, dtype=np.float32)
    own = wall * 1.04
    own[20, 20] = 5.0  # A floater the neighbours see through to the wall.

    fused = fused_depth(view_at(0.0, own), [view_at(0.5, wall), view_at(-0.5, wall)])

    assert fused[20, 20] == 0
    # Each estimate weighs by its inverse square depth.
    weights = np.array([1 / 10.4**2, 1 / 10.0**2, 1 / 10.0**2])
    expected = (weights * [10.4, 10.0, 10.0]).sum() / weights.sum()
    np.testing.assert_allclose(fused[10, 20], expected, rtol=1e-5)


def test_fusion_keeps_pixels_without_evidence_against_them():
    own = np.full((40, 40), 10.0, dtype=np.float32)
    nearer = np.full((40, 40), 5.0, dtype=np.float32)  # Occludes the whole wall.

    assert (fused_depth(view_at(0.0, own), [view_at(0.5, nearer)]) == own).all()
    assert (fused_depth(view_at(0.0, own), []) == own).all()
