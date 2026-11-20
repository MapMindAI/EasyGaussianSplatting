"""Tests for the pose-baseline and sparse-depth group scale."""

import numpy as np
import pytest

from mapping.depth.scale_alignment import (
    PoseScale,
    align_sparse_depth,
    median_relative_error,
    pose_depth_scale,
)


def camera_from_world(center, yaw_degrees=0.0):
    angle = np.radians(yaw_degrees)
    rotation = np.array([
        [np.cos(angle), 0, -np.sin(angle)],
        [0, 1, 0],
        [np.sin(angle), 0, np.cos(angle)],
    ])
    return np.column_stack([rotation, -rotation @ np.asarray(center, float)])


def scaled(poses, factor):
    return [np.column_stack([pose[:, :3], pose[:, 3] * factor]) for pose in poses]


def reference_rig():
    centers = [(0, 0, 0), (1, 0, 0), (1, 0.5, 1), (0, 0.3, 2)]
    return [camera_from_world(center, 10 * index) for index, center in enumerate(centers)]


def test_pose_scale_recovers_the_factor_between_baselines():
    reference = reference_rig()

    pose_scale = pose_depth_scale(scaled(reference, 1 / 4), reference)

    assert pose_scale.scale == pytest.approx(4.0)
    assert pose_scale.median_error == pytest.approx(0.0, abs=1e-9)


def test_pose_scale_refuses_inconsistent_cameras_and_too_few_frames():
    reference = reference_rig()
    predicted = scaled(reference, 1 / 4)
    predicted[2] = camera_from_world((-5, 3, 0), 90)

    assert pose_depth_scale(predicted, reference) is None
    assert pose_depth_scale(predicted[:2], reference[:2]) is None


def sparse_samples(frame_count=3, per_frame=20, factor=3.0):
    generator = np.random.default_rng(0)
    frames = np.repeat(np.arange(frame_count), per_frame)
    points = np.arange(frame_count * per_frame)
    reference = generator.uniform(1.0, 10.0, frame_count * per_frame)
    return frames, points, reference / factor, reference


def test_sparse_scale_recovers_the_ratio_despite_outliers():
    frames, points, predicted, reference = sparse_samples()
    predicted[:8] *= 50

    assert align_sparse_depth(frames, points, predicted, reference) == pytest.approx(3.0)


def test_sparse_scale_balances_frames_rather_than_points():
    frames, points, predicted, reference = sparse_samples(frame_count=2, per_frame=40)
    # One frame sees many more points, all at a slightly different ratio.
    frames = np.concatenate([frames, np.zeros(200, int)])
    points = np.concatenate([points, 1000 + np.arange(200)])
    reference = np.concatenate([reference, np.full(200, 6.0)])
    predicted = np.concatenate([predicted, np.full(200, 6.0 / 3.1)])

    scale = align_sparse_depth(frames, points, predicted, reference)

    # Weighting points equally would give about 3.07.
    assert 3.0 < scale < 3.06


def test_sparse_scale_refuses_too_few_points_or_a_single_frame():
    frames, points, predicted, reference = sparse_samples(per_frame=5)
    assert align_sparse_depth(frames, points, predicted, reference) is None

    frames, points, predicted, reference = sparse_samples(frame_count=1, per_frame=60)
    assert align_sparse_depth(frames, points, predicted, reference) is None


def test_sparse_scale_ignores_far_and_invalid_references():
    frames, points, predicted, reference = sparse_samples()
    reference[:5] = 500.0
    predicted[5:8] = [0.0, -1.0, np.nan]

    assert align_sparse_depth(frames, points, predicted, reference) == pytest.approx(3.0)


def test_a_pose_scale_fixes_a_group_with_too_few_sparse_points():
    frames, points, predicted, reference = sparse_samples(per_frame=3)

    scale = align_sparse_depth(frames, points, predicted, reference, PoseScale(3.0, 0.01))

    assert scale == pytest.approx(3.0)


def test_median_relative_error_measures_the_scaled_fit():
    assert median_relative_error(2.0, [1.0, 2.0, 3.0], [2.0, 4.0, 6.6]) == pytest.approx(0.0)
    assert median_relative_error(2.0, [1.0], [np.nan]) == np.inf
