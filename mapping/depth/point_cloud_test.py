"""Tests for lifting depth rows into the merged, downsampled cloud."""

import importlib.util

import numpy as np
import pytest

from mapping.depth import point_cloud
from mapping.depth.point_cloud import (
    depth_point_cloud,
    image_points,
    merge_voxel_sums,
    read_point_cloud,
    scene_scale,
    unproject,
    voxel_means,
    voxel_sums,
    write_point_cloud,
)
from mapping.vggt_omega.depth_estimator import quaternion_to_matrix

INTRINSICS = np.array([[100.0, 0, 32], [0, 120.0, 24], [0, 0, 1]])


# The merge runs on torch, which the gsplat environment has and the slim CI image does not.
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"
)


def tensor(values):
    import torch

    return torch.tensor(values, dtype=torch.float64)


def test_unproject_inverts_the_projection():
    rotation = quaternion_to_matrix([0.1, -0.2, 0.3, 0.9])
    camera_from_world = np.column_stack([rotation, [0.5, -1.0, 2.0]])
    world = np.array([[0.3, -0.2, 1.0], [-0.5, 0.4, 2.5]])

    camera = world @ rotation.T + camera_from_world[:, 3]
    projected = camera @ INTRINSICS.T
    pixels = projected[:, :2] / projected[:, 2:]

    np.testing.assert_allclose(
        unproject(pixels, camera[:, 2], INTRINSICS, camera_from_world), world, atol=1e-9
    )


def test_scene_scale_is_the_largest_camera_distance_from_their_mean():
    assert scene_scale([[0, 0, 0], [2, 0, 0], [1, 3, 0]]) == pytest.approx(2.0)


@requires_torch
def test_voxels_average_points_and_colours_across_parts():
    first = voxel_sums(tensor([[0.1, 0.1, 0.1], [0.3, 0.1, 0.1]]), tensor([[0, 0, 0], [100, 0, 0]]), 1.0)
    second = voxel_sums(tensor([[0.2, 0.4, 0.1], [5.5, 0.5, 0.5]]), tensor([[50, 0, 0], [9, 9, 9]]), 1.0)

    points, colours = voxel_means(merge_voxel_sums([first, second]), 1.0)

    order = np.argsort(points[:, 0])
    np.testing.assert_allclose(points[order], [[0.2, 0.2, 0.1], [5.5, 0.5, 0.5]])
    np.testing.assert_allclose(colours[order], [[50, 0, 0], [9, 9, 9]])


@requires_torch
def test_voxels_on_either_side_of_the_origin_stay_apart_and_far_points_are_dropped():
    points = tensor([[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [-0.4, 0.6, 0.5], [3e6, 0, 0]])

    part = voxel_sums(points, tensor(np.zeros((4, 3))), 1.0)

    means, _ = voxel_means(part, 1.0)
    assert sorted(part[1][:, 0].tolist()) == [1, 2]
    np.testing.assert_allclose(sorted(means[:, 0].tolist()), [-0.45, 0.5], atol=1e-6)


@requires_torch
def test_depth_point_cloud_folds_to_the_same_cloud_as_one_merge(monkeypatch):
    monkeypatch.setattr(point_cloud, "_FOLD_ROWS", 60)
    generator = np.random.default_rng(0)
    image = generator.integers(0, 255, (20, 30, 3), dtype=np.uint8)
    camera_from_world = np.column_stack([np.eye(3), [0, 0, 1]])
    images = [
        (np.column_stack([generator.random((50, 2)), generator.uniform(1, 3, 50)]).astype(np.float32),
         image, INTRINSICS, camera_from_world)
        for _ in range(12)
    ]

    points, colours = depth_point_cloud(images, 0.2)

    expected_points, expected_colours = voxel_means(merge_voxel_sums([
        voxel_sums(*image_points(*item, "cpu"), 0.2) for item in images
    ]), 0.2)
    order, expected_order = np.lexsort(points.T), np.lexsort(expected_points.T)
    np.testing.assert_allclose(points[order], expected_points[expected_order], atol=1e-6)
    np.testing.assert_array_equal(
        colours[order], np.clip(np.rint(expected_colours[expected_order]), 0, 255).astype(np.uint8)
    )


@requires_torch
def test_depth_rows_land_where_the_camera_saw_them_coloured_from_the_image():
    camera_from_world = np.column_stack([np.eye(3), [0.0, 0.0, 0.0]])
    image = np.zeros((49, 65, 3), dtype=np.uint8)
    image[23, 31] = (10, 20, 30)  # BGR
    rows = np.array([[31 / 64, 23 / 48, 4.0]], dtype=np.float32)

    points, colours = depth_point_cloud([(rows, image, INTRINSICS, camera_from_world)], 0.01)

    # Pixel (31, 23) is centred at (31.5, 23.5), half a pixel short of the
    # principal point on both axes.
    np.testing.assert_allclose(points, [[-0.5 * 4 / 100, -0.5 * 4 / 120, 4.0]], atol=1e-6)
    assert colours.tolist() == [[30, 20, 10]]


def test_point_cloud_round_trips_through_a_binary_ply(tmp_path):
    path = tmp_path / "points.ply"
    write_point_cloud(path, np.array([[1.0, 2.0, 3.0]]), np.array([[255, 128, 0]], dtype=np.uint8))

    points, colours = read_point_cloud(path)

    assert points.tolist() == [[1.0, 2.0, 3.0]]
    assert colours.tolist() == [[255, 128, 0]]
