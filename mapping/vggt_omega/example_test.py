"""Tests for the example's merged point cloud."""

import numpy as np

from mapping.vggt_omega.depth_estimator import quaternion_to_matrix
from mapping.vggt_omega.example import cloud_pixels, world_points, write_point_cloud


def test_world_points_invert_the_projection():
    rotation = quaternion_to_matrix([0.1, -0.2, 0.3, 0.9])
    camera_from_world = np.column_stack([rotation, [0.5, -1.0, 2.0]])
    intrinsics = np.array([[100.0, 0, 32], [0, 120.0, 24], [0, 0, 1]])
    world = np.array([[0.3, -0.2, 1.0], [-0.5, 0.4, 2.5]])

    camera = world @ rotation.T + camera_from_world[:, 3]
    projected = camera @ intrinsics.T
    pixels = np.rint(projected[:, :2] / projected[:, 2:]).astype(int)
    depth = np.zeros((48, 64))
    depth[pixels[:, 1], pixels[:, 0]] = camera[:, 2]

    lifted = world_points(depth, camera_from_world, intrinsics, pixels)
    reprojected = (lifted @ rotation.T + camera_from_world[:, 3]) @ intrinsics.T
    np.testing.assert_allclose(reprojected[:, :2] / reprojected[:, 2:], pixels, atol=1e-9)
    np.testing.assert_allclose(reprojected[:, 2], camera[:, 2])


def test_cloud_pixels_keep_the_confident_half_of_the_grid():
    confidence = np.tile(np.arange(10, dtype=float), (4, 1))
    pixels = cloud_pixels(confidence)
    assert set(pixels[:, 0]) == {5, 6, 7, 8, 9}
    assert len(pixels) == 20


def test_point_cloud_is_a_binary_ply_with_byte_colours(tmp_path):
    path = tmp_path / "points.ply"
    write_point_cloud(path, np.array([[1.0, 2.0, 3.0]]), np.array([[255, 128, 0]], dtype=np.uint8))
    data = path.read_bytes()
    header, body = data.split(b"end_header\n")
    assert b"element vertex 1" in header and b"property uchar red" in header
    assert np.frombuffer(body[:12], "<f4").tolist() == [1.0, 2.0, 3.0]
    assert list(body[12:]) == [255, 128, 0]
