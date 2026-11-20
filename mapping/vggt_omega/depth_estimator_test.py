"""Tests for decoding VGGT-Omega's pose encoding."""

import numpy as np

from mapping.vggt_omega.depth_estimator import decode_pose, quaternion_to_matrix


def test_quaternion_is_scalar_last():
    np.testing.assert_allclose(quaternion_to_matrix([0, 0, 0, 1]), np.eye(3))
    half_turn_about_z = [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    np.testing.assert_allclose(
        quaternion_to_matrix(half_turn_about_z) @ [1, 0, 0], [0, 1, 0], atol=1e-12
    )


def test_intrinsics_follow_each_field_of_view_at_the_image_size():
    encoding = np.array([1, 2, 3, 0, 0, 0, 1, np.pi / 2, np.pi / 3])
    camera_from_world, intrinsics = decode_pose(encoding, 300, 400)
    np.testing.assert_allclose(camera_from_world, np.column_stack([np.eye(3), [1, 2, 3]]))
    np.testing.assert_allclose(intrinsics[0], [200 / np.tan(np.pi / 6), 0, 200])
    np.testing.assert_allclose(intrinsics[1], [0, 150, 150])
