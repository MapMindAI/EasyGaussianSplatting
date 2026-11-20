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


def test_run_infers_one_group_at_a_time_across_threads():
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from mapping.vggt_omega.depth_estimator import VggtOmegaDepthEstimator

    active, peak = [0], [0]
    counter_lock = threading.Lock()

    class Session:
        def run(self, names, feeds):
            with counter_lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with counter_lock:
                active[0] -= 1
            count = feeds["images"].shape[1]
            return (
                np.ones((1, count, 4, 4, 1), np.float32),
                np.ones((1, count, 4, 4), np.float32),
                np.tile([0, 0, 0, 0, 0, 0, 1, 1, 1], (1, count, 1)).astype(np.float32),
            )

    estimator = VggtOmegaDepthEstimator.__new__(VggtOmegaDepthEstimator)
    estimator._session, estimator._input_name = Session(), "images"
    estimator._height = estimator._width = 4
    estimator._run_lock = threading.Lock()
    images = [np.zeros((8, 8, 3), np.uint8)] * 2

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda _: estimator.run(images), range(3)))

    assert peak[0] == 1
