import numpy as np

from mapping.depth.cross_view_fusion import View, covisible_neighbours, fused_depth, fusion_order


def test_neighbours_are_the_frames_sharing_the_most_points():
    neighbours = covisible_neighbours(
        [np.arange(0, 30), np.arange(5, 30), np.arange(20, 50), np.arange(100, 120)], 200
    )

    assert neighbours == [[1, 2], [0, 2], [0, 1], []]


def test_neighbours_sharing_as_many_points_keep_frame_order():
    neighbours = covisible_neighbours([np.arange(0, 20), np.arange(10, 20), np.arange(0, 10)], 20)

    assert neighbours[0] == [1, 2]


def test_fusion_order_takes_each_frame_across_its_faces():
    names = ["front/000002.jpg", "back/000001.jpg", "front/000001.jpg", "back/000002.jpg"]

    assert fusion_order(names) == [
        "back/000001.jpg", "front/000001.jpg", "back/000002.jpg", "front/000002.jpg"
    ]


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
