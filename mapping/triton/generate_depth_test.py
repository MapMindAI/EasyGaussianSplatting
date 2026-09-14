"""Tests for the grouping, scale fitting, and sampling behind DA3 depth."""

import numpy as np
import pytest

from mapping.triton.generate_depth import (
    camera_groups,
    fit_depth_scale,
    group_images,
    observed_depths,
    sample_at,
    supervised_points,
)


class Point2D:
    def __init__(self, xy, point3D_id):
        self.xy, self.point3D_id = np.asarray(xy, float), point3D_id


class Point3D:
    def __init__(self, xyz):
        self.xyz = np.asarray(xyz, float)


class CamFromWorld:
    def __init__(self, matrix):
        self._matrix = np.asarray(matrix, float)

    def matrix(self):
        return self._matrix


class Image:
    def __init__(self, name, camera_id=1, observations=(), cam_from_world=None):
        self.name, self.camera_id = name, camera_id
        self._observations = list(observations)
        self.cam_from_world = CamFromWorld(
            cam_from_world if cam_from_world is not None else np.eye(4)[:3]
        )

    def get_observation_points2D(self):
        return self._observations


class Reconstruction:
    def __init__(self, *images, points3D=None):
        self.images = {index: image for index, image in enumerate(images)}
        self.points3D = points3D or {}


# --- grouping --------------------------------------------------------------


def test_group_images_splits_into_consecutive_groups():
    assert group_images(list(range(10)), 5) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]


def test_a_trailing_short_group_grows_backwards_instead_of_being_padded():
    groups = group_images(list(range(12)), 5)

    assert [len(group) for group in groups] == [5, 5, 5]
    assert groups[-1] == [7, 8, 9, 10, 11]


def test_too_few_images_to_fill_one_group_yields_nothing():
    assert group_images(list(range(4)), 5) == []


def test_camera_groups_never_mixes_cameras_and_orders_by_name():
    reconstruction = Reconstruction(
        *(Image(f"front/{index:06d}.jpg", camera_id=1) for index in (2, 0, 1)),
        *(Image(f"back/{index:06d}.jpg", camera_id=2) for index in range(3)),
    )

    groups = camera_groups(reconstruction, 3)

    assert groups == [
        ["front/000000.jpg", "front/000001.jpg", "front/000002.jpg"],
        ["back/000000.jpg", "back/000001.jpg", "back/000002.jpg"],
    ]


# --- scale -----------------------------------------------------------------


def test_fit_depth_scale_recovers_a_known_ratio():
    predicted = np.linspace(1.0, 5.0, 50)

    assert fit_depth_scale(predicted, predicted * 3.0) == pytest.approx(3.0)


def test_fit_depth_scale_ignores_a_minority_of_wild_outliers():
    predicted = np.full(60, 2.0)
    reference = np.full(60, 6.0)
    reference[:20] = 5000.0

    assert fit_depth_scale(predicted, reference) == pytest.approx(3.0)


def test_fit_depth_scale_refuses_too_few_points():
    assert fit_depth_scale(np.full(5, 2.0), np.full(5, 6.0)) is None


def test_fit_depth_scale_ignores_non_positive_and_non_finite_depths():
    predicted = np.concatenate([np.full(30, 2.0), np.array([0.0, -1.0, np.nan])])
    reference = np.concatenate([np.full(30, 6.0), np.array([9.0, 9.0, 9.0])])

    assert fit_depth_scale(predicted, reference) == pytest.approx(3.0)


# --- sampling --------------------------------------------------------------


def test_sample_at_reads_the_rounded_pixel():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)

    assert sample_at(values, np.array([[0.0, 0.0], [3.4, 2.2]])).tolist() == [0.0, 11.0]


def test_sample_at_clamps_points_outside_the_image():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)

    assert sample_at(values, np.array([[-5.0, -5.0], [99.0, 99.0]])).tolist() == [0.0, 11.0]


def test_supervised_points_normalizes_coordinates():
    depth = np.ones((10, 20), dtype=np.float32)

    rows = supervised_points(depth, None, None, 1000, 0.0, np.random.default_rng(0))

    assert len(rows) == 200
    assert rows[:, 0].min() == 0.0 and rows[:, 0].max() == pytest.approx(1.0)
    assert rows[:, 1].min() == 0.0 and rows[:, 1].max() == pytest.approx(1.0)


def test_supervised_points_drops_unconfident_masked_and_empty_depth():
    depth = np.ones((4, 4), dtype=np.float32)
    depth[0] = 0.0
    confidence = np.full((4, 4), 5.0, dtype=np.float32)
    confidence[1] = 0.5
    mask = np.ones((4, 4), dtype=bool)
    mask[2] = False

    rows = supervised_points(depth, confidence, mask, 1000, 2.0, np.random.default_rng(0))

    assert len(rows) == 4
    assert {round(float(y) * 3) for y in rows[:, 1]} == {3}


def test_supervised_points_subsamples_to_the_requested_count():
    depth = np.ones((50, 50), dtype=np.float32)

    rows = supervised_points(depth, None, None, 17, 0.0, np.random.default_rng(0))

    assert len(rows) == 17


def test_supervised_points_returns_nothing_when_every_pixel_is_rejected():
    rows = supervised_points(
        np.zeros((4, 4), dtype=np.float32), None, None, 10, 0.0, np.random.default_rng(0)
    )

    assert rows.shape == (0, 3)


# --- COLMAP observations ---------------------------------------------------


def test_observed_depths_takes_depth_along_the_camera_axis():
    # Camera looking down +z from the origin, so depth is the point's z.
    reconstruction = Reconstruction(
        Image("a.jpg", observations=[Point2D((10.0, 20.0), 7)]),
        points3D={7: Point3D((1.0, 2.0, 30.0))},
    )

    points, depths = observed_depths(reconstruction)["a.jpg"]

    assert points.tolist() == [[10.0, 20.0]]
    assert depths.tolist() == [30.0]


def test_observed_depths_gives_an_image_without_observations_empty_arrays():
    points, depths = observed_depths(Reconstruction(Image("a.jpg")))["a.jpg"]

    assert points.shape == (0, 2) and depths.shape == (0,)
