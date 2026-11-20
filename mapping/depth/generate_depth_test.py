"""Tests for depth grouping inputs, sampling, filtering, and COLMAP observations."""

from pathlib import Path

import numpy as np
import pytest

from mapping.depth.generate_depth import (
    depth_debug_path,
    depth_edge_mask,
    depth_groups,
    depth_overlay,
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
    def __init__(self, name, observations=(), cam_from_world=None):
        self.name = name
        self._observations = list(observations)
        self.cam_from_world = CamFromWorld(
            cam_from_world if cam_from_world is not None else np.eye(4)[:3]
        )

    def get_observation_points2D(self):
        return self._observations


class CallablePoseImage(Image):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._camera_from_world = self.cam_from_world
        del self.cam_from_world

    def cam_from_world(self):
        return self._camera_from_world


class Reconstruction:
    def __init__(self, *images, points3D=None):
        self.images = {index: image for index, image in enumerate(images)}
        self.points3D = points3D or {}


# --- grouping --------------------------------------------------------------


def camera_at(x, y=0.0):
    """A camera at (x, y, 0) looking down +z, as cam_from_world."""
    return np.column_stack([np.eye(3), [-x, -y, 0.0]])


def test_depth_groups_link_cameras_viewing_the_same_points_and_skip_up_and_down():
    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8)), -1).reshape(-1, 2)
    points3D = {
        index: Point3D((x, y, 10.0)) for index, (x, y) in enumerate(grid)
    }
    observations = [Point2D((0.0, 0.0), index) for index in points3D]
    reconstruction = Reconstruction(
        *(
            Image(f"{face}/{index:06d}.jpg", observations=observations,
                  cam_from_world=camera_at(0.3 * index))
            for face in ("front", "up")
            for index in range(4)
        ),
        Image("back/000000.jpg", cam_from_world=camera_at(50.0)),
        points3D=points3D,
    )

    groups, ungrouped = depth_groups(*observed_depths(reconstruction), 3, 2)

    grouped = sorted(name for group in groups for name in group)
    assert set(grouped) == {f"front/{index:06d}.jpg" for index in range(4)}
    assert all(len(group) <= 3 for group in groups)
    assert ungrouped == ["back/000000.jpg"]


# --- sampling --------------------------------------------------------------


def test_sample_at_reads_the_rounded_pixel():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)

    assert sample_at(values, np.array([[0.0, 0.0], [3.4, 2.2]])).tolist() == [0.0, 11.0]


def test_sample_at_clamps_points_outside_the_image():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)

    assert sample_at(values, np.array([[-5.0, -5.0], [99.0, 99.0]])).tolist() == [0.0, 11.0]


def test_supervised_points_normalizes_coordinates():
    depth = np.ones((10, 20), dtype=np.float32)

    rows = supervised_points(
        depth, None, None, 1000, 0.0, None, None, np.random.default_rng(0)
    )

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

    rows = supervised_points(
        depth, confidence, mask, 1000, 2.0, None, None, np.random.default_rng(0)
    )

    assert len(rows) == 4
    assert {round(float(y) * 3) for y in rows[:, 1]} == {3}


def test_supervised_points_subsamples_to_the_requested_count():
    depth = np.ones((50, 50), dtype=np.float32)

    rows = supervised_points(
        depth, None, None, 17, 0.0, None, None, np.random.default_rng(0)
    )

    assert len(rows) == 17


def test_supervised_points_returns_nothing_when_every_pixel_is_rejected():
    rows = supervised_points(
        np.zeros((4, 4), dtype=np.float32), None, None, 10, 0.0, None, None,
        np.random.default_rng(0)
    )

    assert rows.shape == (0, 3)


def test_supervised_points_keeps_depths_inside_the_requested_range():
    depth = np.array([[0.5, 1.0], [3.0, 7.0]], dtype=np.float32)

    rows = supervised_points(
        depth, None, None, 10, 0.0, 1.0, 3.0, np.random.default_rng(0)
    )

    assert rows[:, 2].tolist() == [1.0, 3.0]


def test_depth_edge_mask_rejects_colour_and_depth_boundaries():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, 10:] = 255
    depth = np.ones((20, 20), dtype=np.float32)
    depth[10:] = 2.0

    edges = depth_edge_mask(image, depth)

    assert edges[:, 10].all()
    assert not edges[0, 0]


def test_depth_overlay_colours_depth_without_changing_image_shape():
    image = np.full((4, 4, 3), 100, dtype=np.uint8)
    depth = np.arange(1, 17, dtype=np.float32).reshape(4, 4)

    overlay = depth_overlay(image, depth)

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    assert not np.array_equal(overlay, image)


def test_depth_overlay_leaves_the_image_unchanged_at_zero_alpha():
    image = np.full((2, 2, 3), 100, dtype=np.uint8)

    assert np.array_equal(depth_overlay(image, np.ones((2, 2)), alpha=0.0), image)


def test_depth_overlay_leaves_filtered_pixels_uncoloured():
    image = np.full((2, 2, 3), 100, dtype=np.uint8)
    depth = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    overlay = depth_overlay(image, depth)

    assert np.array_equal(overlay[0, 0], image[0, 0])
    assert not np.array_equal(overlay[0, 1], image[0, 1])


def test_depth_debug_path_replaces_the_image_suffix_with_depth_postfix():
    assert depth_debug_path(Path("debug"), "front/000001.jpg") == Path(
        "debug/front/000001_depth.png"
    )


# --- COLMAP observations ---------------------------------------------------


def test_observed_depths_takes_depth_along_the_camera_axis():
    # Camera looking down +z from the origin, so depth is the point's z.
    reconstruction = Reconstruction(
        Image("a.jpg", observations=[Point2D((10.0, 20.0), 7)]),
        points3D={7: Point3D((1.0, 2.0, 30.0))},
    )

    points, observations = observed_depths(reconstruction)
    observation = observations["a.jpg"]

    assert observation.pixels.tolist() == [[10.0, 20.0]]
    assert observation.depths.tolist() == [30.0]
    assert points[observation.point_indices].tolist() == [[1.0, 2.0, 30.0]]


def test_observed_depths_gives_an_image_without_observations_empty_arrays():
    observation = observed_depths(Reconstruction(Image("a.jpg")))[1]["a.jpg"]

    assert observation.pixels.shape == (0, 2) and observation.depths.shape == (0,)


def test_observed_depths_accepts_callable_camera_from_world():
    reconstruction = Reconstruction(
        CallablePoseImage("a.jpg", observations=[Point2D((10.0, 20.0), 7)]),
        points3D={7: Point3D((1.0, 2.0, 30.0))},
    )

    assert observed_depths(reconstruction)[1]["a.jpg"].depths.tolist() == [30.0]
