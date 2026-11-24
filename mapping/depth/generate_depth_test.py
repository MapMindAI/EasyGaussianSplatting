"""Tests for depth grouping inputs, sampling, and COLMAP observations."""

from pathlib import Path

import numpy as np
import pytest

from mapping.depth.cross_view_fusion import SUMMARY_NAME
from mapping.depth.depth_maps import depth_path, load_depth_map, save_depth_map
from mapping.depth.generate_depth import (
    Observations,
    depth_debug_path,
    depth_groups,
    depth_overlay,
    observed_depths,
    pending_groups,
    sample_at,
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


def test_sample_at_reads_the_map_cell_under_each_colmap_pixel():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)

    # A 30 x 40 image over a 3 x 4 map: each cell covers 10 x 10 pixels.
    points = np.array([[0.5, 0.5], [9.9, 9.9], [10.0, 25.0], [39.5, 29.5]])
    assert sample_at(values, points, (30, 40)).tolist() == [0.0, 0.0, 9.0, 11.0]


def test_sample_at_clamps_points_outside_the_image():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)

    assert sample_at(values, np.array([[-5.0, -5.0], [99.0, 99.0]]), (3, 4)).tolist() == [0.0, 11.0]


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


def test_pending_groups_keep_any_group_with_an_image_still_missing():
    groups = [["a", "b"], ["b", "c"], ["c", "d"]]

    assert pending_groups(groups, {"a", "b", "c"}) == [["c", "d"]]
    assert pending_groups(groups, set()) == groups


def test_generate_depths_resumes_missing_groups_and_leaves_skipped_images_empty(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    import cv2

    from mapping.depth import generate_depth

    names = ["front/a.jpg", "front/b.jpg", "front/c.jpg", "up/d.jpg"]
    (tmp_path / "images" / "front").mkdir(parents=True)
    (tmp_path / "images" / "up").mkdir()
    for name in names:
        cv2.imwrite(str(tmp_path / "images" / name), np.full((4, 4, 3), 128, np.uint8))
    observation = Observations(np.eye(3, 4), np.zeros((1, 2)), np.ones(1), np.zeros(1, np.int64))
    reconstruction = SimpleNamespace(images={
        index: SimpleNamespace(name=name) for index, name in enumerate(names)
    })
    monkeypatch.setitem(sys.modules, "pycolmap", SimpleNamespace(Reconstruction=lambda path: reconstruction))

    class Progress(list):
        set_postfix = staticmethod(lambda **values: None)
        write = staticmethod(print)

    monkeypatch.setitem(sys.modules, "tqdm", SimpleNamespace(tqdm=lambda items, **options: Progress(items)))
    monkeypatch.setattr(generate_depth, "observed_depths", lambda reconstruction: (
        np.zeros((1, 3)), {name: observation for name in names}
    ))
    # "c" is in no group and "up/d" is an up face; neither is ever predicted.
    monkeypatch.setattr(generate_depth, "depth_groups", lambda *arguments: (
        [["front/a.jpg", "front/b.jpg"]], ["front/c.jpg"]
    ))
    monkeypatch.setattr(generate_depth, "pose_depth_scale", lambda *arguments: 1.0)
    monkeypatch.setattr(generate_depth, "align_sparse_depth", lambda *arguments: 1.0)
    runs = []

    class Estimator:
        def run(self, images):
            runs.append(len(images))
            return {
                "depth_list": [np.ones((4, 4), np.float32)] * len(images),
                "depth_conf_list": [np.ones((4, 4), np.float32)] * len(images),
                "camera_from_world_list": [np.eye(3, 4)] * len(images),
            }

    depth_dir = tmp_path / "depths"
    args = SimpleNamespace(
        model_dir=tmp_path, depth_dir=depth_dir, overwrite=False, mask_dir=None, debug_dir=None,
        num_threads=1, min_confidence=0.5, min_depth=0.1, max_depth=5.0,
    )
    # An interrupted run already wrote "a", and fused it.
    save_depth_map(depth_path(depth_dir, "front/a.jpg"), np.full((4, 4), 7.0), np.ones((4, 4)), 2.0)
    (depth_dir / SUMMARY_NAME).write_text("{}")

    assert generate_depth.generate_depths(args, Estimator, 2, 2) == 1

    assert runs == [2]
    assert not (depth_dir / SUMMARY_NAME).exists()  # New depths still need fusing.
    assert load_depth_map(depth_path(depth_dir, "front/a.jpg")).scale == 2.0
    written = load_depth_map(depth_path(depth_dir, "front/b.jpg"))
    np.testing.assert_array_equal(written.depth, np.ones((4, 4)))
    np.testing.assert_array_equal(written.confidence, np.ones((4, 4)))
    assert load_depth_map(depth_path(depth_dir, "front/c.jpg")) is None
    assert load_depth_map(depth_path(depth_dir, "up/d.jpg")) is None

    assert generate_depth.generate_depths(args, Estimator, 2, 2) == 0
    assert runs == [2]
