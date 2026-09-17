import types

import numpy as np

from mapping.gsplat_world_frame import (
    _rotation_quaternion,
    capture_scene_transform,
    depth_supervision_in_normalized_frame,
    export_in_colmap_frame,
    restore_colmap_coordinates,
)


def _quaternion_to_rotation(quaternion):
    """wxyz quaternion(s) to rotation matrix/matrices."""
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)], -1),
        ],
        -2,
    )


def _normalized_quaternions(rng, count):
    quaternions = rng.normal(size=(count, 4))
    return quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)


def _similarity(rotation, scale, translation):
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform


def _depth_from(camera_to_world, point):
    """The z a camera measures for a world point, as COLMAP defines depth."""
    rotation, position = camera_to_world[:3, :3], camera_to_world[:3, 3]
    return (rotation.T @ (point - position))[2]


class _FakeTensor:
    """The sliver of the torch API `export_in_colmap_frame` touches."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def new_tensor(self, array):
        return _FakeTensor(array)


def test_restores_similarity_normalized_parameters():
    rng = np.random.default_rng(0)
    count = 64

    rotation = _quaternion_to_rotation(_normalized_quaternions(rng, 1)[0])
    scale = 3.7
    translation = rng.normal(size=3)
    transform = _similarity(rotation, scale, translation)

    points = rng.normal(size=(count, 3))
    gaussian_rotation = _quaternion_to_rotation(_normalized_quaternions(rng, count))
    gaussian_scale = rng.uniform(0.01, 1.0, size=(count, 3))

    normalized_points = points @ (scale * rotation).T + translation
    normalized_rotation = rotation @ gaussian_rotation
    normalized_scale = scale * gaussian_scale
    normalized_quats = np.stack(
        [_rotation_quaternion(matrix) for matrix in normalized_rotation]
    )

    means, scales, quats = restore_colmap_coordinates(
        transform,
        normalized_points,
        np.log(normalized_scale),
        normalized_quats,
    )

    assert np.allclose(means, points, atol=1e-4)
    assert np.allclose(np.exp(scales), gaussian_scale, rtol=1e-4)
    assert np.allclose(_quaternion_to_rotation(quats), gaussian_rotation, atol=1e-4)


def test_identity_transform_leaves_parameters_unchanged():
    rng = np.random.default_rng(1)
    means = rng.normal(size=(8, 3))
    scales = rng.normal(size=(8, 3))
    quats = _normalized_quaternions(rng, 8)

    restored_means, restored_scales, restored_quats = restore_colmap_coordinates(
        np.eye(4), means, scales, quats
    )

    assert np.allclose(restored_means, means, atol=1e-6)
    assert np.allclose(restored_scales, scales, atol=1e-6)
    assert np.allclose(
        _quaternion_to_rotation(restored_quats),
        _quaternion_to_rotation(quats),
        atol=1e-6,
    )


def test_captured_transform_is_undone_on_export():
    rng = np.random.default_rng(2)
    rotation = _quaternion_to_rotation(_normalized_quaternions(rng, 1)[0])
    scale = 2.0
    translation = rng.normal(size=3)
    transform = _similarity(rotation, scale, translation)

    class StubParser:
        def __init__(self):
            self.transform = transform

    exported = {}

    def original_export(*args, **kwargs):
        exported.update(kwargs)

    exporter = types.SimpleNamespace(export_splats=original_export)
    capture_scene_transform(StubParser)
    export_in_colmap_frame(exporter)
    StubParser()

    count = 16
    points = rng.normal(size=(count, 3))
    normalized_points = points @ (scale * rotation).T + translation
    exporter.export_splats(
        means=_FakeTensor(normalized_points),
        scales=_FakeTensor(np.zeros((count, 3))),
        quats=_FakeTensor(np.tile([1.0, 0.0, 0.0, 0.0], (count, 1))),
        opacities=_FakeTensor(np.zeros(count)),
        sh0=_FakeTensor(np.zeros((count, 1, 3))),
        shN=_FakeTensor(np.zeros((count, 0, 3))),
    )

    assert np.allclose(exported["means"].array, points, atol=1e-4)


def test_depths_reach_the_frame_the_renderer_measures_in():
    """The scale a camera carried through the similarity actually observes.

    Built from the parser's own normalization rather than from the conversion,
    so an inverted scale fails here instead of agreeing with itself.
    """
    rng = np.random.default_rng(3)
    rotation = _quaternion_to_rotation(_normalized_quaternions(rng, 1)[0])
    scale = 5.0
    transform = _similarity(rotation, scale, rng.normal(size=3))

    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = _quaternion_to_rotation(_normalized_quaternions(rng, 1)[0])
    camera_to_world[:3, 3] = rng.normal(size=3)
    point = rng.normal(size=3)

    # transform_cameras applies the similarity and divides the scale back out of
    # the rotation, which is what leaves the camera a rigid pose.
    normalized_camera = transform @ camera_to_world
    normalized_camera[:3, :3] /= scale
    normalized_point = transform[:3, :3] @ point + transform[:3, 3]

    colmap_depth = _depth_from(camera_to_world, point)
    _, depths = depth_supervision_in_normalized_frame(
        transform, [[0.5, 0.5, colmap_depth]], width=9, height=5
    )

    assert np.allclose(depths[0], _depth_from(normalized_camera, normalized_point), rtol=1e-5)
    assert np.allclose(depths[0], colmap_depth * scale, rtol=1e-5)


def test_normalized_coordinates_span_the_loaded_image():
    rows = [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.5, 0.5, 1.0]]

    points, _ = depth_supervision_in_normalized_frame(np.eye(4), rows, width=9, height=5)

    assert points.tolist() == [[0.0, 0.0], [8.0, 4.0], [4.0, 2.0]]


def test_identity_transform_leaves_depths_unchanged():
    rows = [[0.25, 0.75, 2.5], [0.5, 0.5, 40.0]]

    _, depths = depth_supervision_in_normalized_frame(np.eye(4), rows, width=4, height=4)

    assert np.allclose(depths, [2.5, 40.0])


def test_an_image_generate_depth_skipped_supervises_on_nothing():
    """A missing .npy reaches the conversion as no rows at all."""
    points, depths = depth_supervision_in_normalized_frame(
        _similarity(np.eye(3), 2.0, np.zeros(3)),
        np.zeros((0, 3), dtype=np.float32),
        width=9,
        height=5,
    )

    assert points.shape == (0, 2)
    assert depths.shape == (0,)
