import numpy as np
import pytest

from mapping.tsdf.common import (
    camera_manifest_path,
    depth_path,
    latest_point_cloud,
    progress_bar,
    valid_depth,
)
from mapping.tsdf.render_depth import _camera_matrix, _read_ply, _world_to_camera


def test_depth_path_keeps_cube_face_directory(tmp_path):
    assert depth_path(tmp_path, "front/frame.png") == tmp_path / "front/frame.npy"
    assert camera_manifest_path(tmp_path) == tmp_path / "cameras.json"


def test_valid_depth_requires_coverage_and_a_positive_finite_distance():
    depth = np.array([1.0, 0.0, np.nan, 2.0])
    alpha = np.array([0.5, 1.0, 1.0, 0.4])
    assert valid_depth(depth, alpha, 0.5).tolist() == [True, False, False, False]


def test_progress_bar_reaches_one_hundred_percent():
    assert progress_bar("Fused TSDF", 5, 5, width=5) == "\rFused TSDF: [#####] 100%"


def test_latest_point_cloud_requires_a_gsplat_export(tmp_path):
    with pytest.raises(FileNotFoundError):
        latest_point_cloud(tmp_path)
    ply_directory = tmp_path / "ply"
    ply_directory.mkdir()
    (ply_directory / "point_cloud_100.ply").touch()
    (ply_directory / "point_cloud_20.ply").touch()
    assert latest_point_cloud(tmp_path).name == "point_cloud_100.ply"


class _Camera:
    class model:
        name = "PINHOLE"

    params = [100.0, 101.0, 50.0, 51.0]


class _Pose:
    def matrix(self):
        return np.arange(12).reshape(3, 4)


class _Image:
    cam_from_world = _Pose()


def test_cube_camera_matrix_and_pose_keep_colmap_conventions():
    assert np.array_equal(
        _camera_matrix(_Camera()),
        [[100.0, 0.0, 50.0], [0.0, 101.0, 51.0], [0.0, 0.0, 1.0]],
    )
    assert np.array_equal(_world_to_camera(_Image())[:3], np.arange(12).reshape(3, 4))


def test_reads_the_binary_gsplat_ply_layout_without_plyfile(tmp_path):
    names = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"]
    names += [f"rot_{index}" for index in range(4)]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
    path = tmp_path / "point_cloud_1.ply"
    path.write_bytes(header.encode() + np.arange(len(names), dtype="<f4").tobytes())

    read_names, values = _read_ply(path)

    assert read_names == names
    assert values.shape == (1, len(names))
    assert values[0, -1] == len(names) - 1
