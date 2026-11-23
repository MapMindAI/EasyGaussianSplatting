import numpy as np
import pytest

from mapping.depth.depth_maps import depth_rows, load_depth_map, save_depth_map, upsample


def test_maps_round_trip_as_float16_and_empty_maps_mean_no_depth(tmp_path):
    path = tmp_path / "front" / "frame.jpg.npy"
    depth = np.array([[0.0, 1.5], [80.0, 1000.0]])

    save_depth_map(path, depth)
    save_depth_map(tmp_path / "empty.npy", np.zeros((0, 0)))

    assert np.load(path).dtype == np.float16
    np.testing.assert_allclose(load_depth_map(path), depth, rtol=1e-3)
    assert load_depth_map(tmp_path / "empty.npy") is None
    assert [entry.name for entry in path.parent.iterdir()] == ["frame.jpg.npy"]


def test_depth_beyond_float16_is_refused(tmp_path):
    with pytest.raises(ValueError, match="float16"):
        save_depth_map(tmp_path / "far.npy", [[1e5]])


def test_row_files_are_refused(tmp_path):
    np.save(tmp_path / "rows.npy", np.zeros((3, 3), dtype=np.float32)[:, :, None])

    with pytest.raises(ValueError, match="--overwrite"):
        load_depth_map(tmp_path / "rows.npy")


def test_upsample_interpolates_between_valid_pixels_only():
    depth = np.array([[1.0, 3.0, 0.0]], dtype=np.float32).repeat(2, axis=0)

    upsampled = upsample(depth, 6, 4)

    # Inside the valid pair the depth blends; nothing blends toward the hole.
    assert upsampled[:, 0].tolist() == [1.0] * 4
    assert 1.0 < upsampled[0, 2] < 3.0
    assert (upsampled[:, 4:] == 0).all()
    assert set(np.unique(upsampled[:, 3])) <= {0.0, 3.0}


def test_depth_rows_normalize_the_upsampled_pixels():
    rows = depth_rows(np.full((2, 2), 4.0, dtype=np.float32), 5, 3)

    assert len(rows) == 15
    assert rows[:, 0].min() == 0 and rows[:, 0].max() == 1
    assert rows[:, 1].min() == 0 and rows[:, 1].max() == 1
    assert (rows[:, 2] == 4).all()
