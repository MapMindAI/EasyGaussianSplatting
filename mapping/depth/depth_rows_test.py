import numpy as np
import pytest

from mapping.depth.depth_rows import ROW, convert, load_rows, save_rows


def test_rows_round_trip_to_the_same_pixels_at_half_the_size(tmp_path):
    width, height = 1920, 1080
    columns, lines = np.array([0, 1, 957, 1919]), np.array([0, 539, 540, 1079])
    rows = np.column_stack([columns / (width - 1), lines / (height - 1), [0.5, 3.25, 80.0, 1000.0]])
    path = tmp_path / "front" / "frame.jpg.npy"

    save_rows(path, rows)

    loaded = load_rows(path)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(np.rint(loaded[:, 0] * (width - 1)), columns)
    np.testing.assert_array_equal(np.rint(loaded[:, 1] * (height - 1)), lines)
    np.testing.assert_allclose(loaded[:, 2], rows[:, 2], rtol=1e-3)
    assert np.load(path).nbytes == 6 * len(rows)
    assert [entry.name for entry in path.parent.iterdir()] == ["frame.jpg.npy"]


def test_empty_rows_round_trip(tmp_path):
    save_rows(tmp_path / "empty.npy", np.zeros((0, 3)))

    assert load_rows(tmp_path / "empty.npy").shape == (0, 3)


def test_depth_beyond_float16_is_refused(tmp_path):
    with pytest.raises(ValueError, match="float16"):
        save_rows(tmp_path / "far.npy", [[0.5, 0.5, 1e5]])


def test_old_float32_files_are_refused_until_converted(tmp_path):
    path = tmp_path / "old.npy"
    np.save(path, np.array([[0.25, 0.75, 4.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="mapping.depth.depth_rows"):
        load_rows(path)

    assert convert(path)
    assert np.load(path).dtype == ROW
    np.testing.assert_allclose(load_rows(path), [[0.25, 0.75, 4.0]], atol=1e-4)
    assert not convert(path)
