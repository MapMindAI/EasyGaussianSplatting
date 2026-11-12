import os

import numpy as np

from mapping import split_ply

PREAMBLE = ["ply", "format binary_little_endian 1.0"]
PROPERTIES = ["property float x", "property float y", "property float z", "property float opacity"]


def write_grid(path, width, height):
    rows = np.array([[x, y, 0.0, x * 100 + y] for x in range(width) for y in range(height)], dtype="<f4")
    split_ply.write_ply(path, PREAMBLE, PROPERTIES, rows)
    return rows


def test_read_ply_round_trips(tmp_path):
    rows = write_grid(tmp_path / "grid.ply", 4, 3)
    preamble, properties, read = split_ply.read_ply(tmp_path / "grid.ply")
    assert (preamble, properties) == (PREAMBLE, PROPERTIES)
    assert np.array_equal(read, rows)


def test_bisect_splits_the_longer_axis_first():
    xy = np.array([[x, y] for x in range(8) for y in range(2)], dtype=float)
    parts = split_ply.bisect(xy, max_rows=4)
    assert [sorted(set(xy[part, 0])) for part in parts] == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_split_ply_keeps_every_gaussian_under_the_limit(tmp_path):
    rows = write_grid(tmp_path / "grid.ply", 10, 10)
    max_bytes = os.path.getsize(tmp_path / "grid.ply") // 2
    written = split_ply.split_ply(str(tmp_path / "grid.ply"), str(tmp_path / "parts"), max_bytes)
    assert len(written) == 4 and all(os.path.getsize(path) <= max_bytes for path in written)
    merged = np.concatenate([split_ply.read_ply(path)[2] for path in written])
    assert sorted(merged[:, 3]) == sorted(rows[:, 3])


def test_split_ply_skips_a_small_file(tmp_path):
    write_grid(tmp_path / "grid.ply", 2, 2)
    assert split_ply.split_ply(str(tmp_path / "grid.ply"), str(tmp_path / "parts")) == []
    assert not (tmp_path / "parts").exists()
