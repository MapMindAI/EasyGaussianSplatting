"""The per-image depth supervision files the depth stage writes.

Each "<image_name>.npy" holds one row per supervised pixel: x and y normalized
to [0, 1] as uint16 fixed point, so the file survives whatever resolution the
trainer loads the images at, and depth as float16 in the frame of the input
model. That is 6 bytes a row, half of float32 (x, y, depth); x and y still
round back to the exact pixel for images up to 65536 pixels across.

Run as a script, it converts a folder of float32 (x, y, depth) files in place.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROW = np.dtype([("x", "<u2"), ("y", "<u2"), ("depth", "<f2")])
_COORDINATE_SCALE = np.iinfo(np.uint16).max


def encode(rows):
    """(M, 3) float (x, y, depth) rows as ROW records."""
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, 3)
    encoded = np.empty(len(rows), dtype=ROW)
    encoded["x"] = np.rint(np.clip(rows[:, 0], 0, 1) * _COORDINATE_SCALE)
    encoded["y"] = np.rint(np.clip(rows[:, 1], 0, 1) * _COORDINATE_SCALE)
    if not (np.abs(rows[:, 2]) <= np.finfo(np.float16).max).all():
        raise ValueError("depth beyond float16's range of 65504")
    encoded["depth"] = rows[:, 2]
    return encoded


def decode(encoded):
    """ROW records as (M, 3) float32 (x, y, depth) rows."""
    return np.column_stack([
        encoded["x"] / np.float32(_COORDINATE_SCALE),
        encoded["y"] / np.float32(_COORDINATE_SCALE),
        encoded["depth"],
    ]).astype(np.float32)


def save_rows(path, rows):
    """Writes (M, 3) (x, y, depth) rows through a temporary file, so an
    interrupted run leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.partial")
    with open(temporary_path, "wb") as file:
        np.save(file, encode(rows))
    os.replace(temporary_path, path)


def load_rows(path):
    """(M, 3) float32 (x, y, depth) rows from a save_rows file."""
    encoded = np.load(path)
    if encoded.dtype != ROW:
        raise ValueError(
            f"{path} is in the old float32 format; convert its folder with "
            f"python3 -m mapping.depth.depth_rows <depth_dir>"
        )
    return decode(encoded)


def convert(path):
    """Rewrites a float32 (x, y, depth) file as ROW records; returns whether it did."""
    rows = np.load(path)
    if rows.dtype == ROW:
        return False
    save_rows(path, rows)
    return True


def main(arguments=None):
    from mapping.file_listing import files_under

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("depth_dir", type=Path)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args(arguments)

    paths = [path for path in files_under(args.depth_dir) if path.suffix == ".npy"]
    with ThreadPoolExecutor(args.num_threads) as pool:
        converted = sum(pool.map(convert, paths))
    print(f"Converted {converted} of {len(paths)} depth files in {args.depth_dir}")


if __name__ == "__main__":
    main()
