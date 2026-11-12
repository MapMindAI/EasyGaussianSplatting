#!/usr/bin/env python3
"""Bisect a Gaussian splat PLY into files under a size limit.

Splits at the median of the longer horizontal axis, over the middle 90% of
the means so far outliers don't pick it, until every part is under the limit.
A PLY already under it is left alone.

Usage: python3 -m mapping.split_ply <point_cloud.ply> <output_dir> [--max-bytes N]
"""
import argparse
import os

import numpy as np

# Browser viewers such as SuperSplat cannot load a file of 2 GiB or more.
MAX_BYTES = 2**31 - 1


def read_ply(path):
    """The header lines before `element vertex`, the property lines, and the rows as float32."""
    with open(path, "rb") as file:
        lines = []
        while not lines or lines[-1] != "end_header":
            lines.append(file.readline().decode("ascii").strip())
        offset = file.tell()
    assert "format binary_little_endian 1.0" in lines, f"{path} is not a binary little-endian PLY"
    vertex = next(index for index, line in enumerate(lines) if line.startswith("element vertex "))
    properties = lines[vertex + 1:-1]
    assert all(line.startswith("property float ") for line in properties), f"{path} has non-float properties"
    count = int(lines[vertex].split()[-1])
    rows = np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(count, len(properties)))
    return lines[:vertex], properties, rows


def write_ply(path, preamble, properties, rows):
    header = preamble + [f"element vertex {len(rows)}"] + properties + ["end_header"]
    with open(path, "wb") as file:
        file.write(("\n".join(header) + "\n").encode("ascii"))
        file.write(np.ascontiguousarray(rows, dtype="<f4").tobytes())


def bisect(xy, max_rows):
    """Index arrays of parts with at most `max_rows` rows each."""
    parts, pending = [], [np.arange(len(xy))]
    while pending:
        indices = pending.pop()
        if len(indices) <= max_rows:
            parts.append(indices)
            continue
        low, high = np.percentile(xy[indices], [5, 95], axis=0)
        axis = int(np.argmax(high - low))
        order = indices[np.argsort(xy[indices, axis], kind="stable")]
        middle = len(order) // 2
        pending += [order[middle:], order[:middle]]
    return parts


def split_ply(path, output_dir, max_bytes=MAX_BYTES):
    """Write the parts of `path` as <stem>_NN.ply in output_dir; the paths written."""
    if os.path.getsize(path) <= max_bytes:
        return []
    preamble, properties, rows = read_ply(path)
    header_bytes = sum(len(line) + 1 for line in preamble + properties) + 64
    max_rows = (max_bytes - header_bytes) // rows.strides[0]
    xy = np.array(rows[:, :2])
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    written = []
    for number, indices in enumerate(bisect(xy, max_rows)):
        written.append(os.path.join(output_dir, f"{stem}_{number:02d}.ply"))
        write_ply(written[-1], preamble, properties, rows[np.sort(indices)])
        print(f"{written[-1]}: {len(indices)} Gaussians", flush=True)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ply")
    parser.add_argument("output_dir")
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    arguments = parser.parse_args()
    split_ply(arguments.ply, arguments.output_dir, arguments.max_bytes)


if __name__ == "__main__":
    main()
