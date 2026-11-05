#!/usr/bin/env python3
"""Partition a sparse model into training blocks.

Every method gives each SfM point an owning block, standing in for the coarse
Gaussians a block keeps at merge:

* grid: rows x columns over the points' XY, split at count quantiles.
* tree: recursive median splits along the wider horizontal axis, as
  Hierarchical 3DGS builds its hierarchy, kept to one layer of leaves.
* graph: recursive spectral bisection of the frame covisibility graph; a point
  goes to the block most of its observing frames belong to.

A rig frame then trains every block owning at least `coverage` of the points it
observes, and always the block owning most of them. Writes `<method>.json` and
`<method>.png` per method, and the covisibility map as `covisibility.png`.
"""
import argparse
import json
import logging
import math
import os

import numpy as np
import pycolmap
from scipy import sparse


def load_model(reconstruction):
    """Frame centers, image names per frame, point positions, and their incidence.

    The incidence is a binary (points, frames) matrix: a point observed by
    several faces of one frame counts once.
    """
    frame_ids = sorted(frame_id for frame_id, frame in reconstruction.frames.items() if frame.has_pose)
    column = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    centers = np.zeros((len(frame_ids), 3))
    image_counts = np.zeros(len(frame_ids))
    image_names = [[] for _ in frame_ids]
    for image in reconstruction.images.values():
        if image.frame_id not in column or not image.has_pose:
            continue
        index = column[image.frame_id]
        centers[index] += image.projection_center()
        image_counts[index] += 1
        image_names[index].append(image.name)
    centers /= image_counts[:, None]

    point_ids = sorted(reconstruction.points3D)
    points = np.array([reconstruction.points3D[point_id].xyz for point_id in point_ids])
    rows, columns = [], []
    for row, point_id in enumerate(point_ids):
        for element in reconstruction.points3D[point_id].track.elements:
            rows.append(row)
            columns.append(column[reconstruction.images[element.image_id].frame_id])
    incidence = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(len(point_ids), len(frame_ids))
    )
    incidence.data[:] = 1.0  # Construction summed the faces of one frame.
    return centers, [sorted(names) for names in image_names], points, incidence


def covisibility(incidence):
    """Dense (frames, frames) count of shared points, with a zero diagonal."""
    shared = (incidence.T @ incidence).toarray()
    np.fill_diagonal(shared, 0.0)
    return shared


def bisect_recursively(indices, blocks, score):
    """Split `indices` into `blocks` balanced parts by ordering on `score`.

    `score(indices)` returns one value per index; each split cuts the order at
    the share of blocks each side receives, so counts stay balanced when
    `blocks` is not a power of two.
    """
    if blocks == 1:
        return [indices]
    left_blocks = blocks // 2
    order = indices[np.argsort(score(indices), kind="stable")]
    cut = round(len(order) * left_blocks / blocks)
    return bisect_recursively(order[:cut], left_blocks, score) + bisect_recursively(
        order[cut:], blocks - left_blocks, score
    )


def equal_count_runs(indices, blocks, values):
    """`blocks` equal-count runs of `indices` ordered by `values`."""
    order = indices[np.argsort(values[indices], kind="stable")]
    cuts = [round(len(order) * part / blocks) for part in range(blocks + 1)]
    return [order[start:end] for start, end in zip(cuts, cuts[1:])]


def labels_from_parts(parts, count):
    labels = np.empty(count, dtype=int)
    for label, part in enumerate(parts):
        labels[part] = label
    return labels


def grid_shape(positions, blocks):
    """The rows x columns factorization of `blocks` closest to the middle 90%'s aspect."""
    low, high = np.percentile(positions[:, :2], [5, 95], axis=0)
    aspect = (high[0] - low[0]) / max(high[1] - low[1], 1e-9)
    columns = min(
        (columns for columns in range(1, blocks + 1) if blocks % columns == 0),
        key=lambda columns: abs(math.log(columns * columns / blocks / aspect)),
    )
    return blocks // columns, columns


def partition_grid(positions, rows, columns):
    """Split X into `columns` slabs at count quantiles, then each slab's Y into `rows`."""
    parts = []
    for slab in equal_count_runs(np.arange(len(positions)), columns, positions[:, 0]):
        parts += equal_count_runs(slab, rows, positions[:, 1])
    return labels_from_parts(parts, len(positions))


def partition_tree(positions, blocks):
    def wider_axis(indices):
        low, high = np.percentile(positions[indices, :2], [5, 95], axis=0)
        return positions[indices, int(np.argmax(high - low))]

    parts = bisect_recursively(np.arange(len(positions)), blocks, wider_axis)
    return labels_from_parts(parts, len(positions))


def fiedler_vector(weights):
    """Shi-Malik bisection vector of a weighted graph's normalized Laplacian.

    The second-smallest eigenvector of I - D^-1/2 W D^-1/2, mapped back through
    D^-1/2. Isolated vertices get a tiny degree so the scaling stays finite.
    """
    degree = np.maximum(weights.sum(axis=1), 1e-9)
    scale = 1.0 / np.sqrt(degree)
    laplacian = np.eye(len(weights)) - scale[:, None] * weights * scale[None, :]
    _, vectors = np.linalg.eigh(laplacian)
    return vectors[:, 1] * scale


def partition_graph(shared, blocks):
    def spectral_order(indices):
        if len(indices) < 3:
            return np.arange(len(indices), dtype=float)
        return fiedler_vector(shared[np.ix_(indices, indices)])

    parts = bisect_recursively(np.arange(len(shared)), blocks, spectral_order)
    return labels_from_parts(parts, len(shared))


def one_hot(labels, blocks):
    return sparse.csr_matrix(
        (np.ones(len(labels)), (np.arange(len(labels)), labels)), shape=(len(labels), blocks)
    )


def owners_by_vote(frame_labels, incidence):
    """Each point's block: the one most of its observing frames belong to."""
    votes = incidence @ one_hot(frame_labels, frame_labels.max() + 1)
    return np.asarray(votes.argmax(axis=1)).ravel()


def frame_shares(point_owners, incidence, blocks):
    """(frames, blocks) share of each frame's observed points each block owns."""
    seen = (incidence.T @ one_hot(point_owners, blocks)).toarray()
    return seen / np.maximum(seen.sum(axis=1, keepdims=True), 1.0)


def frame_membership(shares, coverage):
    membership = shares >= coverage
    membership[np.arange(len(shares)), shares.argmax(axis=1)] = True
    return membership


def evaluate(point_owners, shares, membership, shared):
    """Balance, duplication, and the covisibility cut between frames' main blocks."""
    frame_labels = shares.argmax(axis=1)
    total = shared.sum()
    cut = shared[frame_labels[:, None] != frame_labels[None, :]].sum()
    training = membership.sum(axis=0)
    return {
        "cut_fraction": float(cut / total) if total else 0.0,
        "owned_points": np.bincount(point_owners, minlength=membership.shape[1]).tolist(),
        "training_frames": training.tolist(),
        "largest_over_mean_training": float(training.max() / training.mean()),
        "duplication": float(training.sum() / len(membership)),
    }


def manifest(method, shares, membership, image_names, metrics):
    frame_labels = shares.argmax(axis=1)
    blocks = []
    for block in range(membership.shape[1]):
        blocks.append(
            {
                "name": f"block_{block:02d}",
                "main_images": [
                    name for frame in np.flatnonzero(frame_labels == block) for name in image_names[frame]
                ],
                "training_images": [
                    name for frame in np.flatnonzero(membership[:, block]) for name in image_names[frame]
                ],
            }
        )
    return {"method": method, "metrics": metrics, "blocks": blocks}


def view_limits(points, centers):
    """Axes limits covering the frames and the middle 98% of the points."""
    low = np.minimum(np.percentile(points[:, :2], 1, axis=0), centers[:, :2].min(axis=0))
    high = np.maximum(np.percentile(points[:, :2], 99, axis=0), centers[:, :2].max(axis=0))
    return low, high


def pyplot_module():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    return pyplot


def plot_covisibility(path, centers, shared, minimum_shared):
    from matplotlib.collections import LineCollection

    pyplot = pyplot_module()
    figure, (map_axes, matrix_axes) = pyplot.subplots(1, 2, figsize=(20, 9))
    first, second = np.nonzero(np.triu(shared >= minimum_shared))
    weights = shared[first, second]
    order = np.argsort(weights)
    segments = np.stack([centers[first, :2], centers[second, :2]], axis=1)[order]
    lines = LineCollection(segments, array=np.log10(weights[order]), cmap="viridis", linewidths=0.3)
    map_axes.add_collection(lines)
    map_axes.scatter(centers[:, 0], centers[:, 1], s=2, c="black")
    figure.colorbar(lines, ax=map_axes, label="log10 shared points")
    map_axes.set_title(f"{len(first)} frame pairs sharing >= {minimum_shared} points")
    map_axes.set_aspect("equal")
    image = matrix_axes.imshow(np.log10(1.0 + shared), cmap="magma", interpolation="nearest")
    figure.colorbar(image, ax=matrix_axes, label="log10(1 + shared points)")
    matrix_axes.set_title("Covisibility by frame id")
    matrix_axes.set_xlabel("frame")
    matrix_axes.set_ylabel("frame")
    figure.savefig(path, dpi=130, bbox_inches="tight")
    pyplot.close(figure)


def plot_partition(path, method, centers, points, point_owners, membership, metrics):
    pyplot = pyplot_module()
    blocks = membership.shape[1]
    colors = pyplot.get_cmap("tab20")(np.arange(blocks) % 20)
    sample = np.random.default_rng(0).permutation(len(points))[:50_000]
    low, high = view_limits(points, centers)
    columns = min(blocks, 4)
    rows = 1 + math.ceil(blocks / columns)
    figure = pyplot.figure(figsize=(4.5 * columns, 4.5 * rows))

    def frame_axes(axes, title):
        axes.set_xlim(low[0], high[0])
        axes.set_ylim(low[1], high[1])
        axes.set_aspect("equal")
        axes.set_title(title)

    overview = figure.add_subplot(rows, 1, 1)
    overview.scatter(points[sample, 0], points[sample, 1], s=0.5, c=colors[point_owners[sample]])
    overview.scatter(centers[:, 0], centers[:, 1], s=1, c="black")
    frame_axes(overview, f"{method}: cut {metrics['cut_fraction']:.3f}, duplication "
                         f"{metrics['duplication']:.2f}, largest/mean {metrics['largest_over_mean_training']:.2f}")
    for block in range(blocks):
        axes = figure.add_subplot(rows, columns, columns + block + 1)
        owned = sample[point_owners[sample] == block]
        axes.scatter(points[sample, 0], points[sample, 1], s=0.2, c="lightgray")
        axes.scatter(points[owned, 0], points[owned, 1], s=0.3, color=colors[block])
        training = membership[:, block]
        axes.scatter(centers[training, 0], centers[training, 1], s=1.5, c="black")
        frame_axes(axes, f"block_{block:02d}: {metrics['owned_points'][block]} points, "
                         f"{training.sum()} frames")
        axes.set_xticks([])
        axes.set_yticks([])
    figure.savefig(path, dpi=120, bbox_inches="tight")
    pyplot.close(figure)


def point_owners_for(method, points, incidence, shared, blocks):
    if method == "grid":
        return partition_grid(points, *grid_shape(points, blocks))
    if method == "tree":
        return partition_tree(points, blocks)
    return owners_by_vote(partition_graph(shared, blocks), incidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sparse_dir", help="COLMAP model directory, e.g. <workspace>/sparse/0")
    parser.add_argument("output_dir")
    parser.add_argument("--method", choices=["grid", "tree", "graph", "all"], default="all")
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--coverage", type=float, default=0.25,
                        help="Share of a frame's points a block must own for the frame to train it.")
    parser.add_argument("--minimum-shared", type=int, default=500,
                        help="Shared points for an edge to appear in covisibility.png.")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    centers, image_names, points, incidence = load_model(pycolmap.Reconstruction(arguments.sparse_dir))
    shared = covisibility(incidence)
    logging.info("%d frames, %d points, %d covisible frame pairs",
                 len(centers), len(points), int(np.count_nonzero(shared) / 2))
    os.makedirs(arguments.output_dir, exist_ok=True)
    plot_covisibility(os.path.join(arguments.output_dir, "covisibility.png"),
                      centers, shared, arguments.minimum_shared)

    methods = ["grid", "tree", "graph"] if arguments.method == "all" else [arguments.method]
    for method in methods:
        point_owners = point_owners_for(method, points, incidence, shared, arguments.blocks)
        shares = frame_shares(point_owners, incidence, arguments.blocks)
        membership = frame_membership(shares, arguments.coverage)
        metrics = evaluate(point_owners, shares, membership, shared)
        logging.info("%s: %s", method, json.dumps(metrics))
        with open(os.path.join(arguments.output_dir, f"{method}.json"), "w") as file:
            json.dump(manifest(method, shares, membership, image_names, metrics), file, indent=1)
        plot_partition(os.path.join(arguments.output_dir, f"{method}.png"),
                       method, centers, points, point_owners, membership, metrics)


if __name__ == "__main__":
    main()
