#!/usr/bin/env python3
"""Partition a sparse model into training blocks.

Every method gives each SfM point an owning block, standing in for the coarse
Gaussians a block keeps at merge:

* grid: rows x columns over the points' XY, split at count quantiles.
* tree: recursive median splits along the wider horizontal axis, as
  Hierarchical 3DGS builds its hierarchy, kept to one layer of leaves. With
  `--max-images`, it keeps splitting the leaf training the most images instead
  of stopping at `--blocks`.
* graph: recursive spectral bisection of the rig-frame covisibility graph; a
  point goes to the block most of its observing frames belong to.

An image then trains every block whose points cover at least `--visibility` of
it, and the block covering most of it. Writes `<method>.json` and
`<method>.png` per method, the frame covisibility map as `covisibility.png`,
and every method's top view side by side as `comparison.png`.
"""
import argparse
import dataclasses
import json
import logging
import math
import os

import numpy as np
import pycolmap
from scipy import sparse


@dataclasses.dataclass
class Model:
    frame_centers: np.ndarray  # (frames, 3)
    image_names: list
    image_frames: np.ndarray  # (images,) frame index of each image
    projections: np.ndarray  # (images, 3, 4) world to pixel
    image_sizes: np.ndarray  # (images, 2) width, height
    points: np.ndarray  # (points, 3)
    incidence: sparse.csr_matrix  # (points, frames), binary


def load_model(reconstruction):
    """The registered images and rig frames, and the points with their frame incidence.

    A point observed by several faces of one frame counts once in the incidence.
    """
    frame_ids = sorted(frame_id for frame_id, frame in reconstruction.frames.items() if frame.has_pose)
    column = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    images = sorted(
        (image for image in reconstruction.images.values() if image.has_pose and image.frame_id in column),
        key=lambda image: image.name,
    )
    image_frames = np.array([column[image.frame_id] for image in images])
    frame_centers = np.zeros((len(frame_ids), 3))
    np.add.at(frame_centers, image_frames, np.array([image.projection_center() for image in images]))
    frame_centers /= np.bincount(image_frames, minlength=len(frame_ids))[:, None]
    cameras = [reconstruction.cameras[image.camera_id] for image in images]
    projections = np.array([
        camera.calibration_matrix() @ image.cam_from_world().matrix()
        for image, camera in zip(images, cameras)
    ])

    point_ids = sorted(reconstruction.points3D)
    rows, columns = [], []
    for row, point_id in enumerate(point_ids):
        for element in reconstruction.points3D[point_id].track.elements:
            rows.append(row)
            columns.append(column[reconstruction.images[element.image_id].frame_id])
    incidence = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(len(point_ids), len(frame_ids))
    )
    incidence.data[:] = 1.0  # Construction summed the faces of one frame.
    return Model(
        frame_centers=frame_centers,
        image_names=[image.name for image in images],
        image_frames=image_frames,
        projections=projections,
        image_sizes=np.array([(camera.width, camera.height) for camera in cameras]),
        points=np.array([reconstruction.points3D[point_id].xyz for point_id in point_ids]),
        incidence=incidence,
    )


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


def along_wider_axis(positions, indices):
    """Coordinates of `indices` along the axis their middle 90% spans furthest."""
    low, high = np.percentile(positions[indices, :2], [5, 95], axis=0)
    return positions[indices, int(np.argmax(high - low))]


def partition_tree(positions, blocks):
    parts = bisect_recursively(
        np.arange(len(positions)), blocks, lambda indices: along_wider_axis(positions, indices)
    )
    return labels_from_parts(parts, len(positions))


def partition_tree_by_images(positions, coverage, max_images, visibility, minimum_points=1000):
    """Split the leaf training the most images at its median until all train at most `max_images`.

    `coverage(point_indices)` gives the share of each image those points cover;
    a leaf trains the images whose share reaches `visibility`. Leaves under
    `2 * minimum_points` points stay whole.
    """
    def training_images(leaf):
        return int(np.count_nonzero(coverage(leaf) >= visibility))

    leaves = [np.arange(len(positions))]
    counts = [training_images(leaves[0])]
    while True:
        splittable = [index for index, leaf in enumerate(leaves)
                      if counts[index] > max_images and len(leaf) >= 2 * minimum_points]
        if not splittable:
            break
        index = max(splittable, key=lambda index: counts[index])
        leaf = leaves.pop(index)
        counts.pop(index)
        order = leaf[np.argsort(along_wider_axis(positions, leaf), kind="stable")]
        for child in (order[: len(order) // 2], order[len(order) // 2:]):
            leaves.append(child)
            counts.append(training_images(child))
    return labels_from_parts(leaves, len(positions))


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


class ImageCoverage:
    """Share of each image a set of points covers.

    A fixed sample of the points is projected into every image once and binned
    on a `grid` x `grid` lattice over it; a set of points covers the cells its
    sampled points fall in. Occlusion is ignored. Measured geometrically
    because SfM tracks are short: an image that clearly sees a distant block
    shares few tracks with it.
    """

    def __init__(self, projections, image_sizes, points, sample_size=20_000, grid=32, chunk=256):
        sample = np.sort(np.random.default_rng(0).permutation(len(points))[:sample_size])
        self.sample_position = np.full(len(points), -1)
        self.sample_position[sample] = np.arange(len(sample))
        self.grid = grid
        self.chunk = chunk
        homogeneous = np.concatenate([points[sample], np.ones((len(sample), 1))], axis=1)
        # grid * grid marks a point outside the image or behind the camera.
        self.cells = np.full((len(projections), len(sample)), grid * grid, dtype=np.uint16)
        for start in range(0, len(projections), chunk):
            projected = np.einsum("ijk,pk->ipj", projections[start:start + chunk], homogeneous)
            depth = projected[..., 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                pixels = projected[..., :2] / depth[..., None]
            cell = np.floor(pixels / image_sizes[start:start + chunk, None, :] * grid)
            inside = (depth > 0) & np.all((cell >= 0) & (cell < grid), axis=-1)
            cell = np.where(inside[..., None], cell, 0).astype(np.int64)
            self.cells[start:start + chunk] = np.where(inside, cell[..., 1] * grid + cell[..., 0], grid * grid)

    def __call__(self, point_indices):
        sampled = self.sample_position[point_indices]
        sampled = sampled[sampled >= 0]
        shares = np.zeros(len(self.cells))
        for start in range(0, len(self.cells), self.chunk):
            cells = self.cells[start:start + self.chunk, sampled].astype(np.int64)
            occupied = np.zeros((len(cells), self.grid * self.grid + 1), dtype=bool)
            occupied[np.arange(len(cells))[:, None], cells] = True
            shares[start:start + self.chunk] = occupied[:, :-1].mean(axis=1)
        return shares


def block_coverage(point_owners, coverage, blocks):
    """(images, blocks) share of each image each block's points cover."""
    return np.stack([coverage(np.flatnonzero(point_owners == block)) for block in range(blocks)], axis=1)


def image_membership(coverages, visibility):
    """Images train every block covering `visibility` of them, and the one covering most.

    An image no block's points reach, such as an upward face of sky, trains none.
    """
    membership = coverages >= visibility
    sees_any = coverages.max(axis=1) > 0
    membership[np.flatnonzero(sees_any), coverages[sees_any].argmax(axis=1)] = True
    return membership


def per_frame(values, image_frames, frames):
    """Combine per-image rows into per-frame rows: summed, or OR-ed for booleans."""
    combined = np.zeros((frames,) + values.shape[1:], dtype=values.dtype)
    if values.dtype == bool:
        np.logical_or.at(combined, image_frames, values)
    else:
        np.add.at(combined, image_frames, values)
    return combined


def evaluate(point_owners, frame_labels, membership, shared):
    """Balance, duplication, and the covisibility cut between frames' main blocks."""
    total = shared.sum()
    cut = shared[frame_labels[:, None] != frame_labels[None, :]].sum()
    training = membership.sum(axis=0)
    trained = membership.any(axis=1)
    return {
        "cut_fraction": float(cut / total) if total else 0.0,
        "owned_points": np.bincount(point_owners, minlength=membership.shape[1]).tolist(),
        "training_images": training.tolist(),
        "largest_over_mean_training": float(training.max() / training.mean()),
        "duplication": float(training.sum() / max(trained.sum(), 1)),
        "untrained_images": int((~trained).sum()),
    }


def manifest(method, coverages, membership, image_names, metrics):
    main = np.where(coverages.max(axis=1) > 0, coverages.argmax(axis=1), -1)
    blocks = []
    for block in range(membership.shape[1]):
        blocks.append(
            {
                "name": f"block_{block:02d}",
                "main_images": [image_names[image] for image in np.flatnonzero(main == block)],
                "training_images": [image_names[image] for image in np.flatnonzero(membership[:, block])],
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


def plot_partition(path, method, centers, points, point_owners, frame_membership, metrics):
    """Points by owning block, and per block the frames with a face training it."""
    pyplot = pyplot_module()
    blocks = frame_membership.shape[1]
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
        training = frame_membership[:, block]
        axes.scatter(centers[training, 0], centers[training, 1], s=1.5, c="black")
        frame_axes(axes, f"block_{block:02d}: {metrics['owned_points'][block]} points, "
                         f"{metrics['training_images'][block]} images")
        axes.set_xticks([])
        axes.set_yticks([])
    figure.savefig(path, dpi=120, bbox_inches="tight")
    pyplot.close(figure)


def plot_comparison(path, centers, points, results):
    """Top views of every method: point owners, frames' main blocks, and blocks per frame.

    `results` maps a method to its (point_owners, frame_labels, frame_membership, metrics).
    """
    pyplot = pyplot_module()
    low, high = view_limits(points, centers)
    figure, axes = pyplot.subplots(len(results), 3, figsize=(21, 7 * len(results)), squeeze=False)
    for row, (method, (point_owners, frame_labels, membership, metrics)) in enumerate(results.items()):
        colors = pyplot.get_cmap("tab20")(np.arange(membership.shape[1]) % 20)
        points_axes, frames_axes, overlap_axes = axes[row]
        points_axes.scatter(points[:, 0], points[:, 1], s=0.2, c=colors[point_owners], linewidths=0)
        points_axes.set_title(f"{method}: point owners")
        frames_axes.scatter(points[:, 0], points[:, 1], s=0.2, c="lightgray", linewidths=0)
        frames_axes.scatter(centers[:, 0], centers[:, 1], s=4, c=colors[frame_labels])
        frames_axes.set_title(f"{method}: frames by main block, cut {metrics['cut_fraction']:.3f}")
        overlap_axes.scatter(points[:, 0], points[:, 1], s=0.2, c="lightgray", linewidths=0)
        overlap = overlap_axes.scatter(centers[:, 0], centers[:, 1], s=4, c=membership.sum(axis=1),
                                       cmap="plasma", vmin=1, vmax=max(2, membership.sum(axis=1).max()))
        figure.colorbar(overlap, ax=overlap_axes, label="blocks a face of the frame trains")
        overlap_axes.set_title(f"{method}: duplication {metrics['duplication']:.2f}, "
                               f"largest/mean {metrics['largest_over_mean_training']:.2f}")
        for each in axes[row]:
            each.set_xlim(low[0], high[0])
            each.set_ylim(low[1], high[1])
            each.set_aspect("equal")
    figure.savefig(path, dpi=110, bbox_inches="tight")
    pyplot.close(figure)


def point_owners_for(method, model, shared, coverage, arguments):
    blocks = arguments.blocks
    if method == "grid":
        return partition_grid(model.points, *grid_shape(model.points, blocks))
    if method == "tree" and arguments.max_images:
        return partition_tree_by_images(model.points, coverage, arguments.max_images, arguments.visibility)
    if method == "tree":
        return partition_tree(model.points, blocks)
    return owners_by_vote(partition_graph(shared, blocks), model.incidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sparse_dir", help="COLMAP model directory, e.g. <workspace>/sparse/0")
    parser.add_argument("output_dir")
    parser.add_argument("--method", choices=["grid", "tree", "graph", "all"], default="all")
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--max-images", type=int,
                        help="tree only: split until every block trains at most this many images; "
                             "replaces --blocks.")
    parser.add_argument("--visibility", type=float, default=0.1,
                        help="Share of an image a block's points must cover for the image to train it.")
    parser.add_argument("--minimum-shared", type=int, default=500,
                        help="Shared points for an edge to appear in covisibility.png.")
    arguments = parser.parse_args()
    if arguments.max_images and arguments.method != "tree":
        parser.error("--max-images needs --method tree")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model = load_model(pycolmap.Reconstruction(arguments.sparse_dir))
    frames = len(model.frame_centers)
    shared = covisibility(model.incidence)
    coverage = ImageCoverage(model.projections, model.image_sizes, model.points)
    logging.info("%d images in %d frames, %d points, %d covisible frame pairs",
                 len(model.image_names), frames, len(model.points), int(np.count_nonzero(shared) / 2))
    os.makedirs(arguments.output_dir, exist_ok=True)
    plot_covisibility(os.path.join(arguments.output_dir, "covisibility.png"),
                      model.frame_centers, shared, arguments.minimum_shared)

    methods = ["grid", "tree", "graph"] if arguments.method == "all" else [arguments.method]
    results = {}
    for method in methods:
        point_owners = point_owners_for(method, model, shared, coverage, arguments)
        coverages = block_coverage(point_owners, coverage, point_owners.max() + 1)
        membership = image_membership(coverages, arguments.visibility)
        frame_labels = per_frame(coverages, model.image_frames, frames).argmax(axis=1)
        frame_membership = per_frame(membership, model.image_frames, frames)
        metrics = evaluate(point_owners, frame_labels, membership, shared)
        logging.info("%s: %s", method, json.dumps(metrics))
        with open(os.path.join(arguments.output_dir, f"{method}.json"), "w") as file:
            json.dump(manifest(method, coverages, membership, model.image_names, metrics), file, indent=1)
        plot_partition(os.path.join(arguments.output_dir, f"{method}.png"),
                       method, model.frame_centers, model.points, point_owners, frame_membership, metrics)
        results[method] = (point_owners, frame_labels, frame_membership, metrics)
    if len(results) > 1:
        plot_comparison(os.path.join(arguments.output_dir, "comparison.png"),
                        model.frame_centers, model.points, results)


if __name__ == "__main__":
    main()
