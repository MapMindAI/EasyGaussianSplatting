"""Groups of frames that view the same part of the scene, from sparse covisibility.

Ported from ArSession's dense_recon/covisibility_groups (SelectCovisibleGroups).
Two frames are linked when they share enough sparse points with enough overlap,
baseline, parallax, and similar enough viewing directions for a multi-view
depth model to triangulate between them. Each group is an anchor plus the
linked supports that best balance parallax, overlap, camera spread, and the
anchor's points they cover; anchors are chosen greedily until every frame is
in some group or cannot be.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

REQUIRED_SHARED_POINTS = 20
REQUIRED_OVERLAP = 0.2
# In map units; also the spacing below which two supports count as duplicates.
MIN_BASELINE = 0.05
MAX_ROTATION_DEGREES = 45.0
MIN_PARALLAX_DEGREES = 0.5
MAX_PARALLAX_DEGREES = 25.0
TARGET_PARALLAX_DEGREES = 5.0


@dataclass(frozen=True)
class Frame:
    center: np.ndarray  # (3,) camera centre in the world.
    rotation: np.ndarray  # (3, 3) camera-to-world.
    points: np.ndarray  # Sorted distinct indices of the points seen in front of it.


@dataclass(frozen=True)
class Edge:
    neighbor: int
    overlap: float
    parallax: float


def angle_degrees(cosine):
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def rotation_angle_degrees(rotation):
    return float(angle_degrees((np.trace(rotation) - 1) / 2))


def centre_and_rotation(camera_from_world):
    """Camera centre and camera-to-world rotation of a (3, 4) camera-from-world."""
    rotation = np.asarray(camera_from_world)[:, :3].T
    return -rotation @ np.asarray(camera_from_world)[:, 3], rotation


def covisibility_graph(frames, points):
    """Each frame's qualified links, as lists of Edge."""
    frame_indices = np.repeat(np.arange(len(frames)), [len(frame.points) for frame in frames])
    point_indices = np.concatenate([frame.points for frame in frames]).astype(np.int64)
    incidence = sparse.csr_matrix(
        (np.ones(len(point_indices)), (point_indices, frame_indices)),
        shape=(len(points), len(frames)),
    )
    shared = sparse.triu(incidence.T @ incidence, k=1).tocoo()

    graph = [[] for _ in frames]
    candidates = shared.data >= REQUIRED_SHARED_POINTS
    for first_index, second_index, count in sorted(zip(
        shared.row[candidates], shared.col[candidates], shared.data[candidates]
    )):
        first, second = frames[first_index], frames[second_index]
        overlap = count / min(len(first.points), len(second.points))
        if overlap < REQUIRED_OVERLAP:
            continue
        if np.linalg.norm(first.center - second.center) < MIN_BASELINE:
            continue
        if rotation_angle_degrees(first.rotation.T @ second.rotation) > MAX_ROTATION_DEGREES:
            continue
        common = points[np.intersect1d(first.points, second.points, assume_unique=True)]
        to_first, to_second = common - first.center, common - second.center
        cosines = (to_first * to_second).sum(axis=1) / np.maximum(
            np.linalg.norm(to_first, axis=1) * np.linalg.norm(to_second, axis=1), 1e-12
        )
        parallax = float(np.median(angle_degrees(cosines)))
        if not MIN_PARALLAX_DEGREES <= parallax <= MAX_PARALLAX_DEGREES:
            continue
        graph[first_index].append(Edge(int(second_index), overlap, parallax))
        graph[second_index].append(Edge(int(first_index), overlap, parallax))
    return graph


def choose_supports(anchor, frames, graph, group_size, remaining):
    """Up to `group_size - 1` of the anchor's neighbours, best first."""
    anchor_points = frames[anchor].points
    if not graph[anchor]:
        return []
    neighbors = np.array([edge.neighbor for edge in graph[anchor]])
    centers = np.array([frames[neighbor].center for neighbor in neighbors])
    # Fixed per candidate: closeness to the target parallax, and the overlap.
    fixed_score = np.array([
        2 * np.exp(-abs(np.log(max(edge.parallax, 1e-6) / TARGET_PARALLAX_DEGREES)))
        + 0.5 * edge.overlap
        for edge in graph[anchor]
    ])
    shares_point = np.array([
        np.isin(anchor_points, frames[neighbor].points, assume_unique=True)
        for neighbor in neighbors
    ])
    spread = np.linalg.norm(centers - frames[anchor].center, axis=1)
    max_baseline = max(MIN_BASELINE, float(spread.max()))
    covered = np.zeros(len(anchor_points), dtype=bool)
    available = np.ones(len(neighbors), dtype=bool)

    chosen = []
    while available.any() and len(chosen) < group_size - 1:
        coverage = (shares_point & ~covered).sum(axis=1) / len(anchor_points)
        score = fixed_score + np.minimum(spread / max_baseline, 1.0) + coverage
        # The first support is chosen for geometry; later ones also pull in
        # frames no group holds yet. An already-covered anchor must add one.
        fresh = remaining[neighbors] & (bool(chosen) or not remaining[anchor])
        candidates = np.flatnonzero(available)
        best = max(candidates, key=lambda index: (fresh[index], score[index], -neighbors[index]))
        chosen.append(int(neighbors[best]))
        covered |= shares_point[best]
        distance_to_pick = np.linalg.norm(centers - centers[best], axis=1)
        spread = np.minimum(spread, distance_to_pick)
        available &= distance_to_pick >= MIN_BASELINE
        available[best] = False
    return chosen


def select_groups(frames, points, group_size, minimum_group_size):
    """(groups, rejected): groups of `minimum_group_size` to `group_size` frame
    indices, anchor first, and the frames no such group could hold."""
    graph = covisibility_graph(frames, points)
    remaining = np.ones(len(frames), dtype=bool)
    # Frames a group anchored at each frame could still cover; -1 once that
    # anchor has failed to fill a group.
    reach = np.array([1 + len(edges) for edges in graph])

    def take(index):
        remaining[index] = False
        reach[index] -= 1
        for edge in graph[index]:
            reach[edge.neighbor] -= 1

    groups, rejected = [], []
    while remaining.any():
        # Ties go to an uncovered anchor, then to the lowest index.
        anchor = int(np.argmax(2 * reach + remaining))
        supports = choose_supports(anchor, frames, graph, group_size, remaining)
        if len(supports) + 1 < minimum_group_size:
            if remaining[anchor]:
                take(anchor)
                rejected.append(anchor)
            reach[anchor] = -1
            continue
        for index in [anchor, *supports]:
            if remaining[index]:
                take(index)
        groups.append([anchor, *supports])
    return groups, rejected
