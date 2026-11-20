"""Tests for covisibility links and the greedy group selection."""

import numpy as np

from mapping.depth.covisibility_groups import Frame, covisibility_graph, select_groups


def wall(count=100):
    """Points on the plane z = 10 over x in [-2, 2]."""
    x = np.linspace(-2.0, 2.0, count)
    return np.column_stack([x, np.zeros(count), np.full(count, 10.0)])


def yaw(degrees):
    angle = np.radians(degrees)
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])


def frame(x, points, rotation=None):
    return Frame(
        center=np.array([x, 0.0, 0.0]),
        rotation=np.eye(3) if rotation is None else rotation,
        points=np.asarray(points, dtype=np.int64),
    )


def test_cameras_sharing_points_with_parallax_are_linked():
    points = wall()
    graph = covisibility_graph([frame(0.0, range(100)), frame(0.5, range(100))], points)

    assert [edge.neighbor for edge in graph[0]] == [1]
    assert 0.5 < graph[0][0].parallax < 25


def test_too_few_shared_points_too_short_a_baseline_or_turned_away_breaks_a_link():
    points = wall()
    few_shared = [frame(0.0, range(0, 50)), frame(0.5, range(35, 100))]
    no_baseline = [frame(0.0, range(100)), frame(0.01, range(100))]
    turned = [frame(0.0, range(100)), frame(0.5, range(100), rotation=yaw(60))]

    for frames in (few_shared, no_baseline, turned):
        assert covisibility_graph(frames, points) == [[], []]


def test_every_linked_frame_lands_in_a_group_of_at_most_the_group_size():
    points = wall()
    frames = [frame(0.3 * index, range(100)) for index in range(7)]
    frames.append(frame(40.0, []))

    groups, rejected = select_groups(frames, points, 3, 2)

    assert sorted({index for group in groups for index in group}) == list(range(7))
    assert all(2 <= len(group) <= 3 for group in groups)
    assert len(groups) == 3
    assert rejected == [7]


def test_supports_prefer_target_parallax_over_a_barely_moved_camera():
    points = wall()
    frames = [frame(0.0, range(100)), frame(0.12, range(100)), frame(0.9, range(100))]

    groups, _ = select_groups(frames, points, 2, 2)

    # 0.9 m at 10 m is about 5 degrees of parallax; 0.12 m is under 1.
    assert [0, 2] in groups


def test_frames_no_full_group_can_hold_are_rejected_not_lost_in_a_short_group():
    points = wall()
    # Frames 0-3 can fill a group of four; frames 4 and 5 only see each other.
    frames = [frame(0.3 * index, range(60)) for index in range(4)]
    frames += [frame(10.0, range(60, 100)), frame(10.4, range(60, 100))]

    groups, rejected = select_groups(frames, points, 4, 4)

    assert [sorted(group) for group in groups] == [[0, 1, 2, 3]]
    assert sorted(rejected) == [4, 5]
