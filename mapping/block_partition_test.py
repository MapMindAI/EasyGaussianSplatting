import numpy as np
from scipy import sparse

from mapping import block_partition


def incidence_matrix(observations, frames):
    """Binary (points, frames) matrix from one frame list per point."""
    rows = [point for point, observers in enumerate(observations) for _ in observers]
    columns = [frame for observers in observations for frame in observers]
    return sparse.csr_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(len(observations), frames)
    )


def test_covisibility_counts_shared_points_without_self_pairs():
    incidence = incidence_matrix([[0, 1], [0, 1], [1, 2]], frames=3)
    shared = block_partition.covisibility(incidence)
    assert shared.tolist() == [[0, 2, 0], [2, 0, 1], [0, 1, 0]]


def test_bisect_recursively_balances_odd_block_counts():
    parts = block_partition.bisect_recursively(np.arange(10), 3, lambda indices: indices)
    assert [part.tolist() for part in parts] == [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]


def test_grid_balances_counts_per_cell():
    positions = np.random.default_rng(0).uniform(-10, 10, size=(400, 3))
    labels = block_partition.partition_grid(positions, rows=2, columns=3)
    assert np.bincount(labels).tolist() == [66, 67, 67, 67, 66, 67]
    # The first column holds the smallest x values.
    assert positions[labels < 2, 0].max() <= positions[labels >= 2, 0].min()


def test_grid_shape_follows_the_extent():
    positions = np.zeros((100, 3))
    positions[:, 0] = np.linspace(0, 40, 100)
    positions[:, 1] = np.linspace(0, 10, 100)
    assert block_partition.grid_shape(positions, 8) == (2, 4)


def test_tree_splits_along_the_wider_axis_first():
    positions = np.zeros((8, 3))
    positions[:, 0] = [0, 1, 2, 3, 100, 101, 102, 103]
    positions[:, 1] = [0, 1, 0, 1, 0, 1, 0, 1]
    labels = block_partition.partition_tree(positions, 2)
    assert labels.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_graph_separates_weakly_linked_clusters():
    shared = np.zeros((8, 8))
    for cluster in ([0, 2, 4, 6], [1, 3, 5, 7]):
        shared[np.ix_(cluster, cluster)] = 100.0
    shared[6, 7] = shared[7, 6] = 1.0
    np.fill_diagonal(shared, 0.0)
    labels = block_partition.partition_graph(shared, 2)
    assert len(set(labels[[0, 2, 4, 6]])) == 1
    assert len(set(labels[[1, 3, 5, 7]])) == 1
    assert labels[0] != labels[1]


def test_images_train_their_main_block_and_any_well_covered_one():
    coverages = np.array([[0.5, 0.0], [0.3, 0.15], [0.05, 0.08], [0.0, 0.0]])
    membership = block_partition.image_membership(coverages, visibility=0.1)
    # The last image sees no block, like an upward face of sky.
    assert membership.tolist() == [[True, False], [True, True], [False, True], [False, False]]


def test_image_coverage_counts_occupied_cells_in_front_of_the_camera():
    # One 100 x 100 pinhole camera at the origin looking down +z.
    projection = np.array([[[50.0, 0, 50, 0], [0, 50.0, 50, 0], [0, 0, 1, 0]]])
    points = np.array([
        [-0.9, -0.9, 1.0],  # top-left cell
        [-0.89, -0.89, 1.0],  # same cell
        [0.9, 0.9, 1.0],  # bottom-right cell
        [0.0, 0.0, -1.0],  # behind the camera
        [5.0, 0.0, 1.0],  # outside the image
    ])
    coverage = block_partition.ImageCoverage(projection, np.array([[100, 100]]), points, grid=4)
    assert coverage(np.arange(5)).tolist() == [2 / 16]
    assert coverage(np.array([3, 4])).tolist() == [0.0]


def test_tree_by_images_splits_until_the_limit():
    positions = np.zeros((8, 3))
    positions[:, 0] = np.arange(8)

    def coverage(leaf):
        # Ten images, each covered by any leaf holding its point.
        shares = np.zeros(10)
        shares[leaf] = 1.0
        return shares

    labels = block_partition.partition_tree_by_images(positions, coverage, max_images=2, visibility=0.5,
                                                     minimum_points=1)
    assert sorted(np.bincount(labels).tolist()) == [2, 2, 2, 2]
    assert len(set(labels[[0, 1]])) == 1


def test_per_frame_sums_coverage_and_ors_membership():
    image_frames = np.array([0, 0, 1])
    coverages = np.array([[0.1, 0.2], [0.3, 0.0], [0.0, 0.4]])
    assert np.allclose(block_partition.per_frame(coverages, image_frames, 2), [[0.4, 0.2], [0.0, 0.4]])
    membership = np.array([[True, False], [False, False], [False, True]])
    assert block_partition.per_frame(membership, image_frames, 2).tolist() == [[True, False], [False, True]]


def test_points_go_to_the_block_most_observers_belong_to():
    incidence = incidence_matrix([[0, 1, 2], [2]], frames=3)
    owners = block_partition.owners_by_vote(np.array([0, 0, 1]), incidence)
    assert owners.tolist() == [0, 1]


def test_evaluate_reports_cut_and_duplication():
    shared = np.array([[0, 4, 1], [4, 0, 0], [1, 0, 0]], dtype=float)
    membership = np.array([[True, False], [True, True], [False, True], [False, False]])
    metrics = block_partition.evaluate(np.array([0, 0, 1]), np.array([0, 0, 1]), membership, shared)
    assert metrics["cut_fraction"] == 0.2
    assert metrics["training_images"] == [2, 2]
    assert metrics["duplication"] == 4 / 3
    assert metrics["untrained_images"] == 1
