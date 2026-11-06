import json

import numpy as np

from mapping import block_gaussians

# Two blocks split at x = 10: block 0 owns x < 10, block 1 the rest.
POINTS = np.array([[x, y, 0.0] for x in range(20) for y in range(10)], dtype=float)
OWNERS = (POINTS[:, 0] >= 10).astype(int)


def test_nearest_owners_ignore_height():
    positions = np.array([[2.0, 5.0, 100.0], [17.0, 1.0, -50.0]])
    assert block_gaussians.nearest_owners(positions, POINTS, OWNERS).tolist() == [0, 1]


def test_expanded_region_reaches_past_the_block():
    positions = np.array([[5.0, 5.0, 0.0], [9.6, 5.0, 0.0], [10.2, 5.0, 0.0], [12.0, 5.0, 0.0]])
    # Block 0's middle 90% spans x 0.45 to 8.55, so it reaches 0.81 past its last point at x 9.
    region = block_gaussians.expanded_region(positions, POINTS, OWNERS, block=0)
    assert region.tolist() == [True, True, False, False]
    region = block_gaussians.expanded_region(positions, POINTS, OWNERS, block=1)
    assert region.tolist() == [False, True, True, True]


def test_to_colmap_inverts_the_parser_similarity():
    transform = np.diag([2.0, 2.0, 2.0, 1.0])
    transform[:3, 3] = [1.0, 0.0, -1.0]
    colmap = np.array([[1.0, 2.0, 3.0]])
    normalized = colmap @ transform[:3, :3].T + transform[:3, 3]
    assert np.allclose(block_gaussians.to_colmap(transform, normalized), colmap)


def test_load_block_reads_the_points_beside_the_manifest(tmp_path):
    np.savez(tmp_path / "tree_points.npz", points=POINTS, owners=OWNERS)
    manifest = {"points": "tree_points.npz",
                "blocks": [{"name": "block_00", "training_images": ["a"]},
                           {"name": "block_01", "training_images": ["b"]}]}
    (tmp_path / "tree.json").write_text(json.dumps(manifest))
    block, index, points, owners = block_gaussians.load_block(str(tmp_path / "tree.json"), "block_01")
    assert (block["training_images"], index) == (["b"], 1)
    assert np.array_equal(points, POINTS) and np.array_equal(owners, OWNERS)


def test_latest_checkpoint_orders_steps_numerically(tmp_path):
    (tmp_path / "ckpts").mkdir()
    for name in ["ckpt_999_rank0.pt", "ckpt_29999_rank0.pt", "ckpt_29999_rank1.pt"]:
        (tmp_path / "ckpts" / name).touch()
    assert block_gaussians.latest_checkpoint(str(tmp_path)).endswith("ckpt_29999_rank0.pt")
