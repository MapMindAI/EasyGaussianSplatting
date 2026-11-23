import cv2
import numpy as np
import pytest

from mapping.depth.depth_maps import (
    DepthFilter,
    DepthMap,
    depth_edge_mask,
    depth_path,
    depth_rows,
    load_depth_map,
    save_depth_map,
    save_empty_depth_map,
    supervised_depth,
    training_masks,
    upsample,
)


def test_maps_round_trip_with_confidence_and_scale(tmp_path):
    path = depth_path(tmp_path, "front/frame.jpg")
    depth = np.array([[0.0, 1.5], [80.0, 1000.0]])

    save_depth_map(path, depth, [[0.2, 0.5], [1.0, 50.0]], 3.5)

    loaded = load_depth_map(path)
    assert path.name == "frame.jpg.npz"
    np.testing.assert_allclose(loaded.depth, depth, rtol=1e-3)
    # Stored in steps of 0.05, saturating at 12.75.
    np.testing.assert_allclose(loaded.confidence, [[0.2, 0.5], [1.0, 12.75]], rtol=1e-6)
    assert loaded.scale == 3.5
    assert [entry.name for entry in path.parent.iterdir()] == ["frame.jpg.npz"]


def test_an_empty_map_means_no_depth(tmp_path):
    save_empty_depth_map(tmp_path / "empty.npz")

    assert load_depth_map(tmp_path / "empty.npz") is None


def test_depth_beyond_float16_is_refused(tmp_path):
    with pytest.raises(ValueError, match="float16"):
        save_depth_map(tmp_path / "far.npz", [[1e5]], [[1.0]], 1.0)


def test_supervised_depth_filters_confidence_scaled_range_and_mask():
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    depth[0, 0] = 40.0  # 20 model units at scale 2: beyond max_depth.
    confidence = np.full((4, 4), 1.0, dtype=np.float32)
    confidence[1] = 0.4
    mask = np.ones((8, 8), dtype=bool)
    mask[6:] = False  # Covers the map's last row once resized.
    grayscale = np.full((8, 8), 128, dtype=np.uint8)

    kept = supervised_depth(DepthMap(depth, confidence, 2.0), grayscale, mask, DepthFilter(0.5, 0.1, 5.0))

    assert kept[0, 0] == 0
    assert (kept[1] == 0).all() and (kept[3] == 0).all()
    # (0, 1) sits on the depth edge around the far pixel.
    assert (kept[2] == 2.0).all() and (kept[0, 2:] == 2.0).all()
    lenient = supervised_depth(DepthMap(depth, confidence, 2.0), grayscale, None, DepthFilter(0.3, 0.1, 5.0))
    assert (lenient[2:] == 2.0).all() and (lenient[1, 2:] == 2.0).all()


def test_training_masks_read_listed_masks_only(tmp_path):
    (tmp_path / "front").mkdir()
    cv2.imwrite(str(tmp_path / "front" / "a.jpg.png"), np.array([[0, 255]], dtype=np.uint8))

    mask = training_masks(tmp_path)

    assert mask("front/a.jpg").tolist() == [[False, True]]
    assert mask("front/b.jpg") is None
    assert training_masks(None)("front/a.jpg") is None


def test_depth_edge_mask_rejects_colour_and_depth_boundaries():
    grayscale = np.zeros((20, 20), dtype=np.uint8)
    grayscale[:, 10:] = 255
    depth = np.ones((20, 20), dtype=np.float32)
    depth[10:] = 2.0

    edges = depth_edge_mask(grayscale, depth)

    assert edges[:, 9:11].any(axis=1).all()
    assert edges[9:11, :].any(axis=0).all()
    assert not edges[0, 0]


def test_upsample_interpolates_between_valid_pixels_only():
    depth = np.array([[1.0, 3.0, 0.0]], dtype=np.float32).repeat(2, axis=0)

    upsampled = upsample(depth, 6, 4)

    # Inside the valid pair the depth blends; nothing blends toward the hole.
    assert upsampled[:, 0].tolist() == [1.0] * 4
    assert 1.0 < upsampled[0, 2] < 3.0
    assert (upsampled[:, 4:] == 0).all()


def test_depth_rows_normalize_the_upsampled_pixels():
    rows = depth_rows(np.full((2, 2), 4.0, dtype=np.float32), 5, 3)

    assert len(rows) == 15
    assert rows[:, 0].min() == 0 and rows[:, 0].max() == 1
    assert rows[:, 1].min() == 0 and rows[:, 1].max() == 1
    assert (rows[:, 2] == 4).all()
