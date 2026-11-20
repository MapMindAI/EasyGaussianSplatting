"""Tests for the example's pixel selection."""

import numpy as np

from mapping.vggt_omega.example import cloud_pixels


def test_cloud_pixels_keep_the_confident_half_of_the_grid():
    confidence = np.tile(np.arange(10, dtype=float), (4, 1))
    pixels = cloud_pixels(confidence)
    assert set(pixels[:, 0]) == {5, 6, 7, 8, 9}
    assert len(pixels) == 20
