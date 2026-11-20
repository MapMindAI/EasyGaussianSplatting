"""Tests for the sky segmenter's input."""

import numpy as np

from mapping.sky_segmentation.sky_segmenter import IMAGENET_MEAN, IMAGENET_STD, model_input


def test_model_input_is_normalized_rgb_at_the_model_size():
    image = np.zeros((40, 60, 3), np.uint8)
    image[..., 0] = 255  # Blue in OpenCV's BGR.
    tensor = model_input(image, 8, 16)
    assert tensor.shape == (1, 3, 8, 16)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[0, :, 0, 0], (np.array([0, 0, 1]) - IMAGENET_MEAN) / IMAGENET_STD, rtol=1e-6)

