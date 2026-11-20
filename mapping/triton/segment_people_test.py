import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mapping.triton import segment_people
from mapping.triton.segment_people import dilate, mask_overlay, training_mask


def test_training_mask_is_white_where_nothing_is_blocked():
    blocked = np.array([[True, False], [False, True]])

    mask = training_mask(blocked)

    assert mask.tolist() == [[0, 255], [255, 0]]
    assert mask.dtype == np.uint8


def test_dilate_grows_the_mask_by_the_radius():
    mask = np.zeros((9, 9), dtype=bool)
    mask[4, 4] = True

    grown = dilate(mask, 2)

    assert grown[4, 6] and grown[6, 4] and grown[4, 2] and grown[2, 4]
    assert not grown[4, 7]


def test_dilate_with_zero_radius_is_identity():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True

    assert np.array_equal(dilate(mask, 0), mask)


def test_mask_overlay_marks_people_red_over_sky_magenta():
    image = np.zeros((1, 3, 3), dtype=np.uint8)
    people = np.array([[True, False, True]])
    sky = np.array([[False, True, True]])

    overlay = mask_overlay(image, people, sky, alpha=1.0)

    assert overlay.tolist() == [[[0, 0, 255], [255, 0, 255], [0, 0, 255]]]


@pytest.mark.parametrize("mask_people", [False, True])
@pytest.mark.parametrize("mask_sky", [False, True])
def test_mask_choices_only_run_enabled_models(tmp_path, monkeypatch, mask_people, mask_sky):
    image_directory = tmp_path / "images" / "front"
    image_directory.mkdir(parents=True)
    image = np.full((2, 3, 3), 80, dtype=np.uint8)
    cv2.imwrite(str(image_directory / "frame.png"), image)
    people = np.array([[True, False, False], [False, False, False]])
    sky_probability = np.array([[0.0, 0.9, 0.4], [0.0, 0.0, 0.0]], dtype=np.float32)
    calls = []

    class PersonSegmenter:
        def __init__(self):
            assert mask_people
            calls.append("load people")

        def mask(self, source):
            calls.append("mask people")
            return people

    class SkySegmenter:
        def __init__(self):
            assert mask_sky
            calls.append("load sky")

        def probability(self, source):
            calls.append("mask sky")
            return sky_probability

    monkeypatch.setitem(sys.modules, "mapping.triton.people", SimpleNamespace(
        PersonSegmenter=PersonSegmenter,
    ))
    monkeypatch.setitem(sys.modules, "mapping.sky_segmentation.sky_segmenter", SimpleNamespace(
        SkySegmenter=SkySegmenter,
    ))
    arguments = [
        "segment_people", str(tmp_path / "images"), str(tmp_path / "masks"),
        "--mask-people" if mask_people else "--no-mask-people",
        "--mask-sky" if mask_sky else "--no-mask-sky",
        "--dilation", "0", "--num-threads", "1",
        "--debug-dir", str(tmp_path / "debug"),
    ]
    monkeypatch.setattr(sys, "argv", ["segment_people"])

    segment_people.main(arguments[1:])

    assert ("load people" in calls) == mask_people
    assert ("mask people" in calls) == mask_people
    assert ("load sky" in calls) == mask_sky
    assert ("mask sky" in calls) == mask_sky
    if not (mask_people or mask_sky):
        assert not (tmp_path / "masks").exists()
        assert not (tmp_path / "debug").exists()
        return

    blocked = (people & mask_people) | ((sky_probability > 0.5) & mask_sky)
    mask = cv2.imread(str(tmp_path / "masks/front/frame.png.png"), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(mask, training_mask(blocked))
    overlay = cv2.imread(str(tmp_path / "debug/front/frame.png.png"))
    expected_pixels = {
        (True, False): [[40, 40, 168], [40, 40, 40], [40, 40, 40]],
        (False, True): [[40, 40, 40], [168, 40, 168], [40, 40, 40]],
        (True, True): [[40, 40, 168], [168, 40, 168], [40, 40, 40]],
    }
    assert overlay[0].tolist() == expected_pixels[mask_people, mask_sky]

    calls.clear()
    monkeypatch.setattr(sys, "argv", arguments)
    segment_people.main()
    assert calls == []
