import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mapping.triton import segment_people
from mapping.triton.segment_people import (
    ADE20K_NUM_CLASSES,
    CLASS_COLOURS,
    dilate,
    label_colour_legend,
    label_overlay,
    person_overlay,
    training_mask,
)


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


def test_label_overlay_blends_the_class_colour_into_the_image():
    image = np.zeros((1, 2, 3), dtype=np.uint8)
    classes = np.array([[12, 2]], dtype=np.uint8)

    overlay = label_overlay(image, classes, alpha=0.5)

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    # Half of the class colour, since the image under it is black.
    assert np.array_equal(overlay[0, 0], np.round(CLASS_COLOURS[12] * 0.5))
    assert np.array_equal(overlay[0, 1], np.round(CLASS_COLOURS[2] * 0.5))


def test_label_overlay_leaves_the_image_alone_at_zero_alpha():
    image = np.full((2, 2, 3), 40, dtype=np.uint8)
    classes = np.full((2, 2), 12, dtype=np.uint8)

    assert np.array_equal(label_overlay(image, classes, alpha=0.0), image)


def test_person_overlay_marks_people_in_red():
    image = np.zeros((1, 2, 3), dtype=np.uint8)
    people = np.array([[True, False]])

    overlay = person_overlay(image, people, alpha=1.0)

    assert overlay.tolist() == [[[0, 0, 255], [0, 0, 0]]]


def test_every_class_keeps_one_colour_and_neighbours_differ():
    assert CLASS_COLOURS.shape == (256, 3)
    # Distinguishable by eye is the whole point, so no two of the labels this
    # stage acts on may collide.
    assert not np.array_equal(CLASS_COLOURS[2], CLASS_COLOURS[12])


def test_label_colour_legend_uses_the_overlay_palette_for_every_label():
    columns = 10
    legend = label_colour_legend(columns)

    assert legend.shape == (30 + 15 * 28, columns * 100, 3)
    assert np.array_equal(legend[34, 4], CLASS_COLOURS[0])
    row, column = divmod(ADE20K_NUM_CLASSES - 1, columns)
    assert np.array_equal(legend[30 + row * 28 + 4, column * 100 + 4], CLASS_COLOURS[149])


@pytest.mark.parametrize("mask_people", [False, True])
@pytest.mark.parametrize("mask_sky", [False, True])
def test_mask_choices_only_run_enabled_models(tmp_path, monkeypatch, mask_people, mask_sky):
    image_directory = tmp_path / "images" / "front"
    image_directory.mkdir(parents=True)
    image = np.full((2, 3, 3), 80, dtype=np.uint8)
    cv2.imwrite(str(image_directory / "frame.png"), image)
    people = np.array([[True, False, False], [False, False, False]])
    classes = np.array([[0, 2, 0], [0, 0, 0]], dtype=np.uint8)
    calls = []

    class PersonSegmenter:
        def __init__(self):
            assert mask_people
            calls.append("load people")

        def mask(self, source):
            calls.append("mask people")
            return people

    class SemanticSegmenter:
        def __init__(self, endpoint):
            assert mask_sky
            assert endpoint == "triton:8001"
            calls.append("load sky")

        def classes(self, source):
            calls.append("mask sky")
            return classes

    monkeypatch.setitem(sys.modules, "mapping.triton.people", SimpleNamespace(
        PersonSegmenter=PersonSegmenter,
    ))
    monkeypatch.setitem(sys.modules, "mapping.triton.clients", SimpleNamespace(
        SemanticSegmenter=SemanticSegmenter, ADE20K_SKY_CLASS=2,
    ))
    arguments = [
        "segment_people", str(tmp_path / "images"), str(tmp_path / "masks"),
        "--mask-people" if mask_people else "--no-mask-people",
        "--mask-sky" if mask_sky else "--no-mask-sky",
        "--dilation", "0", "--num-threads", "1",
        "--debug-dir", str(tmp_path / "debug"),
    ]
    monkeypatch.delenv("TRITON_URL", raising=False)
    if mask_sky:
        arguments.extend(["--triton-url", "triton:8001"])
    monkeypatch.setattr(sys, "argv", arguments)

    segment_people.main()

    assert ("load people" in calls) == mask_people
    assert ("mask people" in calls) == mask_people
    assert ("load sky" in calls) == mask_sky
    assert ("mask sky" in calls) == mask_sky
    if not (mask_people or mask_sky):
        assert not (tmp_path / "masks").exists()
        assert not (tmp_path / "debug").exists()
        return

    blocked = (people & mask_people) | ((classes == 2) & mask_sky)
    mask = cv2.imread(str(tmp_path / "masks/front/frame.png.png"), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(mask, training_mask(blocked))
    overlay = cv2.imread(str(tmp_path / "debug/front/frame.png.png"))
    expected_pixels = {
        (True, False): [[40, 40, 168], [40, 40, 40]],
        (False, True): [[40, 40, 168], [40, 86, 138]],
        (True, True): [[20, 20, 212], [20, 43, 69]],
    }
    assert overlay[0, :2].tolist() == expected_pixels[mask_people, mask_sky]
    assert (tmp_path / "debug/label_colours.png").exists() == mask_sky

    calls.clear()
    segment_people.main()
    assert calls == []
