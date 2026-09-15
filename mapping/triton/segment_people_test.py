import numpy as np

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
