import numpy as np

from mapping.segment_people import dilate, training_mask


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
