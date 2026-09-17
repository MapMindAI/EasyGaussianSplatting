import numpy as np
import pytest

from mapping.tsdf.view_depth import DepthBrowser, cube_faces, visible_depth


def _depth_tree(root, faces):
    for face, frames in faces.items():
        (root / face).mkdir(parents=True)
        for frame in frames:
            np.save(root / face / f"{frame}.npy", np.zeros((2, 2), dtype=np.float32))
    return root


def test_cube_faces_lists_the_frames_under_every_face(tmp_path):
    _depth_tree(tmp_path, {"front": ["000002", "000001"], "down": ["000001"]})
    (tmp_path / "empty").mkdir()

    assert cube_faces(tmp_path) == {
        "down": ["000001"],
        "front": ["000001", "000002"],
    }


def test_cube_faces_requires_a_rendered_depth(tmp_path):
    with pytest.raises(FileNotFoundError):
        cube_faces(tmp_path)


def test_visible_depth_blanks_the_unrendered_pixels():
    blanked = visible_depth(np.array([[1.5, 0.0]], dtype=np.float32))
    assert blanked[0, 0] == 1.5
    assert np.isnan(blanked[0, 1])


def test_browser_wraps_around_frames_and_faces():
    browser = DepthBrowser({"front": ["1", "2"], "down": ["1", "2"]})

    assert (browser.face, browser.frame) == ("front", "1")
    browser.step_frame(-1)
    assert browser.frame == "2"
    browser.step_frame(1)
    assert browser.frame == "1"
    browser.step_face(-1)
    assert browser.face == "down"


def test_browser_opens_on_a_named_face_and_keeps_the_frame_in_range():
    browser = DepthBrowser({"down": ["1"], "front": ["1", "2"]}, "front")

    assert browser.face == "front"
    browser.step_frame(1)
    browser.step_face(-1)
    assert (browser.face, browser.frame) == ("down", "1")


def test_browser_rejects_a_face_without_depths():
    with pytest.raises(KeyError):
        DepthBrowser({"front": ["1"]}, "up")
