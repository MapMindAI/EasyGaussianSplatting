"""Tests for the mapping pipeline's gravity levelling."""

import sys
import types

import numpy as np
import pycolmap
import pytest

# features.triton_models refuses to import without the EasyTensorRT submodule,
# and reaching it needs a Triton endpoint. The levelling touches neither.
_triton_models = types.ModuleType("mapping.features.triton_models")
for _name in ("FeatureMatcher", "GlobalFeatureExtractor", "LocalFeatureExtractor",
              "LocalFeatures"):
    setattr(_triton_models, _name, type(_name, (), {}))
sys.modules.setdefault("mapping.features.triton_models", _triton_models)

from mapping.mapping_pipeline import (  # noqa: E402
    _rotation_to_world_down,
    align_up_axis,
)
from mapping.panorama_database import RIG_DOWN  # noqa: E402

WORLD_DOWN = np.array([0.0, 0.0, -1.0])


class Frame:
    def __init__(self, rotation=None, has_pose=True):
        self.rig_from_world = pycolmap.Rigid3d(
            rotation if rotation is not None else pycolmap.Rotation3d(), np.zeros(3)
        )
        self.has_pose = has_pose


class Reconstruction:
    """Only what align_up_axis reads: the frames, and the transform it applies."""

    def __init__(self, *frames):
        self.frames = dict(enumerate(frames))
        self.applied = []

    def transform(self, sim3d):
        self.applied.append(sim3d)


def rotation_about_x(degrees):
    half = np.radians(degrees) / 2.0
    return pycolmap.Rotation3d(np.array([np.sin(half), 0.0, 0.0, np.cos(half)]))


def rig_down_in_world(frame):
    return frame.rig_from_world.rotation.inverse() * RIG_DOWN


def unit(vector):
    return vector / np.linalg.norm(vector)


# --- align_up_axis --------------------------------------------------------


def test_a_level_map_puts_the_rig_down_axis_on_minus_z():
    reconstruction = Reconstruction(Frame())

    align_up_axis(reconstruction)

    (applied,) = reconstruction.applied
    assert applied.rotation * unit(RIG_DOWN) == pytest.approx(WORLD_DOWN, abs=1e-6)


@pytest.mark.parametrize("tilt", [10.0, 30.0, 90.0, -45.0, 179.0])
def test_a_tilted_map_is_levelled(tilt):
    frame = Frame(rotation_about_x(tilt))
    reconstruction = Reconstruction(frame)

    align_up_axis(reconstruction)

    (applied,) = reconstruction.applied
    levelled = applied.rotation * rig_down_in_world(frame)
    assert levelled == pytest.approx(WORLD_DOWN, abs=1e-6)


def test_gravity_is_averaged_over_the_frames():
    frames = [Frame(rotation_about_x(20.0)), Frame(rotation_about_x(-40.0))]
    reconstruction = Reconstruction(*frames)

    align_up_axis(reconstruction)

    (applied,) = reconstruction.applied
    average = unit(sum(rig_down_in_world(frame) for frame in frames))
    assert applied.rotation * average == pytest.approx(WORLD_DOWN, abs=1e-6)
    # The average is a direction neither frame reports on its own.
    for frame in frames:
        assert applied.rotation * rig_down_in_world(frame) != pytest.approx(
            WORLD_DOWN, abs=1e-3
        )


def test_frames_without_a_pose_are_ignored():
    posed = Frame(rotation_about_x(25.0))
    with_unposed = Reconstruction(posed, Frame(rotation_about_x(140.0), has_pose=False))
    posed_only = Reconstruction(Frame(rotation_about_x(25.0)))

    align_up_axis(with_unposed)
    align_up_axis(posed_only)

    assert with_unposed.applied[0].rotation.matrix() == pytest.approx(
        posed_only.applied[0].rotation.matrix(), abs=1e-6
    )


def test_a_map_with_no_registered_frames_is_left_alone():
    reconstruction = Reconstruction(Frame(has_pose=False), Frame(has_pose=False))

    align_up_axis(reconstruction)

    assert reconstruction.applied == []


def test_an_empty_reconstruction_is_left_alone():
    reconstruction = Reconstruction()

    align_up_axis(reconstruction)

    assert reconstruction.applied == []


def test_opposing_frames_that_cancel_leave_the_map_as_solved():
    """Averaging to zero has no axis to level, so the map is left as solved."""
    reconstruction = Reconstruction(Frame(), Frame(rotation_about_x(180.0)))

    align_up_axis(reconstruction)

    assert reconstruction.applied == []


def test_levelling_keeps_the_scale_and_position_the_mapper_solved():
    reconstruction = Reconstruction(Frame(rotation_about_x(30.0)))

    align_up_axis(reconstruction)

    (applied,) = reconstruction.applied
    assert applied.scale == pytest.approx(1.0)
    assert applied.translation == pytest.approx(np.zeros(3))


# --- _rotation_to_world_down ---------------------------------------------


def test_an_already_level_axis_needs_no_rotation():
    rotation = _rotation_to_world_down(WORLD_DOWN)

    assert rotation.matrix() == pytest.approx(np.eye(3), abs=1e-9)


def test_an_upside_down_axis_turns_a_half_turn():
    """The antipodal case has no shortest arc, so any horizontal axis serves."""
    rotation = _rotation_to_world_down(np.array([0.0, 0.0, 1.0]))

    assert rotation * np.array([0.0, 0.0, 1.0]) == pytest.approx(WORLD_DOWN, abs=1e-9)


@pytest.mark.parametrize(
    "down",
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 1.0, -1.0],
        [-0.3, 0.5, 0.81],
    ],
)
def test_any_axis_lands_on_minus_z(down):
    down = unit(np.array(down))

    rotation = _rotation_to_world_down(down)

    assert rotation * down == pytest.approx(WORLD_DOWN, abs=1e-6)
    matrix = rotation.matrix()
    assert matrix @ matrix.T == pytest.approx(np.eye(3), abs=1e-6)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_the_rotation_is_the_shortest_arc():
    """A shortest-arc rotation turns about an axis square to both directions."""
    down = unit(np.array([0.4, -0.2, 0.6]))

    rotation = _rotation_to_world_down(down)

    expected_axis = unit(np.cross(down, WORLD_DOWN))
    assert rotation * expected_axis == pytest.approx(expected_axis, abs=1e-6)
