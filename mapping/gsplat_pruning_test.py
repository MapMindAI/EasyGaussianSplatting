from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from gsplat_server.parameters import load_default_parameters, load_parameters
from gsplat_server.proto import gsplat_pb2
from mapping.gsplat_pruning import (
    load_training_mask, removal_mask, validate_pruning, validate_pruning_configuration,
)


def test_pruning_is_disabled_by_default():
    parameters = load_default_parameters()
    assert not parameters.mcmc_prune_steps
    validate_pruning(parameters)


def test_pruning_settings_survive_parameter_loading(tmp_path):
    path = tmp_path / "pruning.proto.txt"
    path.write_text("mcmc_prune_steps: 1000\nmcmc_prune_steps: 2000\nmcmc_prune_keep_fraction: 0.6\n")
    parameters = load_parameters(path)
    validate_pruning(parameters)
    assert list(parameters.mcmc_prune_steps) == [1000, 2000]
    assert parameters.mcmc_prune_keep_fraction == pytest.approx(0.6)
    assert parameters.mcmc_prune_scoring_views == 32
    assert parameters.mcmc_prune_resolution == 256


@pytest.mark.parametrize("steps", [[0], [-1], [30000], [2000, 1000], [1000, 1000]])
def test_invalid_pruning_schedule_is_rejected(steps):
    parameters = load_default_parameters()
    parameters.mcmc_prune_steps.extend(steps)
    with pytest.raises(ValueError, match="increasing, unique"):
        validate_pruning(parameters)


@pytest.mark.parametrize("field,value", [
    ("strategy", gsplat_pb2.STRATEGY_DEFAULT),
    ("mcmc_prune_keep_fraction", 0), ("mcmc_prune_keep_fraction", 1),
    ("mcmc_prune_keep_fraction", float("nan")),
    ("mcmc_prune_scoring_views", 0), ("mcmc_prune_resolution", 0),
])
def test_invalid_pruning_parameters_are_rejected(field, value):
    parameters = load_default_parameters()
    parameters.mcmc_prune_steps.append(1000)
    setattr(parameters, field, value)
    with pytest.raises(ValueError):
        validate_pruning(parameters)


def test_retention_uses_global_ranking_and_preserves_original_row_order():
    scores = np.array([10, 1, 4, 0, 8, 3])
    mask = removal_mask(scores, 0.5)
    assert np.flatnonzero(~mask).tolist() == [0, 2, 4]
    assert scores[~mask].tolist() == [10, 4, 8]


def test_ties_keep_original_order_and_small_models_keep_one_primitive():
    assert np.flatnonzero(~removal_mask([2, 2, 2, 1], 0.5)).tolist() == [0, 1]
    assert removal_mask([1], 0.1).tolist() == [False]


@pytest.mark.parametrize("scores", [[], [0, 0], [1, np.nan], [1, np.inf], [1, -1], [[1, 2]]])
def test_invalid_or_invisible_models_cannot_be_pruned(scores):
    with pytest.raises(ValueError):
        removal_mask(scores, 0.5)


@pytest.mark.parametrize("field,value", [
    ("patch_size", 128), ("with_eval3d", True),
    ("with_ut", True), ("camera_model", "fisheye"),
])
def test_incompatible_resolved_configuration_is_rejected(field, value):
    configuration = SimpleNamespace(patch_size=None, with_eval3d=False,
                                    with_ut=False, camera_model="pinhole")
    setattr(configuration, field, value)
    with pytest.raises(ValueError):
        validate_pruning_configuration(configuration, 1)


@pytest.mark.parametrize("world_size", [0, 2])
def test_pruning_requires_one_gpu(world_size):
    configuration = SimpleNamespace(patch_size=None)
    with pytest.raises(ValueError, match="one GPU"):
        validate_pruning_configuration(configuration, world_size)


def test_training_mask_missing_or_all_black_preserves_camera_mask(tmp_path):
    assert load_training_mask(tmp_path, "face.jpg", (4, 4)) is None
    (tmp_path / "masks").mkdir()
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(tmp_path / "masks/face.jpg.png")
    assert load_training_mask(tmp_path, "face.jpg", (4, 4)) is None


def test_training_mask_uses_nearest_resize_and_threshold(tmp_path):
    (tmp_path / "masks/rig").mkdir(parents=True)
    values = np.array([[0, 127], [128, 255]], dtype=np.uint8)
    Image.fromarray(values).save(tmp_path / "masks/rig/face.jpg.png")
    mask = load_training_mask(tmp_path, "rig/face.jpg", (4, 4))
    np.testing.assert_array_equal(mask, np.repeat(np.repeat(values > 127, 2, axis=0), 2, axis=1))
