"""Run in the gsplat GPU image; the slim CI image skips this module."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA gsplat")

from gsplat_server.parameters import load_default_parameters
from mapping import gsplat_pruning


def make_splats(count):
    means = torch.zeros(count, 3, device="cuda")
    means[:, 0] = torch.arange(count, device="cuda") * 0.2
    means[:, 2] = 3
    quaternions = torch.zeros(count, 4, device="cuda")
    quaternions[:, 0] = 1
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(torch.full((count, 3), -2.0, device="cuda")),
        "quats": torch.nn.Parameter(quaternions),
        "opacities": torch.nn.Parameter(torch.zeros(count, device="cuda")),
        "sh0": torch.nn.Parameter(torch.zeros(count, 1, 3, device="cuda")),
        "shN": torch.nn.Parameter(torch.zeros(count, 8, 3, device="cuda")),
    })


def test_pruning_preserves_adam_and_scene_state_while_mcmc_continues(tmp_path, monkeypatch):
    from gsplat.scene import GaussianScene
    from gsplat.strategy import MCMCStrategy

    parameters = load_default_parameters()
    parameters.mcmc_prune_steps.extend([2, 4])
    parameters.mcmc_prune_keep_fraction = 0.5

    class Runner:
        def __init__(self):
            self.splats = make_splats(6)
            self.optimizers = {name: torch.optim.Adam([value], lr=0.01)
                               for name, value in self.splats.items()}
            self.scene = GaussianScene.from_splats(self.splats, id="test")
            self.scene.signal["row"] = torch.arange(6, device="cuda")
            self.cfg = SimpleNamespace(
                strategy=MCMCStrategy(cap_max=6, noise_lr=0, refine_start_iter=0,
                                      refine_every=1, refine_stop_iter=100),
                patch_size=None, with_eval3d=False, with_ut=False,
                camera_model="pinhole", result_dir=str(tmp_path), max_steps=5,
            )
            self.world_size, self.device, self.trainset = 1, "cuda", [0]

    monkeypatch.setattr(gsplat_pruning, "score_gaussians",
                        lambda runner, _: np.arange(len(runner.splats["means"])) + 1)
    gsplat_pruning.attach_pruning({"Runner": Runner}, parameters)
    runner = Runner()
    original_means = runner.splats["means"].detach().clone()
    state = runner.cfg.strategy.initialize_state()
    binomials = state["binoms"].to("cuda")
    state["binoms"] = binomials
    for step in range(5):
        sum(value.sum() for value in runner.splats.values()).backward()
        for optimizer in runner.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        runner.cfg.strategy.step_post_backward(
            runner.splats, runner.optimizers, state, step, {}, 0.01, scene=runner.scene,
        )
        runner.scene.validate()

    assert len(runner.splats["means"]) == runner.cfg.strategy.cap_max == 2
    assert runner.scene.signal["row"].tolist() == [4, 5]
    torch.testing.assert_close(runner.splats["means"], original_means[[4, 5]] - 0.05)
    assert state["binoms"] is binomials
    for name, optimizer in runner.optimizers.items():
        parameter = runner.splats[name]
        assert optimizer.param_groups[0]["params"][0] is parameter
        assert len(optimizer.state) == 1
        assert optimizer.state[parameter]["step"].item() == 5
        assert optimizer.state[parameter]["exp_avg"].shape == parameter.shape
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg"],
                                   torch.full_like(parameter, 1 - 0.9**5))
    records = [json.loads(line) for line in (tmp_path / "pruning.jsonl").read_text().splitlines()]
    assert [(record["iteration"], record["before"], record["after"]) for record in records] == [
        (2, 6, 3), (4, 3, 2),
    ]


@pytest.mark.parametrize("antialiased", [False, True])
def test_scoring_maps_packed_ids_and_excludes_masked_or_invisible_hits(antialiased, monkeypatch, tmp_path):
    parameters = load_default_parameters()
    splats = make_splats(4)
    with torch.no_grad():
        splats["means"][2, 2] = -3
        splats["means"][3, 0] = 100
    parser = SimpleNamespace(
        data_dir=str(tmp_path), camera_ids=[1, 2], imsize_dict={2: (32, 32)},
        Ks_dict={2: np.array([[24., 0, 16], [0, 24., 16], [0, 0, 1]])},
        camtoworlds=np.stack([np.eye(4), np.eye(4)]),
        image_names=["heldout.jpg", "training.jpg"], mask_dict={2: None},
    )

    class Dataset:
        indices = np.array([1])

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, index):
            pytest.fail("scoring must not decode training RGB images")

    dataset = Dataset()
    dataset.parser = parser
    configuration = SimpleNamespace(
        packed=False, near_plane=0.01, far_plane=100, camera_model="pinhole",
        antialiased=antialiased, pose_opt=False,
    )
    runner = SimpleNamespace(splats=splats, cfg=configuration, trainset=dataset)
    dense_scores = gsplat_pruning.score_gaussians(runner, parameters)
    configuration.packed = True
    packed_scores = gsplat_pruning.score_gaussians(runner, parameters)
    np.testing.assert_array_equal(dense_scores, packed_scores)
    assert (packed_scores[:2] > 0).all()
    assert (packed_scores[2:] == 0).all()
    monkeypatch.setattr(gsplat_pruning, "MAXIMUM_REGION_HITS", 10)
    subdivided_scores = gsplat_pruning.score_gaussians(runner, parameters)
    np.testing.assert_array_equal(subdivided_scores, packed_scores)
    parser.mask_dict[2] = np.zeros((32, 32), dtype=bool)
    masked_scores = gsplat_pruning.score_gaussians(runner, parameters)
    assert (masked_scores == 0).all()
