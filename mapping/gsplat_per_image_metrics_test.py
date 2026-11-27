"""Tests for the per-image metrics written beside simple_trainer's averages."""

import csv
import types

import pytest

from mapping.gsplat_per_image_metrics import record_per_image_metrics


class Runner:
    def __init__(self, stats_dir, color_correction):
        self.stats_dir = stats_dir
        self.world_rank = 0
        self.cfg = types.SimpleNamespace(use_color_correction_metric=color_correction)
        self.parser = types.SimpleNamespace(image_names=["a.jpg", "b.jpg", "c.jpg", "d.jpg"])
        self.valset = types.SimpleNamespace(indices=[0, 2])
        self.psnr = lambda render, target: 20.0 + render
        self.ssim = lambda render, target: 0.5 + render
        self.lpips = lambda render, target: 0.1 + render

    def eval(self, step, stage="val"):
        for image in range(len(self.valset.indices)):
            for render in (image, image + 0.25)[: 2 if self.cfg.use_color_correction_metric else 1]:
                self.psnr(render, None)
                self.ssim(render, None)
                self.lpips(render, None)


def rows(tmp_path, color_correction):
    runner_class = type("Runner", (Runner,), {})
    record_per_image_metrics(runner_class)
    runner = runner_class(str(tmp_path), color_correction)
    psnr = runner.psnr
    runner.eval(step=29999)
    assert runner.psnr is psnr
    with open(tmp_path / "val_step29999_per_image.csv") as file:
        return list(csv.reader(file))


def test_one_row_per_validation_image(tmp_path):
    assert rows(tmp_path, color_correction=False) == [
        ["image", "psnr", "ssim", "lpips"],
        ["a.jpg", "20", "0.5", "0.1"],
        ["c.jpg", "21", "1.5", "1.1"],
    ]


def test_color_corrected_columns(tmp_path):
    table = rows(tmp_path, color_correction=True)
    assert table[0] == ["image", "psnr", "ssim", "lpips", "cc_psnr", "cc_ssim", "cc_lpips"]
    assert table[2] == ["c.jpg", "21", "1.5", "1.1", "21.25", "1.75", "1.35"]


def test_mismatched_counts_raise(tmp_path):
    runner_class = type("Runner", (Runner,), {"eval": lambda self, step, stage="val": self.psnr(0, None)})
    record_per_image_metrics(runner_class)
    with pytest.raises(ValueError):
        runner_class(str(tmp_path), False).eval(step=1)
