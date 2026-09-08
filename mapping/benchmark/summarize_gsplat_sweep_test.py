"""Tests for the gsplat parameter-sweep report generator."""

import json
import os

import numpy as np
import pytest
import yaml
from PIL import Image

from mapping.benchmark import summarize_gsplat_sweep as summarize

PLY_PROPERTIES = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2")


def write_ply(path, rows):
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {len(rows)}\n"
    header += "".join(f"property float {name}\n" for name in PLY_PROPERTIES)
    header += "end_header\n"
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        np.asarray(rows, "<f4").tofile(handle)
    return path


def gaussian(x=0.0, opacity=0.0):
    return [x, 0.0, 0.0, opacity, 0.0, 0.0, 0.0]


METRIC_KEYS = ("gaussians", "psnr", "ssim", "lpips", "train_seconds", "peak_memory_gib")


def run(name, metrics=None, **overrides):
    """A collected run, shaped the way collect_run returns one."""
    return {
        "name": name,
        "parameters": {},
        "metrics": {**dict.fromkeys(METRIC_KEYS), **(metrics or {})},
        "geometry": {},
        "ply_path": None,
        "renders": [],
        "video_path": None,
        "log_path": None,
        "exit_status": None,
        "wall_seconds": None,
        "complete": True,
        **overrides,
    }


# --- small helpers --------------------------------------------------------


def test_flatten_parameters_joins_nested_keys_with_dots():
    flattened = summarize.flatten_parameters(
        {"max_steps": 30000, "strategy": {"cap_max": 900000, "nested": {"deep": 1}}}
    )

    assert flattened == {
        "max_steps": 30000,
        "strategy.cap_max": 900000,
        "strategy.nested.deep": 1,
    }


@pytest.mark.parametrize(
    "name, expected",
    [("val_step6999.json", 6999), ("point_cloud_30000.ply", 30000), ("cfg.yml", -1)],
)
def test_step_number_reads_the_first_run_of_digits(name, expected):
    assert summarize.step_number(name) == expected


def test_latest_step_file_orders_by_step_not_by_name(tmp_path):
    """29999 sorts before 6999 lexicographically, so plain sorting picks the wrong file."""
    for step in (6999, 29999):
        (tmp_path / f"val_step{step}.json").write_text("{}")

    latest = summarize.latest_step_file(tmp_path, "val_step*.json")

    assert latest.name == "val_step29999.json"


def test_latest_step_file_is_none_when_nothing_matches(tmp_path):
    assert summarize.latest_step_file(tmp_path, "val_step*.json") is None


def test_read_json_of_a_missing_file_is_empty():
    assert summarize.read_json(None) == {}


@pytest.mark.parametrize(
    "seconds, expected", [(None, summarize.MISSING), (0, "0 min"), (59, "0 min"),
                          (90, "1 min"), (3660, "1 h 01 min"), (7200, "2 h 00 min")]
)
def test_format_duration(seconds, expected):
    assert summarize.format_duration(seconds) == expected


@pytest.mark.parametrize(
    "value, expected", [([1, 2, 3], "1, 2, 3"), (0.5, "0.5"), (True, "True")]
)
def test_format_value(value, expected):
    assert summarize.format_value(value) == expected


def test_relative_uses_forward_slashes(tmp_path):
    target = tmp_path / "runs" / "a.ply"

    assert summarize.relative(target, tmp_path) == "runs/a.ply"


def test_read_status_splits_each_line_on_the_first_space(tmp_path):
    path = tmp_path / "sweep_status.txt"
    path.write_text("exit_status 0\nwall_seconds 1234.5\n")

    assert summarize.read_status(path) == {"exit_status": "0", "wall_seconds": "1234.5"}


def test_read_status_of_a_missing_file_is_empty(tmp_path):
    assert summarize.read_status(tmp_path / "absent.txt") == {}


# --- PLY reading and geometry --------------------------------------------


def test_read_ply_returns_the_properties_and_the_values(tmp_path):
    path = write_ply(tmp_path / "a.ply", [gaussian(x=1.0), gaussian(x=2.0)])

    names, values = summarize.read_ply(path)

    assert names == list(PLY_PROPERTIES)
    assert values.shape == (2, len(PLY_PROPERTIES))
    assert values[1, 0] == pytest.approx(2.0)


def test_read_ply_rejects_a_truncated_header(tmp_path):
    path = tmp_path / "truncated.ply"
    path.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n")

    with pytest.raises(ValueError, match="truncated PLY header"):
        summarize.read_ply(path)


def test_ply_geometry_converts_opacity_and_scale(tmp_path):
    # opacity 0 is sigmoid 0.5; one Gaussian at -5 is faint. scale 0 is radius 1.
    path = write_ply(tmp_path / "a.ply", [
        gaussian(x=0.0, opacity=0.0),
        gaussian(x=1.0, opacity=0.0),
        gaussian(x=2.0, opacity=0.0),
        gaussian(x=3.0, opacity=-5.0),
    ])

    geometry = summarize.ply_geometry(path)

    assert geometry["gaussians"] == 4
    assert geometry["median_opacity"] == pytest.approx(0.5, abs=1e-3)
    assert geometry["faint_fraction"] == pytest.approx(0.25)
    assert geometry["median_radius"] == pytest.approx(1.0)
    assert geometry["neighbour_spacing"] == pytest.approx(1.0)
    assert geometry["radius_over_spacing"] == pytest.approx(1.0)


def test_ply_geometry_leaves_the_ratio_unset_when_every_point_coincides(tmp_path):
    path = write_ply(tmp_path / "a.ply", [gaussian(), gaussian(), gaussian()])

    geometry = summarize.ply_geometry(path)

    assert geometry["neighbour_spacing"] == pytest.approx(0.0)
    assert geometry["radius_over_spacing"] is None


def test_median_neighbour_distance_needs_two_points():
    assert summarize.median_neighbour_distance(np.zeros((1, 3), "<f4")) is None
    assert summarize.median_neighbour_distance(np.zeros((0, 3), "<f4")) is None


def test_median_neighbour_distance_recovers_a_regular_grid_spacing():
    grid = np.stack(np.meshgrid(*[np.arange(5.0)] * 3), -1).reshape(-1, 3)

    spacing = summarize.median_neighbour_distance(grid.astype("<f4"))

    assert spacing == pytest.approx(1.0, rel=0.15)


# --- table and cell formatting -------------------------------------------


def test_table_writes_a_markdown_header_and_separator():
    rendered = summarize.table(["A", "B"], [["1", "2"], ["3", "4"]])

    assert rendered.splitlines() == [
        "| A | B |", "| --- | --- |", "| 1 | 2 |", "| 3 | 4 |"
    ]


def test_best_values_takes_the_max_for_psnr_and_the_min_for_lpips():
    runs = [run("a", metrics={"psnr": 20.0, "lpips": 0.4}),
            run("b", metrics={"psnr": 25.0, "lpips": 0.2})]

    best = summarize.best_values(runs, summarize.METRIC_COLUMNS, "metrics")

    assert best["psnr"] == 25.0
    assert best["lpips"] == 0.2
    assert best["gaussians"] is None


def test_value_cells_bolds_the_winner_and_marks_the_gaps():
    runs = [run("a", metrics={"psnr": 20.0, "lpips": 0.4}),
            run("b", metrics={"psnr": 25.0, "lpips": 0.2})]
    best = summarize.best_values(runs, summarize.METRIC_COLUMNS, "metrics")

    cells = summarize.value_cells(runs[1], summarize.METRIC_COLUMNS, best, "metrics")

    assert cells == [summarize.MISSING, "**25.00**", summarize.MISSING, "**0.2000**"]


# --- report sections ------------------------------------------------------


def test_parameters_section_lists_only_the_values_that_differ():
    runs = [run("a", parameters={"sh_degree": 3, "max_steps": 30000}),
            run("b", parameters={"sh_degree": 1, "max_steps": 30000})]

    section = summarize.parameters_section(runs)

    assert "`sh_degree`" in section
    assert "`max_steps`" not in section


def test_parameters_section_ignores_the_run_specific_paths():
    runs = [run("a", parameters={"result_dir": "/runs/a", "data_dir": "/data/a"}),
            run("b", parameters={"result_dir": "/runs/b", "data_dir": "/data/b"})]

    assert "Every run used identical settings." in summarize.parameters_section(runs)


def test_parameters_section_says_so_when_nothing_differs():
    runs = [run("a", parameters={"sh_degree": 3}), run("b", parameters={"sh_degree": 3})]

    assert "Every run used identical settings." in summarize.parameters_section(runs)


def test_pending_section_is_empty_when_every_run_finished_cleanly():
    runs = [run("a", exit_status="0"), run("b", exit_status=None)]

    assert summarize.pending_section(runs) == ""


def test_pending_section_reports_unfinished_and_failed_runs():
    runs = [run("a", complete=False), run("b", exit_status="1")]

    section = summarize.pending_section(runs)

    assert "Still training or not yet evaluated:** `a`" in section
    assert "`b` (exit 1)" in section


def test_renders_section_says_so_when_no_renders_exist(tmp_path):
    assert "No validation renders written yet." in summarize.renders_section(
        [run("a")], [], {}, tmp_path
    )


def test_metrics_section_links_the_artifacts_relative_to_the_report(tmp_path):
    ply = write_ply(tmp_path / "point_cloud_30000.ply", [gaussian()])
    runs = [run("a", metrics={"psnr": 25.0, "train_seconds": 3660}, ply_path=ply)]

    section = summarize.metrics_section(runs, tmp_path)

    assert "[ply 0 MB](point_cloud_30000.ply)" in section
    assert "1 h 01 min" in section


def test_build_report_assembles_every_section(tmp_path):
    runs = [run("a", metrics={"psnr": 25.0}, parameters={"sh_degree": 3})]

    report = summarize.build_report(runs, tmp_path / "report.md", [], {}, "the command")

    assert report.startswith("# gsplat parameter sweep")
    assert "1 configurations" in report
    assert "## Metrics and point-cloud geometry" in report
    assert "## Settings under test" in report
    assert "## Renders" in report
    assert "Regenerate with `the command`." in report
    assert report.endswith("\n")


# --- configuration loading and collection --------------------------------


def test_the_loader_reads_a_config_with_a_gsplat_python_tag():
    document = (
        "max_steps: 30000\n"
        "strategy: !!python/object:gsplat.strategy.mcmc.MCMCStrategy\n"
        "  cap_max: 900000\n"
    )

    configuration = yaml.load(document, Loader=summarize.TagTolerantLoader)

    assert configuration["max_steps"] == 30000
    assert configuration["strategy"] == {"cap_max": 900000}


def make_run_directory(root, name, validation=True):
    step = 30000
    run_dir = root / name
    (run_dir / "stats").mkdir(parents=True)
    (run_dir / "ply").mkdir()
    (run_dir / "cfg.yml").write_text(
        "max_steps: 30000\nstrategy: !!python/object:gsplat.strategy.MCMC\n  cap_max: 900000\n"
    )
    if validation:
        (run_dir / "stats" / f"val_step{step}.json").write_text(
            json.dumps({"num_GS": 12345, "psnr": 25.5, "ssim": 0.9, "lpips": 0.15})
        )
    (run_dir / "stats" / f"train_step{step}.json").write_text(
        json.dumps({"ellipse_time": 3600.0, "mem": 12.5})
    )
    write_ply(run_dir / "ply" / f"point_cloud_{step}.ply", [gaussian(), gaussian(x=1.0)])
    (run_dir / "sweep_status.txt").write_text("exit_status 0\nwall_seconds 3700.0\n")
    return run_dir


def test_collect_run_gathers_the_metrics_and_the_artifacts(tmp_path):
    run_dir = make_run_directory(tmp_path, "mcmc_900k")

    collected = summarize.collect_run(run_dir)

    assert collected["name"] == "mcmc_900k"
    assert collected["complete"] is True
    assert collected["exit_status"] == "0"
    assert collected["wall_seconds"] == pytest.approx(3700.0)
    assert collected["metrics"]["gaussians"] == 12345
    assert collected["metrics"]["psnr"] == pytest.approx(25.5)
    assert collected["metrics"]["peak_memory_gib"] == pytest.approx(12.5)
    assert collected["parameters"]["strategy.cap_max"] == 900000
    assert collected["geometry"]["gaussians"] == 2
    assert collected["ply_path"].name == "point_cloud_30000.ply"
    assert collected["log_path"] is None


def test_collect_run_of_a_run_still_training_falls_back_to_the_point_cloud(tmp_path):
    run_dir = make_run_directory(tmp_path, "pending", validation=False)

    collected = summarize.collect_run(run_dir)

    assert collected["complete"] is False
    assert collected["metrics"]["psnr"] is None
    # No val_step*.json yet, so the count comes from the PLY the trainer saved.
    assert collected["metrics"]["gaussians"] == 2


# --- thumbnails -----------------------------------------------------------


def write_render(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 10), "red").save(path)
    return path


def test_write_thumbnails_splits_the_render_into_truth_and_prediction(tmp_path):
    render = write_render(tmp_path / "a" / "renders" / "val_step30000_0000.png")
    runs = [run("a", renders=[render])]

    truth_paths, predicted_paths = summarize.write_thumbnails(
        runs, tmp_path / "assets", view_count=1
    )

    assert [path.name for path in truth_paths] == ["view00_truth.jpg"]
    assert predicted_paths["a"][0].name == "view00_a.jpg"
    assert Image.open(truth_paths[0]).size == (summarize.THUMBNAIL_WIDTH, 210)


def test_write_thumbnails_does_nothing_without_renders(tmp_path):
    assert summarize.write_thumbnails([run("a")], tmp_path / "assets", 6) == ([], {})
    assert not (tmp_path / "assets").exists()


def test_thumbnails_are_left_alone_when_they_are_newer_than_the_render(tmp_path):
    render = write_render(tmp_path / "a" / "renders" / "val_step30000_0000.png")
    runs = [run("a", renders=[render])]
    assets = tmp_path / "assets"
    summarize.write_thumbnails(runs, assets, view_count=1)
    thumbnail = assets / "view00_a.jpg"
    stamp = thumbnail.stat().st_mtime_ns

    summarize.write_thumbnails(runs, assets, view_count=1)

    assert thumbnail.stat().st_mtime_ns == stamp


def test_is_current_compares_modification_times(tmp_path):
    source = write_render(tmp_path / "source.png")
    thumbnail = tmp_path / "thumbnail.jpg"

    assert not summarize.is_current(thumbnail, source)

    thumbnail.write_bytes(b"jpeg")
    os.utime(thumbnail, ns=(source.stat().st_mtime_ns + 10**9,) * 2)

    assert summarize.is_current(thumbnail, source)
