#!/usr/bin/env python3
"""Summarize gsplat parameter-sweep runs as a Markdown comparison report.

Scans a directory of simple_trainer result directories -- each holding cfg.yml,
stats/, ply/ and renders/ -- and writes one report comparing their evaluation
metrics, the point-cloud geometry their PLYs actually contain, the config
values that differ between them, and their renders side by side against the
ground truth. Runs still training are reported as pending rather than skipped.

The report lives under doc/ while the runs, point clouds and render thumbnails
stay under data/, so every artifact path is written relative to the report.
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

# Differ between runs without describing the settings under test: the step
# lists just track max_steps and the paths just track the run name.
IGNORED_PARAMETERS = frozenset(
    {"ckpt", "data_dir", "eval_steps", "ply_steps", "result_dir", "save_steps"}
)

# (heading, metric key, format, which value wins)
METRIC_COLUMNS = (
    ("Gaussians", "gaussians", "{:,.0f}", max),
    ("PSNR", "psnr", "{:.2f}", max),
    ("SSIM", "ssim", "{:.4f}", max),
    ("LPIPS", "lpips", "{:.4f}", min),
)

# As above, plus the note explaining the column.
GEOMETRY_COLUMNS = (
    ("Median opacity", "median_opacity", "{:.3f}", max,
     "Median of sigmoid(opacity). Near zero means the Gaussians barely contribute."),
    ("Faint <0.1", "faint_fraction", "{:.0%}", min,
     "Fraction of Gaussians with opacity below 0.1"),
    ("Median radius", "median_radius", "{:.4f}", max,
     "Median of mean(exp(scale)) in world units"),
    ("Radius / spacing", "radius_over_spacing", "{:.2f}", max,
     "Median radius divided by estimated median nearest-neighbour distance. "
     "Below ~0.5 the Gaussians are too small to tile a surface, leaving gaps."),
)

MISSING = "—"
THUMBNAIL_WIDTH = 420


class TagTolerantLoader(yaml.SafeLoader):
    """Reads simple_trainer's cfg.yml without importing gsplat.

    cfg.yml serializes the strategy as a `!!python/object:...` tag, which
    SafeLoader rejects and the unsafe loaders can only build with gsplat
    importable -- true in the container, not on the host.
    """


def _construct_tagged(loader, tag_suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


TagTolerantLoader.add_multi_constructor("", _construct_tagged)


def flatten_parameters(mapping, prefix=""):
    flattened = {}
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flattened.update(flatten_parameters(value, f"{name}."))
        else:
            flattened[name] = value
    return flattened


def step_number(name):
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else -1


def latest_step_file(directory, pattern):
    candidates = sorted(directory.glob(pattern), key=lambda path: step_number(path.name))
    return candidates[-1] if candidates else None


def read_json(path):
    if path is None:
        return {}
    with path.open() as handle:
        return json.load(handle)


def read_ply(path):
    """Return (property names, [vertices, properties] memory map)."""
    terminator = b"end_header\n"
    with path.open("rb") as handle:
        header = handle.read(8192)
        while terminator not in header:
            chunk = handle.read(8192)
            if not chunk:
                raise ValueError(f"{path}: truncated PLY header")
            header += chunk
    offset = header.index(terminator) + len(terminator)
    header = header[:offset]

    lines = header.decode("ascii", "replace").splitlines()
    names = [line.split()[-1] for line in lines if line.startswith("property")]
    count = next(
        int(line.split()[-1]) for line in lines if line.startswith("element vertex")
    )
    values = np.memmap(
        path, dtype="<f4", mode="r", offset=offset, shape=(count, len(names))
    )
    return names, values


def median_neighbour_distance(points):
    """Median nearest-neighbour distance, estimated from a random subsample.

    Distances within a subsample of fraction f scale as f**(-1/3) under locally
    uniform density, so the subsample's median is corrected by f**(1/3).
    """
    count = len(points)
    if count < 2:
        return None
    generator = np.random.default_rng(0)
    size = min(12_000, count)
    sample = points[generator.choice(count, size, replace=False)].astype("<f4")

    square_norms = (sample**2).sum(axis=1)
    nearest = np.empty(size, "<f4")
    for start in range(0, size, 2048):
        block = sample[start : start + 2048]
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab, which avoids materializing the
        # [block, sample, 3] difference tensor.
        squared = (
            square_norms[start : start + len(block), None]
            + square_norms[None, :]
            - 2.0 * (block @ sample.T)
        )
        np.fill_diagonal(squared[:, start : start + len(block)], np.inf)
        nearest[start : start + len(block)] = np.sqrt(
            np.maximum(squared.min(axis=1), 0.0)
        )

    return float(np.median(nearest) * (size / count) ** (1 / 3))


def ply_geometry(path):
    names, values = read_ply(path)
    # The PLY is row-major, so each column read pages in the whole file. Gather
    # every column needed in one pass instead of one pass apiece.
    wanted = [names.index(name) for name in ("x", "y", "z", "opacity")]
    wanted += [names.index(f"scale_{axis}") for axis in range(3)]
    gathered = np.asarray(values[:, wanted], "<f4")

    opacity = 1.0 / (1.0 + np.exp(-gathered[:, 3]))
    radius = np.exp(gathered[:, 4:]).mean(axis=1)
    spacing = median_neighbour_distance(gathered[:, :3])
    median_radius = float(np.median(radius))
    return {
        "gaussians": len(values),
        "median_opacity": float(np.median(opacity)),
        "faint_fraction": float((opacity < 0.1).mean()),
        "median_radius": median_radius,
        "neighbour_spacing": spacing,
        "radius_over_spacing": median_radius / spacing if spacing else None,
    }


def read_status(path):
    if not path.exists():
        return {}
    fields = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition(" ")
        fields[key] = value
    return fields


def collect_run(run_dir):
    with (run_dir / "cfg.yml").open() as handle:
        configuration = yaml.load(handle, Loader=TagTolerantLoader)

    validation = read_json(latest_step_file(run_dir / "stats", "val_step*.json"))
    training = read_json(latest_step_file(run_dir / "stats", "train_step*.json"))
    ply_path = latest_step_file(run_dir / "ply", "point_cloud_*.ply")
    status = read_status(run_dir / "sweep_status.txt")

    metrics = {
        "gaussians": validation.get("num_GS"),
        "psnr": validation.get("psnr"),
        "ssim": validation.get("ssim"),
        "lpips": validation.get("lpips"),
        "train_seconds": training.get("ellipse_time"),
        "peak_memory_gib": training.get("mem"),
    }
    geometry = ply_geometry(ply_path) if ply_path else {}
    metrics["gaussians"] = metrics["gaussians"] or geometry.get("gaussians")

    return {
        "name": run_dir.name,
        "parameters": flatten_parameters(configuration),
        "metrics": metrics,
        "geometry": geometry,
        "ply_path": ply_path,
        "renders": sorted((run_dir / "renders").glob("val_step*_*.png")),
        "video_path": latest_step_file(run_dir / "videos", "traj_*.mp4"),
        "log_path": run_dir / "train.log" if (run_dir / "train.log").exists() else None,
        "exit_status": status.get("exit_status"),
        "wall_seconds": float(status["wall_seconds"]) if "wall_seconds" in status else None,
        "complete": validation != {},
    }


def format_duration(seconds):
    if seconds is None:
        return MISSING
    minutes, _ = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min" if hours else f"{minutes} min"


def format_value(value):
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def relative(path, base):
    return os.path.relpath(path, base).replace(os.sep, "/")


def is_current(thumbnail, source):
    return thumbnail.exists() and thumbnail.stat().st_mtime >= source.stat().st_mtime


def write_thumbnails(runs, assets_dir, view_count):
    """Split each chosen render into its ground-truth and predicted halves.

    simple_trainer writes validation renders as [ground truth | prediction]
    side by side; the report shows the ground truth once per view. A finished
    run's renders never change, so thumbnails are only rebuilt when stale --
    the report is regenerated repeatedly while other runs are still training.
    """
    rendered = max((len(run["renders"]) for run in runs), default=0)
    if not rendered:
        return [], {}
    indices = np.unique(np.linspace(0, rendered - 1, min(view_count, rendered), dtype=int))

    assets_dir.mkdir(parents=True, exist_ok=True)
    truth_paths, predicted_paths = [], {}
    for position, index in enumerate(indices):
        for run in runs:
            if index >= len(run["renders"]):
                continue
            source = run["renders"][index]
            truth = assets_dir / f"view{position:02d}_truth.jpg"
            predicted = assets_dir / f"view{position:02d}_{run['name']}.jpg"
            needs_truth = position >= len(truth_paths) and not is_current(truth, source)
            image = None

            if needs_truth or not is_current(predicted, source):
                image = Image.open(source)
                width, height = image.size
                scale = THUMBNAIL_WIDTH / (width // 2)
                size = (THUMBNAIL_WIDTH, int(height * scale))

            if position >= len(truth_paths):
                if needs_truth:
                    image.crop((0, 0, width // 2, height)).resize(
                        size, Image.LANCZOS
                    ).convert("RGB").save(truth, quality=88)
                truth_paths.append(truth)

            if image is not None and not is_current(predicted, source):
                image.crop((width // 2, 0, width, height)).resize(
                    size, Image.LANCZOS
                ).convert("RGB").save(predicted, quality=88)
            predicted_paths.setdefault(run["name"], {})[position] = predicted
    return truth_paths, predicted_paths


def best_values(runs, columns, source):
    best = {}
    for _, key, _, wins, *_ in columns:
        values = [run[source][key] for run in runs if run[source].get(key) is not None]
        best[key] = wins(values) if values else None
    return best


def value_cells(run, columns, best, source):
    cells = []
    for _, key, template, *_ in columns:
        value = run[source].get(key)
        if value is None:
            cells.append(MISSING)
            continue
        text = template.format(value)
        cells.append(f"**{text}**" if value == best[key] else text)
    return cells


def table(headings, rows):
    lines = [
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join("---" for _ in headings) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def metrics_section(runs, report_dir):
    quality_best = best_values(runs, METRIC_COLUMNS, "metrics")
    geometry_best = best_values(runs, GEOMETRY_COLUMNS, "geometry")

    rows = []
    for run in runs:
        row = [f"`{run['name']}`"]
        row += value_cells(run, METRIC_COLUMNS, quality_best, "metrics")
        row += value_cells(run, GEOMETRY_COLUMNS, geometry_best, "geometry")
        row.append(format_duration(run["metrics"]["train_seconds"]))

        artifacts = []
        if run["ply_path"] is not None:
            megabytes = run["ply_path"].stat().st_size / 1024**2
            artifacts.append(
                f"[ply {megabytes:.0f} MB]({relative(run['ply_path'], report_dir)})"
            )
        if run["video_path"] is not None:
            artifacts.append(f"[video]({relative(run['video_path'], report_dir)})")
        if run["log_path"] is not None:
            artifacts.append(f"[log]({relative(run['log_path'], report_dir)})")
        row.append(" · ".join(artifacts) or MISSING)
        rows.append(row)

    headings = [heading for heading, *_ in METRIC_COLUMNS + GEOMETRY_COLUMNS]
    notes = "\n".join(
        f"- **{heading}** — {note}" for heading, _, _, _, note in GEOMETRY_COLUMNS
    )
    return f"""## Metrics and point-cloud geometry

Reconstruction quality on the held-out views, then what the saved point cloud
actually contains. Best value per column in bold.

{table(["Run", *headings, "Train time", "Artifacts"], rows)}

{notes}
"""


def parameters_section(runs):
    names = sorted(
        {name for run in runs for name in run["parameters"]} - IGNORED_PARAMETERS
    )
    rows = []
    for name in names:
        values = [format_value(run["parameters"].get(name)) for run in runs]
        if len(set(values)) > 1:
            rows.append([f"`{name}`", *values])
    if not rows:
        return "## Settings under test\n\nEvery run used identical settings.\n"

    headings = ["Parameter", *(f"`{run['name']}`" for run in runs)]
    return f"""## Settings under test

Only the config values that differ between runs.

{table(headings, rows)}
"""


def renders_section(runs, truth_paths, predicted_paths, report_dir):
    """Ground truth beside each run's prediction, a few runs per table.

    Markdown has no grid, so wide comparisons are split into tables of a few
    columns each with the ground truth repeated as the first column.
    """
    if not truth_paths:
        return "## Renders\n\nNo validation renders written yet.\n"

    columns = 3

    blocks = []
    for position, truth in enumerate(truth_paths):
        shown = [run for run in runs if position in predicted_paths.get(run["name"], {})]
        tables = []
        for start in range(0, len(shown), columns):
            group = shown[start : start + columns]
            headings = ["ground truth", *(f"`{run['name']}`" for run in group)]
            cells = [f"![ground truth]({relative(truth, report_dir)})"]
            cells += [
                f"![{run['name']}]({relative(predicted_paths[run['name']][position], report_dir)})"
                for run in group
            ]
            tables.append(table(headings, [cells]))
        blocks.append(f"### View {position + 1}\n\n" + "\n\n".join(tables) + "\n")

    return (
        "## Renders\n\nHeld-out views, ground truth first. Compare sharpness of "
        "edges, text and thin structures.\n\n" + "\n".join(blocks)
    )


def pending_section(runs):
    pending = [run for run in runs if not run["complete"]]
    failed = [run for run in runs if run["exit_status"] not in (None, "0")]
    notes = []
    if pending:
        names = ", ".join(f"`{run['name']}`" for run in pending)
        notes.append(f"> **Still training or not yet evaluated:** {names}")
    if failed:
        names = ", ".join(
            f"`{run['name']}` (exit {run['exit_status']})" for run in failed
        )
        notes.append(
            f"> **Exited non-zero** (check the log, most likely out of GPU memory): {names}"
        )
    return "\n>\n".join(notes) + "\n" if notes else ""


def build_report(runs, report_path, truth_paths, predicted_paths, command):
    report_dir = report_path.parent
    sections = [
        "# gsplat parameter sweep",
        f"{len(runs)} configurations trained from the same cube-map reconstruction.",
        pending_section(runs),
        metrics_section(runs, report_dir),
        parameters_section(runs),
        renders_section(runs, truth_paths, predicted_paths, report_dir),
        f"---\n\nRegenerate with `{command}`.",
    ]
    return "\n\n".join(section for section in sections if section) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("data/panorama/sweep"))
    parser.add_argument("--output", type=Path,
                        default=Path("doc/gsplat_parameter_sweep.md"))
    parser.add_argument("--views", type=int, default=6,
                        help="how many held-out views to show (default: 6)")
    arguments = parser.parse_args()

    run_dirs = sorted(
        path for path in arguments.runs_root.iterdir()
        if path.is_dir() and (path / "cfg.yml").exists()
    )
    if not run_dirs:
        raise SystemExit(f"no run directories with a cfg.yml under {arguments.runs_root}")

    runs = [collect_run(run_dir) for run_dir in run_dirs]
    # Thumbnails are generated data, so they stay beside the runs rather than
    # under the repo-managed doc/ the report is written to.
    truth_paths, predicted_paths = write_thumbnails(
        runs, arguments.runs_root / "report_assets", arguments.views
    )

    command = (
        f"python3 mapping/summarize_gsplat_sweep.py --runs-root {arguments.runs_root}"
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        build_report(runs, arguments.output, truth_paths, predicted_paths, command)
    )
    print(f"Report written to {arguments.output} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
