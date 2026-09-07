#!/usr/bin/env python3
"""Run the parameter sweep against a gsplat training server.

Submits one job per configuration file, sequentially, and writes each result
into <results>/<name>/ in the layout summarize_gsplat_sweep.py reads. The
server returns a point cloud and a training log, so the metrics come from the
log rather than from the stats files the local sweep leaves behind.

A configuration whose result already has metrics is skipped, so an interrupted
sweep resumes where it stopped.
"""
import argparse
import ast
import json
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from gsplat_server.client import archive_model, connect, follow, result_bytes, submit_requests
from gsplat_server.parameters import load_parameters, parameters_to_dict

EVALUATION = re.compile(
    r"PSNR:\s*([\d.]+),\s*SSIM:\s*([\d.]+),\s*LPIPS:\s*([\d.]+).*?Number of GS:\s*(\d+)"
)
TRAINING = re.compile(r"Step:\s*(\d+)\s*(\{.*?\})")


def parse_metrics(log):
    """Pull the evaluation and training summaries out of a finished job's log."""
    evaluation = EVALUATION.search(log)
    training = None
    for match in TRAINING.finditer(log):
        try:
            training = int(match.group(1)), ast.literal_eval(match.group(2))
        except (ValueError, SyntaxError):
            continue
    return evaluation, training


def write_run(directory, parameters, log, point_cloud, step):
    (directory / "stats").mkdir(parents=True, exist_ok=True)
    (directory / "ply").mkdir(parents=True, exist_ok=True)
    (directory / "train.log").write_text(log, errors="replace")
    # The report diffs cfg.yml across runs; the parameters are what differ here.
    (directory / "cfg.yml").write_text(yaml.safe_dump(parameters_to_dict(parameters)))

    evaluation, training = parse_metrics(log)
    if evaluation:
        psnr, ssim, lpips, count = evaluation.groups()
        (directory / "stats" / f"val_step{step}.json").write_text(json.dumps(
            {"psnr": float(psnr), "ssim": float(ssim), "lpips": float(lpips),
             "num_GS": int(count)}))
    if training:
        train_step, values = training
        (directory / "stats" / f"train_step{train_step}_rank0.json").write_text(
            json.dumps(values))
    if point_cloud is not None:
        (directory / "ply" / f"point_cloud_{step}.ply").write_bytes(point_cloud)


def run_configuration(stub, configuration, results_dir, archive_path):
    name = configuration.name.replace(".proto.txt", "")
    directory = results_dir / name
    if list(directory.glob("stats/val_step*.json")):
        print(f"=== {name}: already evaluated, skipping", flush=True)
        return
    parameters = load_parameters(configuration)

    print(f"=== {name}: submitting", flush=True)
    started = time.time()
    job = stub.SubmitJob(submit_requests(archive_path, parameters))
    log = []
    job = follow(stub, job.id, on_log=log.append, poll_seconds=15)
    log = "".join(log)

    point_cloud = None
    if job.state == "succeeded":
        point_cloud = result_bytes(stub, job.id)
    write_run(directory, parameters, log, point_cloud, parameters.iterations - 1)
    # The job is left on the server: its renders are wanted for the report and
    # DownloadResult only returns the point cloud. Delete the jobs once the
    # renders have been collected.
    (directory / "job_id").write_text(job.id)

    minutes = (time.time() - started) / 60
    size = f", {len(point_cloud) / 1e6:.0f} MB ply" if point_cloud else ""
    print(f"=== {name}: {job.state} in {minutes:.0f} min{size} {job.error}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--server", default="localhost:50051")
    parser.add_argument("--configurations", type=Path,
                        default=Path(__file__).resolve().parent / "configurations")
    parser.add_argument("--results", type=Path, required=True)
    arguments = parser.parse_args()

    configurations = sorted(arguments.configurations.glob("*.proto.txt"))
    if not configurations:
        raise SystemExit(f"no configurations under {arguments.configurations}")
    arguments.results.mkdir(parents=True, exist_ok=True)

    # The model is identical for every configuration, so it is packed once and
    # the same archive is re-uploaded per job.
    with tempfile.TemporaryDirectory() as staging:
        archive_path = Path(staging) / "model.zip"
        archive_model(arguments.model_dir, archive_path)
        stub = connect(arguments.server)
        for configuration in configurations:
            run_configuration(stub, configuration, arguments.results, archive_path)
    print(f"Sweep written to {arguments.results}")


if __name__ == "__main__":
    main()
