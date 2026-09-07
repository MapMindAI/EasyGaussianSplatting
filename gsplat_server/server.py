#!/usr/bin/env python3
"""gRPC service that trains a gsplat model from an uploaded COLMAP model."""
import argparse
import json
import os
import queue
import shutil
import stat
import subprocess
import threading
import time
import uuid
import zipfile
from concurrent import futures
from pathlib import Path

import grpc

from gsplat_server.parameters import parameters_from_dict, parameters_to_dict, parameters_to_text
from gsplat_server.proto import gsplat_pb2, gsplat_pb2_grpc

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_ENTRYPOINT = REPO_ROOT / "mapping" / "train_gsplat_with_masks.py"
SEGMENT_ENTRYPOINT = REPO_ROOT / "mapping" / "segment_people.py"
QUEUED, RUNNING, SUCCEEDED, FAILED = "queued", "running", "succeeded", "failed"


def clear_readonly_and_retry(operation, path, _exception):
    """Windows refuses to unlink a read-only file; clear the bit and retry."""
    os.chmod(path, stat.S_IWRITE)
    operation(path)


class JobStore:
    def __init__(self, root):
        self.root, self.lock, self.jobs = root, threading.Lock(), {}
        for job_file in sorted(root.glob("*/job.json")):
            job = json.loads(job_file.read_text())
            self.jobs[job["id"]] = job
            if job["state"] in (QUEUED, RUNNING):
                self.update(job, state=FAILED, error="interrupted by a server restart")

    def create(self, parameters):
        job = {"id": uuid.uuid4().hex[:12], "state": QUEUED,
               "parameters": parameters, "created_at": time.time(),
               "started_at": None, "finished_at": None, "error": None}
        self.directory(job["id"]).mkdir(parents=True)
        self.update(job)
        return job

    def directory(self, job_id):
        return self.root / job_id

    def update(self, job, **changes):
        job.update(changes)
        with self.lock:
            self.jobs[job["id"]] = job
            (self.directory(job["id"]) / "job.json").write_text(json.dumps(job))

    def get(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)

    def list(self):
        with self.lock:
            return sorted(self.jobs.values(), key=lambda job: job["created_at"])

    def remove(self, job):
        # The index entry is dropped only once the files are gone, so a job
        # whose directory survives stays listed instead of leaking silently:
        # each one holds the uploaded images and the checkpoints.
        directory = self.directory(job["id"])
        shutil.rmtree(directory, onerror=clear_readonly_and_retry)
        if directory.exists():
            raise OSError(f"job directory survived deletion: {directory}")
        with self.lock:
            del self.jobs[job["id"]]


def extract_model(archive_path, destination):
    staging = destination.parent / "staging"
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"unsafe path in archive: {name}")
        archive.extractall(staging)
    candidates = [staging, *(entry for entry in staging.iterdir() if entry.is_dir())]
    for candidate in candidates:
        if (candidate / "images").is_dir() and (candidate / "sparse").is_dir():
            candidate.rename(destination)
            shutil.rmtree(staging, ignore_errors=True)
            return
    shutil.rmtree(staging, ignore_errors=True)
    raise ValueError("archive contains no directory with both images/ and sparse/")


def result_ply(job_directory):
    point_clouds = sorted((job_directory / "model" / "gsplat_output" / "ply").glob("*.ply"))
    return point_clouds[-1] if point_clouds else None


def training_command(job_directory):
    # The strategy subcommand and --pose_opt come from the parameters file,
    # which the entrypoint reads; the placeholder is what it replaces.
    return [
        "python3", str(TRAIN_ENTRYPOINT), "default",
        "--job_parameters", str(job_directory / "parameters.textproto"),
        "--data_dir", str(job_directory / "model"),
        # The client downloads only the point cloud, so rendering the
        # trajectory video would be minutes of GPU time nobody collects.
        "--save_ply", "--antialiased", "--disable_viewer", "--disable_video",
        "--result_dir", str(job_directory / "model" / "gsplat_output"),
    ]


def segmentation_command(model_directory):
    return [
        "python3", str(SEGMENT_ENTRYPOINT),
        str(model_directory / "images"), str(model_directory / "masks"),
    ]


def job_steps(job, directory):
    """The commands to run for a job, in order.

    Segmentation only runs when asked for and when the upload brought no masks
    of its own, so an uploaded masks/ is never overwritten.
    """
    model_directory = directory / "model"
    parameters = parameters_from_dict(job["parameters"])
    steps = []
    if parameters.run_segmentation and not (model_directory / "masks").is_dir():
        steps.append(("segmentation", segmentation_command(model_directory)))
    steps.append(("gsplat training", training_command(directory)))
    return steps


def train(store, job):
    store.update(job, state=RUNNING, started_at=time.time())
    directory = store.directory(job["id"])
    with (directory / "train.log").open("wb") as log:
        for step, command in job_steps(job, directory):
            returncode = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
            if returncode != 0:
                break
    error = None if returncode == 0 else f"{step} exited {returncode}"
    if returncode == 0 and result_ply(directory) is None:
        error, returncode = "training wrote no point cloud", 1
    store.update(
        job,
        state=SUCCEEDED if returncode == 0 else FAILED,
        finished_at=time.time(),
        error=error,
    )


def write_upload(requests, store, context):
    parameters = None
    archive = None
    job = None
    try:
        for request in requests:
            if request.HasField("parameters"):
                if parameters is not None:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "parameters sent more than once")
                parameters = request.parameters
                job = store.create(parameters_to_dict(parameters))
                directory = store.directory(job["id"])
                (directory / "parameters.textproto").write_text(parameters_to_text(parameters))
                archive = (directory / "upload.zip").open("wb")
            elif request.HasField("archive_chunk"):
                if archive is None:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "parameters must be first")
                archive.write(request.archive_chunk)
    finally:
        if archive is not None:
            archive.close()
    if job is None:
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing job parameters")
    return job


def as_proto(job):
    return gsplat_pb2.Job(id=job["id"], state=job["state"],
                          parameters=parameters_from_dict(job["parameters"]),
                          created_at=job["created_at"], started_at=job["started_at"] or 0,
                          finished_at=job["finished_at"] or 0, error=job["error"] or "")


class GsplatService(gsplat_pb2_grpc.GsplatServiceServicer):
    def __init__(self, store, pending):
        self.store, self.pending = store, pending

    def _job(self, job_id, context):
        job = self.store.get(job_id)
        if job is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"no such job: {job_id}")
        return job

    def SubmitJob(self, requests, context):
        job = write_upload(requests, self.store, context)
        directory = self.store.directory(job["id"])
        archive_path = directory / "upload.zip"
        try:
            extract_model(archive_path, directory / "model")
        except (ValueError, zipfile.BadZipFile) as error:
            self.store.update(job, state=FAILED, finished_at=time.time(), error=str(error))
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        finally:
            archive_path.unlink(missing_ok=True)
        self.pending.put(job)
        return as_proto(job)

    def ListJobs(self, request, context):
        return gsplat_pb2.ListJobsResponse(jobs=[as_proto(job) for job in self.store.list()])

    def GetJob(self, request, context):
        return as_proto(self._job(request.id, context))

    def StreamLog(self, request, context):
        self._job(request.id, context)
        path = self.store.directory(request.id) / "train.log"
        if path.exists():
            offset = request.offset
            with path.open("rb") as log:
                log.seek(offset)
                # Training logs reach several MB, which as a single message
                # would exceed gRPC's default 4 MB receive limit.
                while data := log.read(1 << 20):
                    offset += len(data)
                    yield gsplat_pb2.LogChunk(data=data, offset=offset)

    def DownloadResult(self, request, context):
        job = self._job(request.id, context)
        if job["state"] != SUCCEEDED:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"job {request.id} is {job['state']}")
        point_cloud = result_ply(self.store.directory(request.id))
        if point_cloud is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "no result yet")
        with point_cloud.open("rb") as result:
            while data := result.read(1 << 20):
                yield gsplat_pb2.ResultChunk(data=data)

    def DeleteJob(self, request, context):
        job = self._job(request.id, context)
        if job["state"] in (QUEUED, RUNNING):
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"job {request.id} is {job['state']}")
        try:
            self.store.remove(job)
        except OSError as error:
            context.abort(grpc.StatusCode.INTERNAL, str(error))
        return gsplat_pb2.DeleteJobResponse(id=request.id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()
    args.jobs_dir.mkdir(parents=True, exist_ok=True)
    store, pending = JobStore(args.jobs_dir), queue.Queue()

    def worker():
        while True:
            train(store, pending.get())
    threading.Thread(target=worker, daemon=True).start()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    gsplat_pb2_grpc.add_GsplatServiceServicer_to_server(GsplatService(store, pending), server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    print(f"gsplat gRPC server listening on {args.host}:{args.port}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
