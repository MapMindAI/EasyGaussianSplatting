#!/usr/bin/env python3
"""Command-line client for the gsplat gRPC training service."""
import argparse
import sys
import tempfile
import time
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grpc
from gsplat_server.parameters import load_default_parameters, load_parameters
from gsplat_server.proto import gsplat_pb2, gsplat_pb2_grpc

UPLOADED_SUBDIRECTORIES = ("images", "sparse", "masks")
CHUNK_SIZE = 1 << 20


def archive_model(model_dir, archive_path):
    present = [name for name in UPLOADED_SUBDIRECTORIES if (model_dir / name).is_dir()]
    if "images" not in present or "sparse" not in present:
        raise SystemExit(f"{model_dir} needs both an images/ and a sparse/ directory")
    # The payload is JPEG images and PNG masks, so deflating them spends
    # CPU over hundreds of MB for no useful size reduction.
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
        for name in present:
            for path in sorted((model_dir / name).rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(model_dir))
    print(f"Packed {', '.join(present)} into {archive_path.stat().st_size / 1e6:.0f} MB")


def submit_requests(archive_path, parameters):
    yield gsplat_pb2.SubmitJobRequest(parameters=parameters)
    with archive_path.open("rb") as archive:
        while chunk := archive.read(CHUNK_SIZE):
            yield gsplat_pb2.SubmitJobRequest(archive_chunk=chunk)


def connect(server):
    return gsplat_pb2_grpc.GsplatServiceStub(grpc.insecure_channel(server))


def write_to_stdout(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def follow(stub, job_id, on_log=write_to_stdout, poll_seconds=5):
    """Poll a job to completion, passing each new slice of its log to on_log."""
    offset = 0
    while True:
        job = stub.GetJob(gsplat_pb2.GetJobRequest(id=job_id))
        for chunk in stub.StreamLog(gsplat_pb2.StreamLogRequest(id=job_id, offset=offset)):
            offset = chunk.offset
            on_log(chunk.data.decode("utf-8", "replace"))
        if job.state in ("succeeded", "failed"):
            return job
        time.sleep(poll_seconds)


def result_bytes(stub, job_id):
    return b"".join(
        chunk.data
        for chunk in stub.DownloadResult(gsplat_pb2.DownloadResultRequest(id=job_id))
    )


def submit(stub, args):
    with tempfile.TemporaryDirectory() as staging:
        archive_path = Path(staging) / "model.zip"
        archive_model(args.model_dir, archive_path)
        parameters = load_parameters(args.parameters) if args.parameters else load_default_parameters()
        job = stub.SubmitJob(submit_requests(archive_path, parameters))
    print(f"Submitted job {job.id}")
    job = follow(stub, job.id)
    if job.state != "succeeded":
        raise SystemExit(f"job {job.id} failed: {job.error}")
    fetch(stub, job.id, args.output or Path(f"{job.id}.ply"))
    stub.DeleteJob(gsplat_pb2.DeleteJobRequest(id=job.id))
    print(f"Removed job {job.id}")


def fetch(stub, job_id, output_path):
    output_path.write_bytes(result_bytes(stub, job_id))
    print(f"Point cloud written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--server", default="localhost:50051", help="gRPC host:port")
    parser.add_argument("--output", type=Path, help="output PLY path")
    parser.add_argument("--parameters", type=Path, help="text-protobuf training parameters")
    args = parser.parse_args()
    submit(connect(args.server), args)


if __name__ == "__main__":
    main()
