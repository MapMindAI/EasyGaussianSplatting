"""Tests for the gsplat gRPC service's job store, uploads and command building."""

import json
import queue
import zipfile

import grpc
import pytest

from gsplat_server import server
from gsplat_server.parameters import load_default_parameters, parameters_to_dict
from gsplat_server.proto import gsplat_pb2


class Aborted(Exception):
    def __init__(self, code, details):
        super().__init__(details)
        self.code, self.details = code, details


class FakeContext:
    """Stands in for a gRPC context, whose abort() raises rather than returns."""

    def abort(self, code, details):
        raise Aborted(code, details)


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "jobs"
    root.mkdir()
    return server.JobStore(root)


@pytest.fixture
def parameters():
    return parameters_to_dict(load_default_parameters())


def write_archive(path, names):
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, "content")
    return path


# --- JobStore -------------------------------------------------------------


def test_create_writes_a_queued_job_to_disk(store, parameters):
    job = store.create(parameters)

    assert job["state"] == server.QUEUED
    assert job["started_at"] is None
    on_disk = json.loads((store.directory(job["id"]) / "job.json").read_text())
    assert on_disk == job


def test_update_persists_the_change(store, parameters):
    job = store.create(parameters)

    store.update(job, state=server.SUCCEEDED, error=None)

    assert store.get(job["id"])["state"] == server.SUCCEEDED
    on_disk = json.loads((store.directory(job["id"]) / "job.json").read_text())
    assert on_disk["state"] == server.SUCCEEDED


def test_get_returns_none_for_an_unknown_job(store):
    assert store.get("nope") is None


def test_list_is_ordered_by_creation_time(store, parameters):
    first = store.create(parameters)
    second = store.create(parameters)
    store.update(first, created_at=2.0)
    store.update(second, created_at=1.0)

    assert [job["id"] for job in store.list()] == [second["id"], first["id"]]


@pytest.mark.parametrize("state", [server.QUEUED, server.RUNNING])
def test_a_restart_fails_the_jobs_that_were_in_flight(store, parameters, state):
    job = store.create(parameters)
    store.update(job, state=state)

    reloaded = server.JobStore(store.root).get(job["id"])

    assert reloaded["state"] == server.FAILED
    assert reloaded["error"] == "interrupted by a server restart"


@pytest.mark.parametrize("state", [server.SUCCEEDED, server.FAILED])
def test_a_restart_leaves_finished_jobs_alone(store, parameters, state):
    job = store.create(parameters)
    store.update(job, state=state, error=None)

    assert server.JobStore(store.root).get(job["id"])["state"] == state


def test_remove_deletes_the_directory_and_the_index_entry(store, parameters):
    job = store.create(parameters)
    directory = store.directory(job["id"])

    store.remove(job)

    assert not directory.exists()
    assert store.get(job["id"]) is None


# --- extract_model --------------------------------------------------------


def test_a_flat_archive_becomes_the_model_directory(tmp_path):
    archive = write_archive(tmp_path / "u.zip", ["images/a.png", "sparse/0/cameras.bin"])
    destination = tmp_path / "model"

    server.extract_model(archive, destination)

    assert (destination / "images" / "a.png").is_file()
    assert (destination / "sparse" / "0" / "cameras.bin").is_file()
    assert not (tmp_path / "staging").exists()


def test_a_single_wrapper_directory_is_stripped(tmp_path):
    archive = write_archive(
        tmp_path / "u.zip", ["capture/images/a.png", "capture/sparse/0/cameras.bin"]
    )
    destination = tmp_path / "model"

    server.extract_model(archive, destination)

    assert (destination / "images" / "a.png").is_file()
    assert not (destination / "capture").exists()


@pytest.mark.parametrize("name", ["../escape.txt", "nested/../../escape.txt", "/etc/passwd"])
def test_a_path_outside_the_destination_is_refused(tmp_path, name):
    archive = write_archive(tmp_path / "u.zip", ["images/a.png", "sparse/x.bin", name])

    with pytest.raises(ValueError, match="unsafe path in archive"):
        server.extract_model(archive, tmp_path / "model")

    assert not (tmp_path / "model").exists()


def test_an_archive_without_images_and_sparse_is_refused(tmp_path):
    archive = write_archive(tmp_path / "u.zip", ["images/a.png"])

    with pytest.raises(ValueError, match="no directory with both"):
        server.extract_model(archive, tmp_path / "model")

    assert not (tmp_path / "staging").exists()


def test_a_corrupt_archive_raises_badzipfile(tmp_path):
    archive = tmp_path / "u.zip"
    archive.write_bytes(b"not a zip")

    with pytest.raises(zipfile.BadZipFile):
        server.extract_model(archive, tmp_path / "model")


# --- results and commands -------------------------------------------------


def test_result_ply_is_none_before_training_writes_one(tmp_path):
    assert server.result_ply(tmp_path) is None


def test_result_ply_finds_the_saved_point_cloud(tmp_path):
    ply_dir = tmp_path / "model" / "gsplat_output" / "ply"
    ply_dir.mkdir(parents=True)
    (ply_dir / "point_cloud_30000.ply").write_bytes(b"ply")

    assert server.result_ply(tmp_path) == ply_dir / "point_cloud_30000.ply"


def test_training_command_points_at_the_job_directory(tmp_path):
    command = server.training_command(tmp_path)

    assert str(server.TRAIN_ENTRYPOINT) in command
    assert str(tmp_path / "parameters.textproto") in command
    assert str(tmp_path / "model") in command
    assert "--disable_viewer" in command and "--disable_video" in command


def test_segmentation_command_reads_images_and_writes_masks(tmp_path):
    command = server.segmentation_command(tmp_path)

    assert command[-2:] == [str(tmp_path / "images"), str(tmp_path / "masks")]


def test_training_is_the_only_step_when_segmentation_is_off(tmp_path, parameters):
    parameters["run_segmentation"] = False

    steps = server.job_steps({"parameters": parameters}, tmp_path)

    assert [name for name, _ in steps] == ["gsplat training"]


def test_segmentation_runs_before_training_when_asked_for(tmp_path, parameters):
    parameters["run_segmentation"] = True

    steps = server.job_steps({"parameters": parameters}, tmp_path)

    assert [name for name, _ in steps] == ["segmentation", "gsplat training"]


def test_an_uploaded_masks_directory_is_never_overwritten(tmp_path, parameters):
    parameters["run_segmentation"] = True
    (tmp_path / "model" / "masks").mkdir(parents=True)

    steps = server.job_steps({"parameters": parameters}, tmp_path)

    assert [name for name, _ in steps] == ["gsplat training"]


def test_as_proto_maps_unset_timestamps_to_zero(store, parameters):
    job = store.create(parameters)
    store.update(job, created_at=1.5)

    proto = server.as_proto(job)

    assert (proto.id, proto.state) == (job["id"], server.QUEUED)
    assert proto.created_at == pytest.approx(1.5)
    assert proto.started_at == 0 and proto.finished_at == 0
    assert proto.error == ""


# --- write_upload ---------------------------------------------------------


def submit(parameters=None, chunks=()):
    requests = []
    if parameters is not None:
        requests.append(gsplat_pb2.SubmitJobRequest(parameters=parameters))
    requests += [gsplat_pb2.SubmitJobRequest(archive_chunk=chunk) for chunk in chunks]
    return requests


def test_write_upload_saves_the_archive_and_the_parameters(store):
    parameters = load_default_parameters()

    job = server.write_upload(
        submit(parameters, [b"abc", b"def"]), store, FakeContext()
    )

    directory = store.directory(job["id"])
    assert (directory / "upload.zip").read_bytes() == b"abcdef"
    assert "iterations: 30000" in (directory / "parameters.textproto").read_text()


def test_a_chunk_before_the_parameters_is_refused(store):
    with pytest.raises(Aborted) as error:
        server.write_upload(submit(chunks=[b"abc"]), store, FakeContext())

    assert error.value.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "parameters must be first" in error.value.details


def test_parameters_sent_twice_are_refused(store):
    parameters = load_default_parameters()
    requests = submit(parameters, [b"abc"]) + submit(parameters)

    with pytest.raises(Aborted, match="parameters sent more than once"):
        server.write_upload(requests, store, FakeContext())


def test_an_upload_without_parameters_is_refused(store):
    with pytest.raises(Aborted, match="missing job parameters"):
        server.write_upload(submit(), store, FakeContext())


# --- service methods ------------------------------------------------------


@pytest.fixture
def service(store):
    return server.GsplatService(store, queue.Queue())


def test_get_job_reports_an_unknown_id_as_not_found(service):
    with pytest.raises(Aborted) as error:
        service.GetJob(gsplat_pb2.GetJobRequest(id="missing"), FakeContext())

    assert error.value.code == grpc.StatusCode.NOT_FOUND


def test_list_jobs_returns_every_stored_job(service, store, parameters):
    store.create(parameters)
    store.create(parameters)

    response = service.ListJobs(gsplat_pb2.ListJobsRequest(), FakeContext())

    assert len(response.jobs) == 2


def test_a_queued_job_cannot_be_deleted(service, store, parameters):
    job = store.create(parameters)

    with pytest.raises(Aborted) as error:
        service.DeleteJob(gsplat_pb2.DeleteJobRequest(id=job["id"]), FakeContext())

    assert error.value.code == grpc.StatusCode.FAILED_PRECONDITION
    assert store.get(job["id"]) is not None


def test_a_finished_job_is_deleted(service, store, parameters):
    job = store.create(parameters)
    store.update(job, state=server.SUCCEEDED)

    response = service.DeleteJob(
        gsplat_pb2.DeleteJobRequest(id=job["id"]), FakeContext()
    )

    assert response.id == job["id"]
    assert store.get(job["id"]) is None


def test_the_result_of_an_unfinished_job_cannot_be_downloaded(service, store, parameters):
    job = store.create(parameters)

    with pytest.raises(Aborted, match="is queued"):
        list(service.DownloadResult(
            gsplat_pb2.DownloadResultRequest(id=job["id"]), FakeContext()
        ))


def test_a_succeeded_job_without_a_point_cloud_reports_no_result(service, store, parameters):
    job = store.create(parameters)
    store.update(job, state=server.SUCCEEDED)

    with pytest.raises(Aborted, match="no result yet"):
        list(service.DownloadResult(
            gsplat_pb2.DownloadResultRequest(id=job["id"]), FakeContext()
        ))


def test_the_point_cloud_is_streamed_back(service, store, parameters):
    job = store.create(parameters)
    store.update(job, state=server.SUCCEEDED)
    ply_dir = store.directory(job["id"]) / "model" / "gsplat_output" / "ply"
    ply_dir.mkdir(parents=True)
    (ply_dir / "point_cloud_30000.ply").write_bytes(b"point cloud")

    chunks = list(service.DownloadResult(
        gsplat_pb2.DownloadResultRequest(id=job["id"]), FakeContext()
    ))

    assert b"".join(chunk.data for chunk in chunks) == b"point cloud"


def test_the_log_is_streamed_from_the_requested_offset(service, store, parameters):
    job = store.create(parameters)
    (store.directory(job["id"]) / "train.log").write_bytes(b"0123456789")

    chunks = list(service.StreamLog(
        gsplat_pb2.StreamLogRequest(id=job["id"], offset=4), FakeContext()
    ))

    assert b"".join(chunk.data for chunk in chunks) == b"456789"
    assert chunks[-1].offset == 10


def test_streaming_the_log_of_a_job_that_has_not_started_yields_nothing(service, store, parameters):
    job = store.create(parameters)

    assert list(service.StreamLog(
        gsplat_pb2.StreamLogRequest(id=job["id"]), FakeContext()
    )) == []
