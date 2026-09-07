"""JobParameters text-protobuf helpers."""

from pathlib import Path

from google.protobuf import json_format, text_format

from gsplat_server.proto import gsplat_pb2

DEFAULT_PARAMETERS_PATH = (
    Path(__file__).resolve().parent / "config" / "gsplat_train_defaults.proto.txt"
)


def load_default_parameters():
    parameters = gsplat_pb2.JobParameters()
    text_format.Parse(DEFAULT_PARAMETERS_PATH.read_text(), parameters)
    return parameters


def load_parameters(path):
    """Read a parameter file, layered over the shipped defaults.

    A file only has to name what it changes; anything it leaves out keeps the
    default. Without this an omitted field would silently read as proto3's
    zero -- sh_degree 0, ssim_lambda 0.0 -- rather than the shipped value.
    """
    parameters = load_default_parameters()
    text_format.Merge(Path(path).read_text(), parameters)
    return parameters


def parameters_from_dict(data):
    parameters = gsplat_pb2.JobParameters()
    json_format.ParseDict(data, parameters)
    return parameters


def parameters_to_dict(parameters):
    return json_format.MessageToDict(
        parameters, preserving_proto_field_name=True
    )


def parameters_to_text(parameters):
    return text_format.MessageToString(parameters)
