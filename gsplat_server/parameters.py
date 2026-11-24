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


def parameters_to_complete_text(parameters):
    """Text that load_parameters reads back unchanged, whatever the defaults.

    Text format omits proto3 scalars at zero, and load_parameters would then
    fill them from the shipped defaults; the zeros are written out explicitly.
    """
    lines = [parameters_to_text(parameters).rstrip("\n")]
    for field in parameters.DESCRIPTOR.fields:
        if field.message_type is not None or field.is_repeated or getattr(parameters, field.name):
            continue
        if field.enum_type is not None:
            zero = field.enum_type.values_by_number[0].name
        elif field.type == field.TYPE_BOOL:
            zero = "false"
        else:
            zero = "0"
        lines.append(f"{field.name}: {zero}")
    return "\n".join(line for line in lines if line) + "\n"


def masking_enabled(parameters):
    """Whether training and the depth cloud read masks/."""
    return parameters.mask_people or parameters.mask_sky
