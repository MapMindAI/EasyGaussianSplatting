"""Tests for the JobParameters text-protobuf helpers."""

import pytest

from gsplat_server.parameters import (
    load_default_parameters,
    load_parameters,
    parameters_from_dict,
    parameters_to_dict,
    parameters_to_text,
)
from gsplat_server.proto import gsplat_pb2


def test_defaults_carry_the_shipped_values():
    parameters = load_default_parameters()
    assert parameters.iterations == 30000
    assert parameters.sh_degree == 3
    assert parameters.ssim_lambda == pytest.approx(0.5)
    assert parameters.strategy == gsplat_pb2.STRATEGY_MCMC
    assert parameters.absgrad is True


def test_omitted_fields_keep_the_default_rather_than_proto3_zero(tmp_path):
    """The layering load_parameters exists for: a partial file is not a reset."""
    path = tmp_path / "partial.textproto"
    path.write_text("iterations: 7000\n")

    parameters = load_parameters(path)

    assert parameters.iterations == 7000
    assert parameters.sh_degree == 3
    assert parameters.ssim_lambda == pytest.approx(0.5)
    assert parameters.strategy == gsplat_pb2.STRATEGY_MCMC


def test_named_fields_override_the_default(tmp_path):
    path = tmp_path / "override.textproto"
    path.write_text("strategy: STRATEGY_DEFAULT\nsh_degree: 1\nabsgrad: false\n")

    parameters = load_parameters(path)

    assert parameters.strategy == gsplat_pb2.STRATEGY_DEFAULT
    assert parameters.sh_degree == 1
    assert parameters.absgrad is False
    assert parameters.iterations == 30000


def test_unknown_field_is_rejected(tmp_path):
    path = tmp_path / "typo.textproto"
    path.write_text("iteration_count: 10\n")

    with pytest.raises(Exception):
        load_parameters(path)


def test_dict_round_trip_keeps_snake_case_names():
    parameters = parameters_from_dict({"iterations": 500, "sh_degree": 2})

    assert parameters.iterations == 500
    assert parameters_to_dict(parameters) == {"iterations": 500, "sh_degree": 2}


def test_dict_round_trip_preserves_the_enum():
    parameters = parameters_from_dict({"strategy": "STRATEGY_MCMC", "cap_max": 1000})

    assert parameters.strategy == gsplat_pb2.STRATEGY_MCMC
    assert parameters_to_dict(parameters)["strategy"] == "STRATEGY_MCMC"


def test_defaults_survive_a_text_round_trip(tmp_path):
    original = load_default_parameters()
    path = tmp_path / "round_trip.textproto"
    path.write_text(parameters_to_text(original))

    assert load_parameters(path) == original


def test_dict_of_defaults_reparses_to_the_same_message():
    original = load_default_parameters()

    assert parameters_from_dict(parameters_to_dict(original)) == original
