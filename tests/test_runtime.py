import torch
import pytest

from runners.on_policy import OnPolicyRunner
from runners.off_policy import OffPolicyRunner

from app.utils.runtime import (
    RuntimeContext,
    create_runtime_context,
)


def test_create_runtime_context():

    config = {
        "device": "cpu",
        "dtype": "float32",
        "num_threads": 4,
        "seed": 123,
        "deterministic": True,
    }

    context = create_runtime_context(config)

    assert isinstance(context, RuntimeContext)

    assert context.device == torch.device("cpu")
    assert context.dtype == torch.float32
    assert context.num_threads == 4
    assert context.seed == 123
    assert context.deterministic is True


def test_create_runtime_context_with_invalid_dtype():

    config = {
        "device": "cpu",
        "dtype": "float16",
        "seed": 1,
        "deterministic": False,
    }

    with pytest.raises(ValueError):
        create_runtime_context(config)


def test_create_runtime_context_with_zero_threads(monkeypatch):

    config = {
        "device": "cpu",
        "dtype": "float32",
        "num_threads": 0,
        "seed": 1,
        "deterministic": False,
    }

    called = False

    def fake_set_num_threads(_):
        nonlocal called
        called = True

    monkeypatch.setattr(
        torch,
        "set_num_threads",
        fake_set_num_threads,
    )

    context = create_runtime_context(config)

    assert context.num_threads == 0
    assert called is False


@pytest.mark.parametrize(
    "dtype_name, expected",
    [
        ("float32", torch.float32),
        ("float64", torch.float64),
    ],
)
def test_create_runtime_context_with_dtypes(dtype_name, expected):

    config = {
        "device": "cpu",
        "dtype": dtype_name,
        "seed": 1,
        "deterministic": False,
    }

    context = create_runtime_context(config)

    assert context.dtype == expected


def test_create_runtime_context_with_missing_runtime_config():

    with pytest.raises(TypeError):
        create_runtime_context()


def test_create_runtime_context_with_empty_runtime_config():

    with pytest.raises(ValueError):
        create_runtime_context({})


def test_create_runtime_context_with_invalid_runner_type():

    config = {
        "device": "cpu",
        "dtype": "float16",
        "seed": 1,
        "deterministic": False,
    }

    with pytest.raises(ValueError):
        create_runtime_context(config)