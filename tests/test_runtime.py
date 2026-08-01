import torch
import pytest

from rl.utils.runtime import (
    RuntimeContext,
    create_runtime_context,
)


def test_create_runtime_context(monkeypatch):

    config = {
        "runtime": {
            "device": "cpu",
            "dtype": "float32",
            "num_threads": 4,
            "seed": 123,
            "deterministic": True,
        }
    }

    seed_called = {}

    def fake_set_seed(seed, deterministic):
        seed_called["seed"] = seed
        seed_called["deterministic"] = deterministic

    monkeypatch.setattr(
        "rl.utils.runtime.set_seed",
        fake_set_seed,
    )

    threads_called = {}

    def fake_set_num_threads(num_threads):
        threads_called["num_threads"] = num_threads

    monkeypatch.setattr(
        torch,
        "set_num_threads",
        fake_set_num_threads,
    )

    context = create_runtime_context(config)

    assert isinstance(context, RuntimeContext)

    assert context.device == torch.device("cpu")
    assert context.dtype == torch.float32
    assert context.num_threads == 4
    assert context.seed == 123
    assert context.deterministic is True

    assert seed_called == {
        "seed": 123,
        "deterministic": True,
    }

    assert threads_called["num_threads"] == 4


def test_create_runtime_context_with_invalid_dtype():

    config = {
        "runtime": {
            "device": "cpu",
            "dtype": "float16",
            "seed": 1,
            "deterministic": False,
        }
    }

    with pytest.raises(ValueError):
        create_runtime_context(config)


def test_create_runtime_context_with_zero_threads(monkeypatch):

    config = {
        "runtime": {
            "device": "cpu",
            "dtype": "float32",
            "num_threads": 0,
            "seed": 1,
            "deterministic": False,
        }
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

    monkeypatch.setattr(
        "rl.utils.runtime.set_seed",
        lambda *args, **kwargs: None,
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
def test_create_runtime_context_with_dtypes(dtype_name, expected, monkeypatch):

    config = {
        "runtime": {
            "device": "cpu",
            "dtype": dtype_name,
            "seed": 1,
            "deterministic": False,
        }
    }

    monkeypatch.setattr(
        "rl.utils.runtime.set_seed",
        lambda *args, **kwargs: None,
    )

    context = create_runtime_context(config)

    assert context.dtype == expected


def test_create_runtime_context_with_missing_runtime_config():

    with pytest.raises(ValueError):
        create_runtime_context({})