import torch
import pytest
from typing import Any
from pathlib import Path

from app.utils.context import RuntimeContext, create_runtime_context


def test_create_runtime_context():

    config = {
        "device": "cpu",
        "dtype": "float32",
        "num_threads": 4,
        "seed": 123,
        "deterministic": True,
    }

    context = create_runtime_context(
        config,
        load_dir=Path("./a/b"),
        save_dir=Path("./c/d"),
    )

    assert isinstance(context, RuntimeContext)

    assert context.device == torch.device("cpu")
    assert context.dtype == torch.float32
    assert context.num_threads == 4
    assert context.seed == 123
    assert context.deterministic is True
    assert context.load_dir == Path("./a/b")
    assert context.save_dir == Path("./c/d")


def test_create_runtime_context_with_invalid_dtype():

    config = {
        "device": "cpu",
        "dtype": "float16",
        "num_threads": 4,
        "seed": 123,
        "deterministic": True,
    }

    with pytest.raises(ValueError):
        create_runtime_context(
            config,
            load_dir=Path("./a/b"),
            save_dir=Path("./c/d"),
        )


def test_create_runtime_context_with_zero_threads(monkeypatch):

    config = {
        "device": "cpu",
        "dtype": "float32",
        "num_threads": 0,
        "seed": 123,
        "deterministic": True,
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

    context = create_runtime_context(
        config,
        load_dir=Path("./a/b"),
        save_dir=Path("./c/d"),
    )

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
        "num_threads": 4,
        "seed": 123,
        "deterministic": True,
    }

    context = create_runtime_context(
        config,
        load_dir=Path("./a/b"),
        save_dir=Path("./c/d"),
    )

    assert context.dtype == expected


def test_create_runtime_context_with_missing_runtime_config():

    with pytest.raises(TypeError):
        create_runtime_context()    # pyright: ignore[reportCallIssue]


def test_create_runtime_context_with_empty_runtime_config():

    with pytest.raises(ValueError):
        create_runtime_context(
            {},
            load_dir=Path("./a/b"),
            save_dir=Path("./c/d"),
        )


def test_create_runtime_context_with_invalid_load_dir():

    config = {
        "device": "cpu",
        "dtype": "float16",
        "num_threads": 4,
        "seed": 123,
        "deterministic": True,
    }

    invalid_dir: Any = "./a/b"

    with pytest.raises(ValueError):
        create_runtime_context(
            config,
            load_dir=invalid_dir,
            save_dir=Path("./c/d"),
        )


def test_create_runtime_context_with_invalid_save_dir():

    config = {
        "device": "cpu",
        "dtype": "float16",
        "num_threads": 4,
        "seed": 123,
        "deterministic": True,
    }

    invalid_dir: Any = "./c/d"

    with pytest.raises(ValueError):
        create_runtime_context(
            config,
            load_dir=Path("./a/b"),
            save_dir=invalid_dir,
        )