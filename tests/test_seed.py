import pytest
import torch
from app.utils.seed import set_seed


@pytest.mark.parametrize(
    "seed, deterministic_ops",
    [
        (1, True),
        ("1", True),
        (None, True),
        (1, False),
        ("1", False),
        (None, False),
        (1, None),
        ("1", None),
        (None, None)
    ],
)


def test_set_legal_seed(seed, deterministic_ops):
    set_seed(seed, deterministic_ops)


@pytest.mark.parametrize(
    "seed, deterministic_ops",
    [
        (2**32, False),
        ("error", False),
        (-1, False),
    ],
)


def test_set_illegal_str_seed(seed, deterministic_ops):
    with pytest.raises(ValueError):
        set_seed(seed, deterministic_ops)


@pytest.mark.parametrize("deterministic_ops", [True, False])
def test_set_seed_configures_deterministic_ops(
    monkeypatch,
    deterministic_ops,
):
    configured_value = None

    def fake_use_deterministic_algorithms(value):
        nonlocal configured_value
        configured_value = value

    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        fake_use_deterministic_algorithms,
    )

    _, returned_value = set_seed(
        seed=1,
        deterministic_ops=deterministic_ops,
    )

    assert configured_value is deterministic_ops
    assert returned_value is deterministic_ops
