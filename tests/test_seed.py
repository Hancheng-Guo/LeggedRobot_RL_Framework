import pytest
import torch
from rl.utils.seed import set_seed


@pytest.mark.parametrize(
    "seed, deterministic",
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
def test_set_legal_seed(seed, deterministic):
    set_seed(seed, deterministic)


@pytest.mark.parametrize(
    "seed, deterministic",
    [
        (2**32, False),
        ("error", False),
        (-1, False),
    ],
)
def test_set_illegal_str_seed(seed, deterministic):
    with pytest.raises(ValueError):
        set_seed(seed, deterministic)