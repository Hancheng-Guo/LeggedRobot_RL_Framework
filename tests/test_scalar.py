import pytest
import torch

from utils.scalar import scalar_metrics, scalar_value


def test_scalar_metrics_converts_numeric_scalars() -> None:
    info = {
        "scalar_tensor": torch.tensor(1.25, requires_grad=True),
        "single_element_tensor": torch.tensor([2]),
        "float": 3.5,
        "integer": 4,
    }

    assert scalar_metrics(info) == {
        "scalar_tensor": 1.25,
        "single_element_tensor": 2.0,
        "float": 3.5,
        "integer": 4.0,
    }


def test_scalar_metrics_ignores_structured_and_unsupported_values() -> None:
    info = {
        "vector": torch.tensor([1.0, 2.0]),
        "matrix": torch.ones(2, 2),
        "boolean": True,
        "text": "value",
        "mapping": {"nested": 1.0},
        "sequence": [1.0],
        "none": None,
    }

    assert scalar_metrics(info) == {}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (torch.tensor(1.5), 1.5),
        (torch.tensor([2.0]), 2.0),
        (3, 3.0),
        (4.5, 4.5),
    ],
)
def test_scalar_value_converts_supported_values(
    value: object,
    expected: float,
) -> None:
    assert scalar_value(value) == expected


@pytest.mark.parametrize("value", [True, "1.0", [1.0], None])
def test_scalar_value_returns_none_for_unsupported_values(
    value: object,
) -> None:
    assert scalar_value(value) is None


def test_scalar_value_rejects_multi_element_tensor() -> None:
    value = torch.tensor([1.0, 2.0])

    with pytest.raises(ValueError, match="must be scalar"):
        scalar_value(value)
