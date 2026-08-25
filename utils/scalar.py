import torch
from collections.abc import Mapping
from numbers import Real
from typing import Any


def scalar_metrics(info: Mapping[str, Any]) -> dict[str, float]:
    """Return numeric scalar entries and ignore structured diagnostics."""

    metrics: dict[str, float] = {}

    for name, value in info.items():

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                metrics[name] = float(value.detach().item())

        elif isinstance(value, Real) and not isinstance(value, bool):
            metrics[name] = float(value)

    return metrics


def scalar_value(value: Any) -> float | None:

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"{value!r} must be scalar."
            )
        return float(value.detach().item())
        
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
