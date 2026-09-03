import math
import torch


def explained_variance(
    values: torch.Tensor,
    returns: torch.Tensor,
) -> float:
    """Measure how much of the variance in returns is explained by values."""

    flat_values = values.detach().flatten()
    flat_returns = returns.detach().flatten()

    if flat_values.numel() != flat_returns.numel():
        raise ValueError("'values' and 'returns' must have the same size.")
    if flat_returns.numel() == 0:
        raise ValueError("Cannot compute explained variance from empty tensors.")

    returns_variance = torch.var(flat_returns, correction=0)
    if float(returns_variance.item()) == 0.0:
        return math.nan

    residual_variance = torch.var(
        flat_returns - flat_values,
        correction=0,
    )
    result = 1.0 - residual_variance / returns_variance
    return float(result.item())
