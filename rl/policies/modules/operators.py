import torch
from collections.abc import Sequence


class Add(torch.nn.Module):
    """Element-wise sum of equally-sized concatenated inputs."""

    def __init__(
        self,
        input_features: Sequence[int],
    ) -> None:
        super().__init__()

        if isinstance(input_features, (str, bytes)):
            raise TypeError("'input_features' must be a sequence of integers.")
        dimensions = tuple(input_features)
        if len(dimensions) < 2:
            raise ValueError("Add requires at least two inputs.")
        if any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
            for dimension in dimensions
        ):
            raise ValueError(
                "Add input feature dimensions must be positive integers."
            )
        if len(set(dimensions)) != 1:
            raise ValueError(
                "Add inputs must have identical feature dimensions, got "
                f"{dimensions}."
            )

        self.input_features = dimensions
        self.in_features = sum(dimensions)
        self.out_features = dimensions[0]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim == 0 or input.shape[-1] != self.in_features:
            raise ValueError(
                "Add input must have final dimension "
                f"{self.in_features}, got {tuple(input.shape)}."
            )

        values = torch.split(input, list(self.input_features), dim=-1)
        result = values[0]
        for value in values[1:]:
            result = result + value
        return result
