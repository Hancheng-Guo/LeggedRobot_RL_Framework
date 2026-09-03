import math
import torch
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, TypeVar


class BaseActionDistribution(torch.nn.Module, ABC):

    @abstractmethod
    def forward(
        self,
        parameters: torch.Tensor,
    ) -> torch.distributions.Distribution:
        raise NotImplementedError


class DiagonalGaussian(BaseActionDistribution):

    def __init__(
        self,
        action_dim: int,
        initial_std: float = 0.5,
        min_std: float = 0.1,
        max_std: float = 1.0,
    ) -> None:
        
        super().__init__()

        if action_dim <= 0:
            raise ValueError("'action_dim' must be greater than 0.")
        if initial_std <= 0.0:
            raise ValueError("'initial_std' must be greater than 0.")

        self.action_dim = action_dim
        self.log_std = torch.nn.Parameter(
            torch.full((action_dim,), math.log(initial_std))
        )

        self.min_log_std = math.log(min_std)
        self.max_log_std = math.log(max_std)


    def forward(
        self,
        parameters: torch.Tensor,
    ) -> torch.distributions.Normal:
        
        mean = parameters
        if mean.ndim < 2 or mean.shape[-1] != self.action_dim:
            raise ValueError(
                "Actor output must have shape [batch, action_dim] with "
                f"action_dim {self.action_dim}, got {tuple(mean.shape)}."
            )
        log_std = self.log_std.clamp(
            self.min_log_std,
            self.max_log_std,
        )
        std = log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)


ACTION_DISTRIBUTION_TYPE_MAP: dict[
    str,
    type[BaseActionDistribution],
] = {
    "diagonal_gaussian": DiagonalGaussian,
}


ActionDistributionType = TypeVar(
    "ActionDistributionType",
    bound=BaseActionDistribution,
)


def register_action_distribution(
    name: str,
) -> Callable[
    [type[ActionDistributionType]],
    type[ActionDistributionType],
]:
    if not name:
        raise ValueError("Distribution registration name cannot be empty.")

    def decorator(
        distribution_type: type[ActionDistributionType],
    ) -> type[ActionDistributionType]:
        if name in ACTION_DISTRIBUTION_TYPE_MAP:
            raise ValueError(
                f"Action distribution {name!r} is already registered."
            )
        ACTION_DISTRIBUTION_TYPE_MAP[name] = distribution_type
        return distribution_type

    return decorator


def build_action_distribution(
    config: Mapping[str, Any],
) -> BaseActionDistribution:
    
    distribution_config = dict(config)
    distribution_type_name = distribution_config.pop("type", None)

    if (
        not isinstance(distribution_type_name, str)
        or not distribution_type_name
    ):
        raise ValueError("Distribution config requires a non-empty 'type'.")
    if distribution_type_name not in ACTION_DISTRIBUTION_TYPE_MAP:
        valid_names = ", ".join(ACTION_DISTRIBUTION_TYPE_MAP)
        raise ValueError(
            f"Invalid action distribution: {distribution_type_name!r}. "
            f"Expected one of: {valid_names}."
        )

    distribution_type = ACTION_DISTRIBUTION_TYPE_MAP[
        distribution_type_name
    ]
    try:
        return distribution_type(**distribution_config)
    except TypeError as error:
        raise TypeError(
            f"Invalid parameters for action distribution "
            f"{distribution_type_name!r}: {distribution_config!r}."
        ) from error
