import torch
from collections.abc import Callable


RewardFunction = Callable[..., torch.Tensor]


REWARD_FUNCTION_MAP: dict[str, RewardFunction] = {}


def register_reward(
    function: RewardFunction,
) -> RewardFunction:

    name = function.__name__

    if name in REWARD_FUNCTION_MAP:
        raise ValueError(
            f"Reward function '{name}' already registered."
        )

    REWARD_FUNCTION_MAP[name] = function

    return function


def get_reward_function(reward_name: str):

    function = REWARD_FUNCTION_MAP.get(reward_name, None)

    if function is None:
        raise ValueError(
            f"Unknown reward function: '{reward_name}'."
        )

    if not callable(function):
        raise TypeError(
            f"Reward function '{reward_name}' is not callable."
        )

    return function

