from typing import TypeVar

from envs.tasks.managers.reward.terms.base import BaseRewardTerm
from utils.string import camel_to_snake


REWARD_CLASS_MAP: dict[str, type[BaseRewardTerm]] = {}


RewardTermType = TypeVar("RewardTermType", bound=BaseRewardTerm)


def register_reward(
    cls: type[RewardTermType]
) -> type[RewardTermType]:

    name = camel_to_snake(cls.__name__)

    if name in REWARD_CLASS_MAP:
        raise ValueError(
            f"Reward term '{name}' is already registered."
        )

    REWARD_CLASS_MAP[name] = cls
    return cls


def get_reward_class(name: str) -> type[BaseRewardTerm]:

    if name not in REWARD_CLASS_MAP:
        raise ValueError(f"Unknown reward term '{name}'.")

    return REWARD_CLASS_MAP[name]
