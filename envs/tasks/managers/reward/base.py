import torch
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from app.utils.context import RuntimeContext
from envs.tasks.managers.reward.terms.registry import get_reward_function


RewardFunction = Callable[..., torch.Tensor]


@dataclass
class RewardTerm:
    function: RewardFunction
    weight: float
    params: dict[str, Any]


class RewardManager:

    def __init__(
        self,
        num_envs: int,
        context: RuntimeContext,
        terms: dict[str, dict[str, Any]],
        *args, **kwargs,
    ) -> None:

        self.num_envs = num_envs
        self.context = context

        self.terms: dict[str, RewardTerm] = {}
        self._build_terms(terms)


    def _build_terms(
        self,
        terms: dict[str, dict[str, Any]],
    ) -> None:

        for name, config in terms.items():

            function = get_reward_function(name)

            weight: float = config.get("weight", 1.0)
            params: dict = config.get("params", {})

            self._add_term(
                name=name,
                function=function,
                weight=weight,
                params=params,
            )


    def _add_term(
        self,
        name: str,
        function: RewardFunction,
        weight: float,
        params: dict[str, Any] | None = None,
    ) -> None:

        if name in self.terms:
            raise ValueError(
                f"Reward term '{name}' already exists."
            )

        self.terms[name] = RewardTerm(
            name=name,
            function=function,
            weight=weight,
            params=params or {},
        )



    def remove_term(
        self,
        name: str,
    ) -> None:

        if name not in self.terms:
            raise KeyError(
                f"Reward term '{name}' does not exist."
            )

        del self.terms[name]

        self.term_rewards.pop(
            name,
            None,
        )

    def compute(
        self,
        context: Any,
    ) -> torch.Tensor:

        self.reward.zero_()
        self.term_rewards.clear()

        for name, term in self.terms.items():

            value = term.function(
                context,
                **term.params,
            )

            if value.shape != (self.num_envs,):
                raise ValueError(
                    f"Reward term '{name}' must return shape "
                    f"({self.num_envs},), but got {value.shape}."
                )

            weighted_reward = value * term.weight

            self.term_rewards[name] = weighted_reward

            self.reward += weighted_reward

        return self.reward

    def get_term_reward(
        self,
        name: str,
    ) -> torch.Tensor:

        if name not in self.term_rewards:
            raise KeyError(
                f"Reward term '{name}' has not been computed."
            )

        return self.term_rewards[name]

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        if env_ids is None:
            self.reward.zero_()

            for value in self.term_rewards.values():
                value.zero_()

            return

        self.reward[env_ids] = 0.0

        for value in self.term_rewards.values():
            value[env_ids] = 0.0