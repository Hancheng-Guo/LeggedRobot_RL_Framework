import torch
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from app.utils.context import RuntimeContext
from envs.simulators.utils.context import ModelContext
from envs.tasks.base import TaskContext
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
        model_context: ModelContext,
        terms: dict[str, dict[str, Any]],
        *args, **kwargs,
    ) -> None:

        self.num_envs = num_envs
        self.context = context
        self.model_context = model_context

        self.terms: dict[str, RewardTerm] = {}
        self.term_rewards: dict[str, torch.Tensor] = {}
        self._build_terms(terms)


    def _build_terms(
        self,
        terms: dict[str, dict[str, Any]],
    ) -> None:

        for name, config in terms.items():

            function = get_reward_function(name)

            weight: float = config.get("weight", 1.0)
            params: dict = config.get("params", {})

            if params is None or not isinstance(params, dict):
                params = {}

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
        params: dict[str, Any],
    ) -> None:

        if name in self.terms:
            raise ValueError(
                f"Reward term '{name}' already exists."
            )

        self.terms[name] = RewardTerm(
            function=function,
            weight=weight,
            params=params,
        )
        self.term_rewards[name] = torch.zeros(
            self.num_envs,
            device=self.context.device,
        )


    def compute(
        self,
        task_context: TaskContext,
    ) -> tuple[torch.Tensor, dict]:

        self.term_rewards.clear()
        weighted_reward_sum: torch.Tensor = torch.zeros(
            self.num_envs,
            device=self.context.device,
        )
        weighted_reward_mean = dict[str, torch.Tensor] = {}

        for name, term in self.terms.items():

            reward = term.function(
                task_context,
                self.model_context,
                **term.params,
            )
            weighted_reward = reward * term.weight
            self.term_rewards[name].copy_(weighted_reward)
            weighted_reward_sum += weighted_reward
            weighted_reward_mean[name] = weighted_reward.mean()
            
        return weighted_reward_sum, weighted_reward_mean


    def get_term_reward(
        self,
        name: str,
    ) -> torch.Tensor:

        return self.term_rewards[name]
    

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        if env_ids is None:
            for value in self.term_rewards.values():
                value.zero_()

            return

        for value in self.term_rewards.values():
            value[env_ids] = 0.0
