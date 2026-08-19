import torch
from typing import Any

from app.utils.context import RuntimeContext
from envs.simulators.utils.context import ModelContext
from envs.tasks.utils.context import TaskContext
from envs.tasks.managers.reward.terms.base import BaseRewardTerm
from envs.tasks.managers.reward.terms.registry import get_reward_class


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

        self.terms: dict[str, BaseRewardTerm] = {}
        self.term_rewards: dict[str, torch.Tensor] = {}
        self._build_terms(terms)


    def _build_terms(
        self,
        terms: dict[str, dict[str, Any]],
    ) -> None:

        for name, config in terms.items():

            weight: float = config.get("weight", 1.0)
            params: dict = config.get("params", {})

            if params is None or not isinstance(params, dict):
                params = {}

            cls = get_reward_class(name)
            self._add_term(
                name=name,
                cls=cls,
                weight=weight,
                params=params,
            )


    def _add_term(
        self,
        name: str,
        cls: type[BaseRewardTerm],
        weight: float,
        params: dict[str, Any],
    ) -> None:

        if name in self.terms:
            raise ValueError(
                f"Reward term '{name}' already exists."
            )

        self.terms[name] = cls(
            weight=weight,
            model_context=self.model_context,
            **params,
        )
        self.term_rewards[name] = torch.zeros(
            self.num_envs,
            dtype=self.context.dtype,
            device=self.context.device,
        )


    def compute(
        self,
        task_context: TaskContext,
    ) -> tuple[torch.Tensor, dict]:

        weighted_reward_sum: torch.Tensor = torch.zeros(
            self.num_envs,
            dtype=self.context.dtype,
            device=self.context.device,
        )
        weighted_reward_mean: dict[str, torch.Tensor] = {}

        for name, term in self.terms.items():

            reward = term.compute(task_context)
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

        for term in self.terms.values():
            term.reset(env_ids)

        if env_ids is None:
            for value in self.term_rewards.values():
                value.zero_()

            return

        for value in self.term_rewards.values():
            value[env_ids] = 0.0
