import torch
from typing import Any

from envs.simulators.utils import ModelContext
from envs.tasks.managers.reward.terms import BaseRewardTerm, get_reward_class
from envs.tasks.utils import TaskContext
from utils import RuntimeContext


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
        self.raw_term_rewards: dict[str, torch.Tensor] = {}
        self.term_rewards: dict[str, torch.Tensor] = {}
        self.total_reward = torch.zeros(
            self.num_envs,
            dtype=self.context.dtype,
            device=self.context.device,
        )
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
            num_envs=self.num_envs,
            weight=weight,
            context=self.context,
            model_context=self.model_context,
            **params,
        )
        self.term_rewards[name] = torch.zeros(
            self.num_envs,
            dtype=self.context.dtype,
            device=self.context.device,
        )
        self.raw_term_rewards[name] = torch.zeros_like(self.term_rewards[name])


    def compute(
        self,
        task_context: TaskContext,
    ) -> tuple[torch.Tensor, dict]:

        weighted_reward_sum: torch.Tensor = torch.zeros(
            self.num_envs,
            dtype=self.context.dtype,
            device=self.context.device,
        )
        weighted_reward_info: dict[str, torch.Tensor] = {}

        for name, term in self.terms.items():

            reward = term.compute(task_context)
            self.raw_term_rewards[name].copy_(reward)
            weighted_reward = reward * term.weight
            self.term_rewards[name].copy_(weighted_reward)
            weighted_reward_sum += weighted_reward
            weighted_reward_info[f"reward/{name}"] = weighted_reward

        self.total_reward.copy_(weighted_reward_sum)
        weighted_reward_info["reward"] = weighted_reward_sum.mean()
            
        return weighted_reward_sum, weighted_reward_info


    def get_term_reward(
        self,
        name: str,
    ) -> torch.Tensor:

        return self.term_rewards[name]


    def get_raw_term_reward(
        self,
        name: str,
    ) -> torch.Tensor:

        return self.raw_term_rewards[name]
    

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        for term in self.terms.values():
            term.reset(env_ids)

        if env_ids is None:
            self.total_reward.zero_()
            for value in self.term_rewards.values():
                value.zero_()
            for value in self.raw_term_rewards.values():
                value.zero_()

            return

        self.total_reward[env_ids] = 0.0
        for value in self.term_rewards.values():
            value[env_ids] = 0.0
        for value in self.raw_term_rewards.values():
            value[env_ids] = 0.0
