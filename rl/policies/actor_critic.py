import torch
from abc import abstractmethod

from app.utils.context import RuntimeContext
from rl.policies.base import (
    BasePolicy,
    PolicyActionOutput,
    PolicyEvaluation,
)


class ActorCritic(BasePolicy):

    def __init__(
        self,
        context: RuntimeContext
    ) -> None:
        
        super().__init__(context=context)


    @abstractmethod
    def actor_forward(
        self,
        obs: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


    @abstractmethod
    def critic_forward(
        self,
        obs: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


    @abstractmethod
    def get_action_distribution(
        self,
        obs: torch.Tensor,
    ) -> torch.distributions.Distribution:
        raise NotImplementedError


    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> PolicyActionOutput:
        
        distribution = self.get_action_distribution(obs)
        action = distribution.mean if deterministic else distribution.sample()

        return PolicyActionOutput(
            action=action,
            log_prob=self._sum_action_dims(distribution.log_prob(action)),
            value=self.predict_values(obs),
        )


    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> PolicyEvaluation:
        
        distribution = self.get_action_distribution(obs)
        return PolicyEvaluation(
            log_prob=self._sum_action_dims(distribution.log_prob(actions)),
            entropy=self._sum_action_dims(distribution.entropy()),
            value=self.predict_values(obs),
        )


    def predict_values(
        self,
        obs: torch.Tensor
    ) -> torch.Tensor:
        
        value = self.critic_forward(obs)
        if value.ndim < 2 or value.shape[-1] != 1:
            raise ValueError(
                "Critic must output exactly one value per observation."
            )
        return value.squeeze(-1)


    @staticmethod
    def _sum_action_dims(
        value: torch.Tensor
    ) -> torch.Tensor:
        
        if value.ndim <= 1:
            return value
        return value.sum(dim=-1)
