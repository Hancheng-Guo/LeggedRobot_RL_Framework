import torch
from collections.abc import Mapping, Sequence
from typing import Any

from app.utils.context import RuntimeContext
from rl.policies.actor_critic import ActorCritic
from rl.policies.distributions import (
    BaseActionDistribution,
    build_action_distribution,
)
from rl.policies.modules.network import ConfigurableNetwork
from utils.component import Component


class ConfigurableActorCritic(ActorCritic):

    def __init__(
        self,
        context: RuntimeContext
    ) -> None:
        
        super().__init__(context=context)

        self.actor: ConfigurableNetwork
        self.critic: ConfigurableNetwork
        self.action_distribution: BaseActionDistribution


    def config_update(
        self,
        component: Component,
        action_dim: int,
        actor: Sequence[Mapping[str, Any]],
        critic: Sequence[Mapping[str, Any]],
        distribution: Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        
        if action_dim <= 0:
            raise ValueError("'action_dim' must be greater than 0.")
        if "action_dim" in distribution:
            raise ValueError(
                "Distribution 'action_dim' is provided by the environment."
            )

        dimensions = {"action_dim": action_dim}
        self.actor = ConfigurableNetwork(
            modules=actor,
            variables=dimensions,
        )
        self.critic = ConfigurableNetwork(modules=critic)
        distribution_config = dict(distribution)
        distribution_config["action_dim"] = action_dim
        self.action_distribution = build_action_distribution(
            distribution_config
        )
        self.to(
            device=self.context.device,
            dtype=self.context.dtype,
        )


    def actor_forward(
        self,
        obs: torch.Tensor
    ) -> torch.Tensor:
        return self.actor(obs)


    def critic_forward(
        self,
        obs: torch.Tensor
    ) -> torch.Tensor:
        return self.critic(obs)


    def get_action_distribution(
        self,
        obs: torch.Tensor,
    ) -> torch.distributions.Distribution:
        return self.action_distribution(self.actor_forward(obs))


    def close(self) -> None:
        return None
