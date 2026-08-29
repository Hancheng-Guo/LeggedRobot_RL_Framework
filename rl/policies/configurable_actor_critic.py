import torch
from collections.abc import Mapping, Sequence
from typing import Any

from app.utils.context import RuntimeContext
from rl.policies.actor_critic import ActorCritic
from rl.policies.base import PolicyEvaluation, RecurrentState
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
        obs_dim: int,
        action_dim: int,
        actor: Sequence[Mapping[str, Any]],
        critic: Sequence[Mapping[str, Any]],
        distribution: Mapping[str, Any],
        *args, **kwargs,
    ) -> None:
        
        if obs_dim <= 0:
            raise ValueError("'obs_dim' must be greater than 0.")
        if action_dim <= 0:
            raise ValueError("'action_dim' must be greater than 0.")
        if "action_dim" in distribution:
            raise ValueError(
                "Distribution 'action_dim' is provided by the environment."
            )

        dimensions = {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
        }
        self.actor = ConfigurableNetwork(
            modules=actor,
            variables=dimensions,
        )
        self.critic = ConfigurableNetwork(
            modules=critic,
            variables=dimensions,
        )
        if self.critic.is_recurrent:
            raise ValueError(
                "Recurrent critic modules are not supported yet."
            )
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


    @property
    def is_recurrent(self) -> bool:
        return self.actor.is_recurrent


    def get_recurrent_state(
        self,
        batch_size: int | None = None,
    ) -> RecurrentState:
        return self.actor.get_recurrent_state(
            batch_size=batch_size,
            device=self.context.device,
            dtype=self.context.dtype,
        )


    def reset_recurrent_state(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self.actor.reset_recurrent_state(env_ids)


    def evaluate_recurrent_sequences(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        initial_state: RecurrentState,
        reset_mask: torch.Tensor,
    ) -> tuple[PolicyEvaluation, RecurrentState]:
        
        if not self.is_recurrent:
            raise RuntimeError("Policy actor is not recurrent.")

        actor_output, final_state = self.actor.forward_sequence(
            inputs=obs,
            initial_state=initial_state,
            reset_mask=reset_mask,
        )
        distribution = self.action_distribution(actor_output)
        evaluation = PolicyEvaluation(
            log_prob=self._sum_action_dims(
                distribution.log_prob(actions)
            ),
            entropy=self._sum_action_dims(distribution.entropy()),
            value=self.predict_values(obs),
        )
        return evaluation, final_state


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
        self.reset_recurrent_state()
