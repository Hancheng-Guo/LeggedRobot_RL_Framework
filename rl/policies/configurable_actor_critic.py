import torch
from collections.abc import Mapping, Sequence
from pathlib import Path
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
from utils.save import atomic_save


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
        observation_slices: Mapping[str, slice] | None = None,
        module_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
        *args, **kwargs,
    ) -> None:
        
        if (
            not isinstance(obs_dim, int)
            or isinstance(obs_dim, bool)
            or obs_dim <= 0
        ):
            raise ValueError("'obs_dim' must be a positive integer.")
        if (
            not isinstance(action_dim, int)
            or isinstance(action_dim, bool)
            or action_dim <= 0
        ):
            raise ValueError("'action_dim' must be a positive integer.")
        if not isinstance(distribution, Mapping):
            raise TypeError("'distribution' must be a mapping.")
        if "action_dim" in distribution:
            raise ValueError(
                "Distribution 'action_dim' is provided by the environment."
            )

        dimensions = {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
        }
        previous_actor = getattr(self, "actor", None)
        previous_critic = getattr(self, "critic", None)
        pending_state = self._pending_checkpoint_state or {}
        pending_artifacts = pending_state.get("module_artifacts", {})
        if not isinstance(pending_artifacts, Mapping):
            raise TypeError("Checkpoint 'module_artifacts' must be a mapping.")
        if module_artifacts is not None and not isinstance(
            module_artifacts, Mapping
        ):
            raise TypeError("'module_artifacts' must be a mapping.")
        saved_artifacts = dict(pending_artifacts)
        saved_artifacts.update(module_artifacts or {})
        
        actor_artifacts = {
            name.removeprefix("actor."): artifact
            for name, artifact in saved_artifacts.items()
            if name.startswith("actor.")
        }
        critic_artifacts = {
            name.removeprefix("critic."): artifact
            for name, artifact in saved_artifacts.items()
            if name.startswith("critic.")
        }

        if previous_actor is not None:
            actor_artifacts.update(previous_actor.export_module_artifacts())
        if previous_critic is not None:
            critic_artifacts.update(previous_critic.export_module_artifacts())

        actor_network = ConfigurableNetwork(
            modules=actor,
            variables=dimensions,
            module_artifacts=actor_artifacts,
            input_slices=observation_slices,
        )
        critic_network = ConfigurableNetwork(
            modules=critic,
            variables=dimensions,
            module_artifacts=critic_artifacts,
            input_slices=observation_slices,
        )
        if critic_network.is_recurrent:
            raise ValueError(
                "Recurrent critic modules are not supported yet."
            )
        if actor_network.output_features != action_dim:
            raise ValueError(
                "Actor output features must equal action_dim: "
                f"{actor_network.output_features} != {action_dim}."
            )
        if critic_network.output_features != 1:
            raise ValueError(
                "Critic output features must equal 1: "
                f"got {critic_network.output_features}."
            )
        if previous_actor is not None:
            actor_network.inherit_modules_from(previous_actor)
        if previous_critic is not None:
            critic_network.inherit_modules_from(previous_critic)

        distribution_config = dict(distribution)
        distribution_config["action_dim"] = action_dim
        action_distribution = build_action_distribution(
            distribution_config
        )
        self.actor = actor_network
        self.critic = critic_network
        self.action_distribution = action_distribution
        self.to(
            device=self.context.device,
            dtype=self.context.dtype,
        )


    def export_module_artifacts(self) -> dict[str, dict[str, Any]]:
        
        actor_artifacts = {
            f"actor.{name}": artifact
            for name, artifact in self.actor.export_module_artifacts().items()
        }
        critic_artifacts = {
            f"critic.{name}": artifact
            for name, artifact in self.critic.export_module_artifacts().items()
        }

        return actor_artifacts | critic_artifacts


    def checkpoint_state_dict(self) -> dict[str, Any]:
        state = super().checkpoint_state_dict()
        state["module_artifacts"] = self.export_module_artifacts()
        return state


    def save_module_artifacts(self, directory: Path) -> list[Path]:
        paths: list[Path] = []
        for network_name, network in (
            ("actor", self.actor),
            ("critic", self.critic),
        ):
            for module_name, artifact in (
                network.export_portable_module_artifacts().items()
            ):
                module_path = Path(*module_name.split("."))
                paths.append(atomic_save(
                    artifact,
                    directory / network_name
                    / module_path.parent / f"{module_path.name}.pt",
                ))
        return paths


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
