import torch
from collections.abc import Mapping
from typing import Any

from app.utils.context import RuntimeContext
from rl.algorithms.base import OnPolicyAlgorithm, PolicyOutput
from rl.policies.base import BasePolicy
from rl.policies.registry import POLICY_TYPE_MAP
from rl.utils.gae import compute_gae
from rl.utils.storage import RolloutBatch, RolloutStorage
from utils.param import update_attributes
from utils.component import Component, ComponentInfo
from utils.config import load_yaml


class PPO(OnPolicyAlgorithm):

    def __init__(
        self,
        context: RuntimeContext,
        *args, **kwargs,
    ) -> None:

        self.context = context
        self.policy: BasePolicy
        self.storage: RolloutStorage
        self.optimizer: torch.optim.Optimizer

        self.learning_rate: float
        self.gamma: float
        self.gae_lambda: float
        self.clip_range: float
        self.entropy_coef: float
        self.value_coef: float
        self.max_grad_norm: float
        self.num_epochs: int
        self.num_mini_batches: int


    def config_update(
        self,
        component: Component,
        learning_rate: float | None = None,
        gamma: float | None = None,
        gae_lambda: float | None = None,
        clip_range: float | None = None,
        entropy_coef: float | None = None,
        value_coef: float | None = None,
        max_grad_norm: float | None = None,
        num_epochs: int | None = None,
        num_mini_batches: int | None = None,
    ) -> None:

        update_attributes(
            self,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
            num_epochs=num_epochs,
            num_mini_batches=num_mini_batches,
        )
        self._validate_config()
        self._build(component=component)


    def _validate_config(self) -> None:
        
        if self.learning_rate <= 0.0:
            raise ValueError("'learning_rate' must be greater than 0.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("'gamma' must be in [0, 1].")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("'gae_lambda' must be in [0, 1].")
        if self.clip_range <= 0.0:
            raise ValueError("'clip_range' must be greater than 0.")
        if self.entropy_coef < 0.0:
            raise ValueError("'entropy_coef' must be nonnegative.")
        if self.value_coef < 0.0:
            raise ValueError("'value_coef' must be nonnegative.")
        if self.max_grad_norm <= 0.0:
            raise ValueError("'max_grad_norm' must be greater than 0.")
        if self.num_epochs <= 0:
            raise ValueError("'num_epochs' must be greater than 0.")
        if self.num_mini_batches <= 0:
            raise ValueError("'num_mini_batches' must be greater than 0.")


    def _build(
        self,
        component: Component
    ) -> None:

        policy = component.policy
        
        if policy is None:

            self._update_policy_config(component=component)
            self._update_optimizer_config()
            self.storage.clear()

        else:

            policy_type = self._check_policy_rebuild(policy=policy)
            if policy_type:
                self.policy = policy_type(
                    context=self.context,
                )
                self.optimizer = torch.optim.Adam(
                    self.policy.parameters(),
                )
                self.storage = RolloutStorage(
                    context=self.context
                )

            policy_config = load_yaml(policy.config)
            self._update_policy_config(
                component=component,
                **policy_config,
            )
            self._update_optimizer_config()
            self.storage.clear()


    def _update_policy_config(
        self,
        component: Component,
        *args, **kwargs,
    ) -> None:

        if (
            hasattr(self, "policy") is False
            or self.policy is None
        ):
            raise RuntimeError("policy instance is required.")
        self.policy.config_update(
            component=component,
            **kwargs
        )


    def _update_optimizer_config(self) -> None:

        if (
            hasattr(self, "optimizer") is False
            or self.optimizer is None
        ):
            raise RuntimeError("optimizer is not built.")
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate


    def _check_policy_rebuild(
        self,
        policy: ComponentInfo
    ) -> type[BasePolicy] | None:
        
        policy_type_name = policy.type
        if policy_type_name not in POLICY_TYPE_MAP:
            raise ValueError(
                f"Invalid policy type: {policy_type_name!r}."
            )

        policy_type = POLICY_TYPE_MAP[policy_type_name]
        if (
            hasattr(self, "policy") is False
            or not isinstance(self.policy, policy_type)
        ):
            return policy_type
        return None


    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> PolicyOutput:

        policy_output = self.policy.act(
            obs=obs,
            deterministic=deterministic,
        )

        return PolicyOutput(
            action=policy_output.action,
            log_prob=policy_output.log_prob,
            value=policy_output.value,
        )
    

    def process_transition(
        self,
        obs: torch.Tensor,
        policy_output: PolicyOutput,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        next_obs: torch.Tensor,
        info: Mapping[str, Any] | None = None,
    ) -> None:

        self.storage.add(
            obs=obs,
            action=policy_output.action,
            log_prob=policy_output.log_prob,
            value=policy_output.value,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            next_obs=next_obs,
        )


    def compute_returns(
        self,
        last_obs: torch.Tensor,
    ) -> None:

        rollout = self.storage.tensors()
        values = rollout.values

        with torch.no_grad():
            next_values = torch.empty_like(values)
            if values.shape[0] > 1:
                next_values[:-1] = values[1:]
            next_values[-1] = self.policy.predict_values(last_obs)

            if torch.any(rollout.truncated):
                truncated_obs = rollout.next_obs[rollout.truncated]
                next_values[rollout.truncated] = (
                    self.policy.predict_values(truncated_obs)
                )

        returns, advantages = compute_gae(
            rewards=rollout.rewards,
            values=rollout.values,
            next_values=next_values,
            terminated=rollout.terminated,
            truncated=rollout.truncated,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        self.storage.set_returns(
            returns=returns,
            advantages=advantages,
        )


    def update(self) -> dict[str, float]:

        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
        }
        num_updates = 0

        for _ in range(self.num_epochs):
            for batch in self.storage.mini_batches(
                num_mini_batches=self.num_mini_batches,
            ):
                loss, metrics = self._compute_loss(batch)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm,
                )   # grad will be modified in place
                self.optimizer.step()

                totals["loss"] += float(loss.detach().item())
                totals["grad_norm"] += float(grad_norm.detach().item())
                for name, value in metrics.items():
                    totals[name] += float(value.detach().item())
                num_updates += 1

        if num_updates == 0:
            raise RuntimeError("storage produced no mini-batches.")

        info = {
            f"ppo/{name}": value / num_updates
            for name, value in totals.items()
        }
        self.storage.clear()
        return info


    def _compute_loss(
        self,
        batch: RolloutBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        evaluation = self.policy.evaluate_actions(
            obs=batch.obs,
            actions=batch.actions,
        )

        log_ratio = (
            evaluation.log_prob - batch.old_log_probs
        )
        ratio = torch.exp(log_ratio)

        unclipped_loss = -batch.advantages * ratio
        clipped_loss = -batch.advantages * torch.clamp(
            ratio,
            1.0 - self.clip_range,
            1.0 + self.clip_range,
        )
        policy_loss = torch.maximum(
            unclipped_loss,
            clipped_loss,
        ).mean()

        value_loss = torch.nn.functional.mse_loss(
            evaluation.value,
            batch.returns,
        )
        entropy_mean = evaluation.entropy.mean()

        loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy_mean
        )

        with torch.no_grad():
            approx_kl = (ratio - 1.0 - log_ratio).mean()
            clip_fraction = (
                torch.abs(ratio - 1.0) > self.clip_range
            ).float().mean()

        return loss, {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy_mean,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }


    def close(self) -> None:

        if hasattr(self, "policy"):
            self.policy.close()
            del self.policy

        if hasattr(self, "optimizer"):
            del self.optimizer

        if hasattr(self, "storage"):
            self.storage.close()
            del self.storage
