import torch
from dataclasses import dataclass
from typing import Iterator

from app.utils.context import RuntimeContext


@dataclass(slots=True)
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


@dataclass(slots=True)
class RolloutTensors:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    next_obs: torch.Tensor


class RolloutStorage:

    def __init__(
        self,
        context: RuntimeContext
    ) -> None:
        
        self.context = context
        
        self.obs: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.terminated: list[torch.Tensor] = []
        self.truncated: list[torch.Tensor] = []
        self.next_obs: list[torch.Tensor] = []
        self.returns: torch.Tensor | None = None
        self.advantages: torch.Tensor | None = None


    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        
        self.obs.append(self._prepare(obs))
        self.actions.append(self._prepare(action))
        self.log_probs.append(self._prepare(log_prob))
        self.values.append(self._prepare(value))
        self.rewards.append(self._prepare(reward))
        self.terminated.append(self._prepare(terminated))
        self.truncated.append(self._prepare(truncated))
        self.next_obs.append(self._prepare(next_obs))


    def tensors(self) -> RolloutTensors:

        return RolloutTensors(
            obs=torch.stack(self.obs),
            actions=torch.stack(self.actions),
            log_probs=torch.stack(self.log_probs),
            values=torch.stack(self.values),
            rewards=torch.stack(self.rewards),
            terminated=torch.stack(self.terminated),
            truncated=torch.stack(self.truncated),
            next_obs=torch.stack(self.next_obs),
        )


    def set_returns(
        self,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> None:
        
        self.returns = self._prepare(returns)
        self.advantages = self._prepare(advantages)


    def mini_batches(
        self,
        num_mini_batches: int,
    ) -> Iterator[RolloutBatch]:
        
        if self.returns is None or self.advantages is None:
            raise RuntimeError("returns have not been computed.")
        if num_mini_batches <= 0:
            raise ValueError("'num_mini_batches' must be greater than 0.")

        rollout = self.tensors()
        num_steps, num_envs = rollout.rewards.shape[:2]
        batch_size = num_steps * num_envs
        if num_mini_batches > batch_size:
            raise ValueError(
                "'num_mini_batches' cannot exceed rollout batch size."
            )

        indices = torch.randperm(
            batch_size,
            device=self.context.device,
        )

        for batch_indices in torch.tensor_split(
            indices,
            num_mini_batches,
        ):
            yield RolloutBatch(
                obs=self._flatten(rollout.obs)[batch_indices],
                actions=self._flatten(rollout.actions)[batch_indices],
                old_log_probs=self._flatten_scalar(
                    rollout.log_probs
                )[batch_indices],
                old_values=self._flatten_scalar(
                    rollout.values
                )[batch_indices],
                returns=self._flatten_scalar(
                    self.returns
                )[batch_indices],
                advantages=self._flatten_scalar(
                    self.advantages
                )[batch_indices],
            )


    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.terminated.clear()
        self.truncated.clear()
        self.next_obs.clear()
        self.returns = None
        self.advantages = None


    def close(self) -> None:
        self.clear()


    def _prepare(self, value: torch.Tensor) -> torch.Tensor:
        return value.detach().to(device=self.context.device)


    @staticmethod
    def _flatten(value: torch.Tensor) -> torch.Tensor:
        return value.flatten(start_dim=0, end_dim=1)


    @classmethod
    def _flatten_scalar(cls, value: torch.Tensor) -> torch.Tensor:
        return cls._flatten(cls._squeeze_scalar(value))


    @staticmethod
    def _squeeze_scalar(value: torch.Tensor) -> torch.Tensor:
        if value.ndim > 2 and value.shape[-1] == 1:
            return value.squeeze(-1)
        return value
