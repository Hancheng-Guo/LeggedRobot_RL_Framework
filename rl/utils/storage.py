import torch
from dataclasses import dataclass
from typing import Iterator

from app.utils.context import RuntimeContext
from rl.policies.base import RecurrentState


@dataclass(slots=True)
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    initial_recurrent_state: RecurrentState | None = None
    reset_mask: torch.Tensor | None = None


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
    recurrent_states: RecurrentState | None


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
        self.recurrent_states: list[RecurrentState] = []
        self._is_recurrent: bool | None = None
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
        recurrent_state: RecurrentState | None = None,
    ) -> None:
        
        if recurrent_state is not None and not recurrent_state:
            raise ValueError("Recurrent state cannot be empty.")
        
        is_recurrent = recurrent_state is not None
        if self._is_recurrent is None:
            self._is_recurrent = is_recurrent
        elif self._is_recurrent != is_recurrent:
            raise ValueError(
                "A rollout cannot mix recurrent and non-recurrent states."
            )
        
        self.obs.append(self._prepare(obs))
        self.actions.append(self._prepare(action))
        self.log_probs.append(self._prepare(log_prob))
        self.values.append(self._prepare(value))
        self.rewards.append(self._prepare(reward))
        self.terminated.append(self._prepare(terminated))
        self.truncated.append(self._prepare(truncated))
        self.next_obs.append(self._prepare(next_obs))
        if recurrent_state is not None:
            self.recurrent_states.append({
                name: self._prepare(value)
                for name, value in recurrent_state.items()
            })


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
            recurrent_states=self._stack_recurrent_states(),
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
        if rollout.recurrent_states is not None:
            yield from self._recurrent_mini_batches(
                rollout=rollout,
                num_mini_batches=num_mini_batches,
            )
            return

        yield from self._feedforward_mini_batches(
            rollout=rollout,
            num_mini_batches=num_mini_batches,
        )


    def _feedforward_mini_batches(
        self,
        rollout: RolloutTensors,
        num_mini_batches: int,
    ) -> Iterator[RolloutBatch]:
        
        assert self.returns is not None
        assert self.advantages is not None
        
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


    def _recurrent_mini_batches(
        self,
        rollout: RolloutTensors,
        num_mini_batches: int,
    ) -> Iterator[RolloutBatch]:
        assert rollout.recurrent_states is not None
        assert self.returns is not None
        assert self.advantages is not None

        _, num_envs = rollout.rewards.shape[:2]
        if num_mini_batches > num_envs:
            raise ValueError(
                "For recurrent rollouts, 'num_mini_batches' cannot exceed "
                "the number of environments."
            )

        env_indices = torch.randperm(
            num_envs,
            device=self.context.device,
        )
        done = rollout.terminated | rollout.truncated
        reset_mask = torch.zeros_like(done, dtype=torch.bool)
        reset_mask[1:] = done[:-1]

        for batch_env_ids in torch.tensor_split(
            env_indices,
            num_mini_batches,
        ):
            yield RolloutBatch(
                obs=rollout.obs[:, batch_env_ids],
                actions=rollout.actions[:, batch_env_ids],
                old_log_probs=self._squeeze_scalar(
                    rollout.log_probs
                )[:, batch_env_ids],
                old_values=self._squeeze_scalar(
                    rollout.values
                )[:, batch_env_ids],
                returns=self._squeeze_scalar(
                    self.returns
                )[:, batch_env_ids],
                advantages=self._squeeze_scalar(
                    self.advantages
                )[:, batch_env_ids],
                initial_recurrent_state={
                    name: value[0, batch_env_ids]
                    for name, value in rollout.recurrent_states.items()
                },
                reset_mask=reset_mask[:, batch_env_ids],
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
        self.recurrent_states.clear()
        self._is_recurrent = None
        self.returns = None
        self.advantages = None


    def close(self) -> None:
        self.clear()


    def _prepare(self, value: torch.Tensor) -> torch.Tensor:
        return value.detach().to(device=self.context.device)


    def _stack_recurrent_states(self) -> RecurrentState | None:
        if not self.recurrent_states:
            return None
        expected_names = set(self.recurrent_states[0])
        if any(set(state) != expected_names for state in self.recurrent_states):
            raise ValueError("Recurrent state keys changed during rollout.")
        return {
            name: torch.stack([
                state[name]
                for state in self.recurrent_states
            ])
            for name in expected_names
        }


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
