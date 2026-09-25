import torch

from envs.simulators.utils.context import ModelContext
from envs.tasks.managers.reward.terms.base import BaseRewardTerm
from envs.tasks.managers.reward.terms.registry import register_reward
from envs.tasks.utils.context import TaskContext


@register_reward
class ActionDiffL2(BaseRewardTerm):

    def __init__(
        self,
        num_envs: int,
        model_context: ModelContext,
        max_lag: int = 1,
        decay: float = 0.5,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        if not isinstance(max_lag, int) or max_lag < 1:
            raise ValueError("'max_lag' must be a positive integer.")
        if not 0.0 <= decay <= 1.0:
            raise ValueError("'decay' must be in the range [0, 1].")

        self.max_lag = max_lag
        self.decay = decay
        self._action_history = torch.zeros(
            (num_envs, max_lag, model_context.nu),
            dtype=self.context.dtype,
            device=self.context.device,
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:

        action = task_context.action

        lag_weights = torch.tensor(
            [self.decay**lag for lag in range(self.max_lag)],
            dtype=self.context.dtype,
            device=self.context.device,
        )
        action_diffs = action.unsqueeze(1) - self._action_history
        action_diffs_l2 = action_diffs.square().mean(dim=-1)

        if self.max_lag > 1:
            self._action_history[:, 1:] = self._action_history[:, :-1].clone()
        self._action_history[:, 0] = action

        return (action_diffs_l2 * lag_weights).sum(dim=-1)


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        
        if env_ids is None:
            self._action_history.zero_()
        else:
            self._action_history[env_ids] = 0
