import torch

from envs.tasks.managers.observation.terms.base import BaseObservationTerm
from envs.tasks.managers.observation.terms.registry import register_observation
from envs.tasks.utils.context import TaskContext


@register_observation
class LastAction(BaseObservationTerm):

    def __init__(
        self,
        action_dim: int,
        num_envs: int,
        lags: int = 1,
        *args, **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        if (
            not isinstance(lags, int)
            or isinstance(lags, bool)
            or lags < 1
        ):
            raise ValueError("'lags' must be a positive integer.")

        self.lags = lags
        self.output_dim = action_dim * lags
        self.history = torch.zeros(
            (num_envs, lags, action_dim),
            dtype=self.context.dtype,
            device=self.context.device,
        )
        self.recorded_step = torch.full(
            (num_envs,),
            -1,
            dtype=torch.long,
            device=self.context.device,
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        env_ids = task_context.env_ids
        if env_ids is None:
            env_ids = torch.arange(
                self.history.shape[0],
                device=self.context.device,
            )

        update = task_context.episode_step > self.recorded_step[env_ids]
        update_ids = env_ids[update]
        if update_ids.numel() > 0:
            previous = self.history[update_ids].clone()
            if self.lags > 1:
                self.history[update_ids, 1:] = previous[:, :-1]
            self.history[update_ids, 0] = task_context.action[update].detach()
            self.recorded_step[update_ids] = task_context.episode_step[update]

        return self.history[env_ids].reshape(-1, self.output_dim)


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        
        if env_ids is None:
            self.history.zero_()
            self.recorded_step.fill_(-1)
        else:
            self.history[env_ids] = 0
            self.recorded_step[env_ids] = -1
