import math

import torch
from collections.abc import Sequence

from envs.tasks.managers.observation.terms.base import BaseObservationTerm
from envs.tasks.managers.observation.terms.registry import register_observation
from envs.tasks.utils.context import TaskContext


class _BaseVelocityErrorIntegral(BaseObservationTerm):

    def __init__(
        self,
        num_envs: int,
        output_dim: int,
        command_names: Sequence[str],
        integral_length: int = 100,
        *args, **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        if (
            not isinstance(integral_length, int)
            or isinstance(integral_length, bool)
            or integral_length < 1
        ):
            raise ValueError("'integral_length' must be a positive integer.")

        self.output_dim = output_dim
        self.command_names = tuple(command_names)
        self.error_history = torch.zeros(
            (num_envs, integral_length, output_dim),
            dtype=self.context.dtype,
            device=self.context.device,
        )
        self.recorded_step = torch.full(
            (num_envs,),
            -1,
            dtype=torch.long,
            device=self.context.device,
        )


    def _compute_integral(
        self,
        task_context: TaskContext,
        velocity: torch.Tensor,
    ) -> torch.Tensor:

        env_ids = task_context.env_ids
        if env_ids is None:
            env_ids = torch.arange(
                self.error_history.shape[0],
                device=self.context.device,
            )

        update = (
            (task_context.episode_step > 0)
            & (task_context.episode_step > self.recorded_step[env_ids])
        )
        update_ids = env_ids[update]
        if update_ids.numel() > 0:
            command = torch.cat(
                [task_context.last_command[name] for name in self.command_names],
                dim=-1,
            )
            if command.shape != velocity.shape:
                raise ValueError(
                    f"Tracked command has shape {tuple(command.shape)}, "
                    f"expected {tuple(velocity.shape)}."
                )
            self.error_history[update_ids] = torch.roll(
                self.error_history[update_ids],
                shifts=-1,
                dims=1,
            )
            self.error_history[update_ids, -1] = (
                (command[update] - velocity[update]) * task_context.step_dt
            ).detach()
            self.recorded_step[update_ids] = task_context.episode_step[update]

        return self.error_history[env_ids].sum(dim=1)


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        
        if env_ids is None:
            self.error_history.zero_()
            self.recorded_step.fill_(-1)
        else:
            self.error_history[env_ids] = 0
            self.recorded_step[env_ids] = -1


@register_observation
class TrackLinearVelocityXErrorIntegral(_BaseVelocityErrorIntegral):

    def __init__(
        self,
        command_names: Sequence[str] = ("lin_vel_x",),
        *args, **kwargs,
    ) -> None:
        
        super().__init__(
            output_dim=1,
            command_names=command_names,
            *args, **kwargs,
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        return self._compute_integral(
            task_context,
            task_context.state.base_lin_vel_body[:, :1],
        )


@register_observation
class TrackLinearVelocityYErrorIntegral(_BaseVelocityErrorIntegral):

    def __init__(
        self,
        command_names: Sequence[str] = ("lin_vel_y",),
        *args, **kwargs,
    ) -> None:
        
        super().__init__(
            output_dim=1,
            command_names=command_names,
            *args, **kwargs,
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        return self._compute_integral(
            task_context, task_context.state.base_lin_vel_body[:, 1:2],
        )


@register_observation
class TrackAngularVelocityZErrorIntegral(_BaseVelocityErrorIntegral):

    def __init__(
        self,
        command_names: Sequence[str] = ("ang_vel_z",),
        *args, **kwargs,
    ) -> None:
        
        super().__init__(
            output_dim=1,
            command_names=command_names,
            *args, **kwargs,
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        return self._compute_integral(
            task_context, task_context.state.base_ang_vel_body[:, 2:3],
        )


class _BaseTanhVelocityErrorIntegral(_BaseVelocityErrorIntegral):

    def __init__(
        self,
        alpha: float = 1.0,
        *args, **kwargs
    ) -> None:
        
        super().__init__(*args, **kwargs)

        if (
            not isinstance(alpha, (int, float))
            or isinstance(alpha, bool)
            or not math.isfinite(alpha)
            or alpha <= 0
        ):
            raise ValueError("'alpha' must be a finite positive number.")
        
        self.alpha = float(alpha)


    def _compute_tanh_integral(
        self,
        task_context: TaskContext,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        
        return torch.tanh(
            self.alpha * self._compute_integral(task_context, velocity)
        )


@register_observation
class TrackLinearVelocityXErrorIntegralTanh(_BaseTanhVelocityErrorIntegral):

    def __init__(
        self,
        command_names: Sequence[str] = ("lin_vel_x",),
        *args, **kwargs,
    ) -> None:
        
        super().__init__(
            output_dim=1,
            command_names=command_names,
            *args, **kwargs,
        )


    def compute(self, task_context: TaskContext) -> torch.Tensor:
        return self._compute_tanh_integral(
            task_context,
            task_context.state.base_lin_vel_body[:, :1],
        )


@register_observation
class TrackLinearVelocityYErrorIntegralTanh(_BaseTanhVelocityErrorIntegral):

    def __init__(
        self,
        command_names: Sequence[str] = ("lin_vel_y",),
        *args, **kwargs,
    ) -> None:
        
        super().__init__(
            output_dim=1,
            command_names=command_names,
            *args, **kwargs,
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        return self._compute_tanh_integral(
            task_context,
            task_context.state.base_lin_vel_body[:, 1:2],
        )


@register_observation
class TrackAngularVelocityZErrorIntegralTanh(_BaseTanhVelocityErrorIntegral):

    def __init__(
        self,
        command_names: Sequence[str] = ("ang_vel_z",),
        *args, **kwargs
    ) -> None:
        
        super().__init__(
            output_dim=1,
            command_names=command_names,
            *args, **kwargs,
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        return self._compute_tanh_integral(
            task_context,
            task_context.state.base_ang_vel_body[:, 2:3],
        )
