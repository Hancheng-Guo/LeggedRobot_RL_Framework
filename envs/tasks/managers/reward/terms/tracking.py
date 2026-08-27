import torch
from collections.abc import Sequence

from envs.simulators.utils.context import ModelContext
from envs.tasks.managers.reward.terms.base import BaseRewardTerm
from envs.tasks.managers.reward.terms.registry import register_reward
from envs.tasks.utils.context import TaskContext


def _command_vector(
    task_context: TaskContext,
    names: Sequence[str] | None,
) -> torch.Tensor:
    
    if names is None:
        names = tuple(task_context.command)
    else:
        missing_names = tuple(
            name
            for name in names
            if name not in task_context.command
        )
        if missing_names:
            raise ValueError(f"Unknown command name(s): {missing_names}.")

    values = [task_context.command[name] for name in names]
    return torch.cat(values, dim=-1)


def _command_target_check(
    command: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:

    if command.shape != reference.shape:
        raise ValueError(
            "Command dimension does not match the tracked velocity dimension: "
            f"command shape is {tuple(command.shape)}, "
            f"reference shape is {tuple(reference.shape)}."
        )
    return command


class _BaseCommandTracking(BaseRewardTerm):

    def __init__(
        self,
        command_names: Sequence[str] | None = None,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.command_names = None if command_names is None else tuple(command_names)


    def command_vector(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        return _command_vector(task_context, self.command_names)


@register_reward
class TrackLinearVelocityXyL2Exp(_BaseCommandTracking):

    def __init__(
        self,
        x_std: float = 1.0,
        y_std: float = 1.0,
        command_names: Sequence[str] = ("lin_vel_x", "lin_vel_y"),
        *args, **kwargs,
    ) -> None:
        
        super().__init__(command_names=command_names, *args, **kwargs)

        self.std = torch.tensor(
            [x_std, y_std],
            dtype=self.context.dtype,
            device=self.context.device,
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        command = self.command_vector(task_context)
        velocity = task_context.state["base_lin_vel_body"][:, :2]
        target = _command_target_check(command, velocity)
        error = velocity - target
        normalized_error = error / self.std
        return torch.exp(-torch.sum(normalized_error.square(), dim=-1))


@register_reward
class TrackLinearVelocityXyL2ExpAndLogCosh(_BaseCommandTracking):

    def __init__(
        self,
        x_std: float = 1.0,
        y_std: float = 1.0,
        logcosh_weight: float = 0.5,
        command_names: Sequence[str] = ("lin_vel_x", "lin_vel_y"),
        *args, **kwargs,
    ) -> None:
        
        super().__init__(command_names=command_names, *args, **kwargs)

        self.std = torch.tensor(
            [x_std, y_std],
            dtype=self.context.dtype,
            device=self.context.device,
        )
        self.logcosh_weight = logcosh_weight


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        command = self.command_vector(task_context)
        velocity = task_context.state["base_lin_vel_body"][:, :2]
        target = _command_target_check(command, velocity)
        error = velocity - target
        normalized_error = error / self.std
        l2_exp = torch.exp(-torch.sum(normalized_error.square(), dim=-1))
        error_norm = torch.linalg.norm(normalized_error, dim=-1)
        logcosh = 1.0 - torch.log(torch.cosh(2.0 * error_norm))

        return (1.0 - self.logcosh_weight) * l2_exp + self.logcosh_weight * logcosh


@register_reward
class TrackLinearVelocityXyErrorIntegralL2(_BaseCommandTracking):

    def __init__(
        self,
        num_envs: int,
        integral_length: int = 100,
        command_names: Sequence[str] = ("lin_vel_x", "lin_vel_y"),
        *args, **kwargs,
    ) -> None:
        
        super().__init__(command_names=command_names, *args, **kwargs)

        self.integral_length = integral_length
        self.error_history = torch.zeros(
            (num_envs, integral_length),
            dtype=self.context.dtype,
            device=self.context.device,
        )
        self.error_history_length = torch.zeros(
            num_envs,
            dtype=self.context.dtype,
            device=self.context.device,
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        command = self.command_vector(task_context)
        velocity = task_context.state["base_lin_vel_body"][:, :2]
        target = _command_target_check(command, velocity)
        error = torch.linalg.norm(target - velocity, dim=-1)
        self.error_history = torch.roll(self.error_history, shifts=-1, dims=-1)
        self.error_history[:, -1] = error
        self.error_history_length = torch.clamp(
            self.error_history_length + 1.0,
            max=float(self.integral_length),
        )
        return (
            torch.sum(self.error_history, dim=-1) / self.error_history_length
        ).square()


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        
        if env_ids is None:
            self.error_history.zero_()
            self.error_history_length.zero_()
        else:
            self.error_history[env_ids] = 0.0
            self.error_history_length[env_ids] = 0.0


@register_reward
class TrackAngularVelocityZL2Exp(_BaseCommandTracking):

    def __init__(
        self,
        std: float = 1.0,
        command_names: Sequence[str] = ("ang_vel_z",),
        *args, **kwargs,
    ) -> None:
        
        super().__init__(command_names=command_names, *args, **kwargs)

        self.std = std


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        command = self.command_vector(task_context)
        velocity = task_context.state["base_ang_vel_body"][:, 2:3]
        target = _command_target_check(command, velocity)
        error = velocity - target
        normalized_error = error / self.std
        return torch.exp(-normalized_error.square()).squeeze(-1)


@register_reward
class TrackAngularVelocityZL2ExpAndLogCosh(TrackAngularVelocityZL2Exp):

    def __init__(
        self,
        logcosh_weight: float = 0.5,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.logcosh_weight = logcosh_weight


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        command = self.command_vector(task_context)
        velocity = task_context.state["base_ang_vel_body"][:, 2:3]
        target = _command_target_check(command, velocity)
        error = velocity - target
        normalized_error = error / self.std
        l2_exp = torch.exp(-normalized_error.square()).squeeze(-1)
        logcosh = (
            1.0 - torch.log(torch.cosh(2.0 * normalized_error))
        ).squeeze(-1)
        return (1.0 - self.logcosh_weight) * l2_exp + self.logcosh_weight * logcosh


@register_reward
class TrackAngularVelocityZErrorIntegralL2(_BaseCommandTracking):

    def __init__(
        self,
        num_envs: int,
        integral_length: int = 100,
        command_names: Sequence[str] = ("ang_vel_z",),
        *args, **kwargs,
    ) -> None:
        
        super().__init__(command_names=command_names, *args, **kwargs)

        self.integral_length = integral_length
        self.error_history = torch.zeros(
            (num_envs, integral_length),
            dtype=self.context.dtype,
            device=self.context.device,
        )
        self.error_history_length = torch.zeros(
            num_envs,
            dtype=self.context.dtype,
            device=self.context.device,
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        command = self.command_vector(task_context)
        velocity = task_context.state["base_ang_vel_body"][:, 2:3]
        target = _command_target_check(command, velocity)
        error = (target - velocity).abs()
        self.error_history = torch.roll(self.error_history, shifts=-1, dims=-1)
        self.error_history[:, -1] = error.squeeze(-1)
        self.error_history_length = torch.clamp(
            self.error_history_length + 1.0,
            max=float(self.integral_length),
        )
        return (
            torch.sum(self.error_history, dim=-1) / self.error_history_length
        ).square()


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        
        if env_ids is None:
            self.error_history.zero_()
            self.error_history_length.zero_()
        else:
            self.error_history[env_ids] = 0.0
            self.error_history_length[env_ids] = 0.0
