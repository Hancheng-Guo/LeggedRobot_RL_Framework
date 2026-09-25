import math
import torch
from collections.abc import Sequence

from envs.tasks.managers.reward.terms.base import BaseRewardTerm
from envs.tasks.managers.reward.terms.registry import register_reward
from envs.tasks.managers.reward.terms.utils import command_vector
from envs.tasks.utils.context import TaskContext


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
        
        return command_vector(task_context, self.command_names)


class _BaseVelocityErrorIntegralL2(_BaseCommandTracking):

    def __init__(
        self,
        num_envs: int,
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

        self.integral_length = integral_length
        self.error_history = torch.zeros(
            (num_envs, integral_length),
            dtype=self.context.dtype,
            device=self.context.device,
        )


    def _compute_integral(
        self,
        task_context: TaskContext,
        velocity: torch.Tensor,
    ) -> torch.Tensor:

        command = self.command_vector(task_context)
        target = _command_target_check(command, velocity)
        error = target - velocity
        self.error_history = torch.roll(self.error_history, shifts=-1, dims=-1)
        self.error_history[:, -1] = error.squeeze(-1) * task_context.step_dt
        return self.error_history.sum(dim=-1).square()


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:

        if env_ids is None:
            self.error_history.zero_()
        else:
            self.error_history[env_ids] = 0.0


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

        if not math.isfinite(x_std) or x_std <= 0.0:
            raise ValueError("'x_std' must be positive and finite.")
        if not math.isfinite(y_std) or y_std <= 0.0:
            raise ValueError("'y_std' must be positive and finite.")

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
        velocity = task_context.state.base_lin_vel_body[:, :2]
        target = _command_target_check(command, velocity)
        error = velocity - target
        normalized_error = error / self.std
        return torch.exp(-torch.sum(normalized_error.square(), dim=-1))


@register_reward
class TrackLinearVelocityXL2Exp(_BaseCommandTracking):

    def __init__(
        self,
        std: float = 1.0,
        command_names: Sequence[str] = ("lin_vel_x",),
        *args, **kwargs,
    ) -> None:

        super().__init__(command_names=command_names, *args, **kwargs)

        if not math.isfinite(std) or std <= 0.0:
            raise ValueError("'std' must be positive and finite.")

        self.std = torch.tensor(
            [std],
            dtype=self.context.dtype,
            device=self.context.device,
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:

        command = self.command_vector(task_context)
        velocity = task_context.state.base_lin_vel_body[:, :1]
        target = _command_target_check(command, velocity)
        error = velocity - target
        normalized_error = error / self.std
        return torch.exp(-torch.sum(normalized_error.square(), dim=-1))


@register_reward
class TrackLinearVelocityYL2Exp(_BaseCommandTracking):

    def __init__(
        self,
        std: float = 1.0,
        command_names: Sequence[str] = ("lin_vel_y",),
        *args, **kwargs,
    ) -> None:

        super().__init__(command_names=command_names, *args, **kwargs)

        if not math.isfinite(std) or std <= 0.0:
            raise ValueError("'std' must be positive and finite.")

        self.std = torch.tensor(
            [std],
            dtype=self.context.dtype,
            device=self.context.device,
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:

        command = self.command_vector(task_context)
        velocity = task_context.state.base_lin_vel_body[:, 1:2]
        target = _command_target_check(command, velocity)
        error = velocity - target
        normalized_error = error / self.std
        return torch.exp(-torch.sum(normalized_error.square(), dim=-1))


@register_reward
class TrackLinearVelocityXyL2ExpAndLogcosh(_BaseCommandTracking):

    def __init__(
        self,
        x_std: float = 1.0,
        y_std: float = 1.0,
        logcosh_weight: float = 0.5,
        command_names: Sequence[str] = ("lin_vel_x", "lin_vel_y"),
        *args, **kwargs,
    ) -> None:

        super().__init__(command_names=command_names, *args, **kwargs)

        if not math.isfinite(x_std) or x_std <= 0.0:
            raise ValueError("'x_std' must be positive and finite.")
        if not math.isfinite(y_std) or y_std <= 0.0:
            raise ValueError("'y_std' must be positive and finite.")
        if (
            not math.isfinite(logcosh_weight)
            or not 0.0 <= logcosh_weight <= 1.0
        ):
            raise ValueError(
                "'logcosh_weight' must be finite and within [0, 1]."
            )

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
        velocity = task_context.state.base_lin_vel_body[:, :2]
        target = _command_target_check(command, velocity)
        error = velocity - target
        normalized_error = error / self.std
        l2_exp = torch.exp(-torch.sum(normalized_error.square(), dim=-1))
        error_norm = torch.linalg.norm(normalized_error, dim=-1)
        logcosh = 1.0 - torch.log(torch.cosh(2.0 * error_norm))

        return (1.0 - self.logcosh_weight) * l2_exp + self.logcosh_weight * logcosh


@register_reward
class TrackLinearVelocityXL2ExpAndLogcosh(_BaseCommandTracking):

    def __init__(
        self,
        std: float = 1.0,
        logcosh_weight: float = 0.5,
        command_names: Sequence[str] = ("lin_vel_x",),
        *args, **kwargs,
    ) -> None:

        super().__init__(command_names=command_names, *args, **kwargs)

        if not math.isfinite(std) or std <= 0.0:
            raise ValueError("'std' must be positive and finite.")
        if (
            not math.isfinite(logcosh_weight)
            or not 0.0 <= logcosh_weight <= 1.0
        ):
            raise ValueError(
                "'logcosh_weight' must be finite and within [0, 1]."
            )

        self.std = torch.tensor(
            [std],
            dtype=self.context.dtype,
            device=self.context.device,
        )
        self.logcosh_weight = logcosh_weight


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:

        command = self.command_vector(task_context)
        velocity = task_context.state.base_lin_vel_body[:, :1]
        target = _command_target_check(command, velocity)
        error = velocity - target
        normalized_error = error / self.std
        l2_exp = torch.exp(-torch.sum(normalized_error.square(), dim=-1))
        error_norm = torch.linalg.norm(normalized_error, dim=-1)
        logcosh = 1.0 - torch.log(torch.cosh(2.0 * error_norm))

        return (1.0 - self.logcosh_weight) * l2_exp + self.logcosh_weight * logcosh


@register_reward
class TrackLinearVelocityYL2ExpAndLogcosh(_BaseCommandTracking):

    def __init__(
        self,
        std: float = 1.0,
        logcosh_weight: float = 0.5,
        command_names: Sequence[str] = ("lin_vel_y",),
        *args, **kwargs,
    ) -> None:

        super().__init__(command_names=command_names, *args, **kwargs)

        if not math.isfinite(std) or std <= 0.0:
            raise ValueError("'std' must be positive and finite.")
        if (
            not math.isfinite(logcosh_weight)
            or not 0.0 <= logcosh_weight <= 1.0
        ):
            raise ValueError(
                "'logcosh_weight' must be finite and within [0, 1]."
            )

        self.std = torch.tensor(
            [std],
            dtype=self.context.dtype,
            device=self.context.device,
        )
        self.logcosh_weight = logcosh_weight


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:

        command = self.command_vector(task_context)
        velocity = task_context.state.base_lin_vel_body[:, 1:2]
        target = _command_target_check(command, velocity)
        error = velocity - target
        normalized_error = error / self.std
        l2_exp = torch.exp(-torch.sum(normalized_error.square(), dim=-1))
        error_norm = torch.linalg.norm(normalized_error, dim=-1)
        logcosh = 1.0 - torch.log(torch.cosh(2.0 * error_norm))

        return (1.0 - self.logcosh_weight) * l2_exp + self.logcosh_weight * logcosh


@register_reward
class TrackAngularVelocityZL2Exp(_BaseCommandTracking):

    def __init__(
        self,
        std: float = 1.0,
        command_names: Sequence[str] = ("ang_vel_z",),
        *args, **kwargs,
    ) -> None:

        super().__init__(command_names=command_names, *args, **kwargs)

        if not math.isfinite(std) or std <= 0.0:
            raise ValueError("'std' must be positive and finite.")

        self.std = torch.tensor(
            [std],
            dtype=self.context.dtype,
            device=self.context.device,
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:

        command = self.command_vector(task_context)
        velocity = task_context.state.base_ang_vel_body[:, 2:3]
        target = _command_target_check(command, velocity)
        error = velocity - target
        normalized_error = error / self.std
        return torch.exp(-normalized_error.square()).squeeze(-1)


@register_reward
class TrackAngularVelocityZL2ExpAndLogcosh(_BaseCommandTracking):

    def __init__(
        self,
        std: float = 1.0,
        logcosh_weight: float = 0.5,
        command_names: Sequence[str] = ("ang_vel_z",),
        *args, **kwargs,
    ) -> None:

        super().__init__(command_names=command_names, *args, **kwargs)

        if not math.isfinite(std) or std <= 0.0:
            raise ValueError("'std' must be positive and finite.")
        if (
            not math.isfinite(logcosh_weight)
            or not 0.0 <= logcosh_weight <= 1.0
        ):
            raise ValueError(
                "'logcosh_weight' must be finite and within [0, 1]."
            )

        self.std = torch.tensor(
            [std],
            dtype=self.context.dtype,
            device=self.context.device,
        )
        self.logcosh_weight = logcosh_weight


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:

        command = self.command_vector(task_context)
        velocity = task_context.state.base_ang_vel_body[:, 2:3]
        target = _command_target_check(command, velocity)
        error = velocity - target
        normalized_error = error / self.std
        l2_exp = torch.exp(-normalized_error.square()).squeeze(-1)
        logcosh = (
            1.0 - torch.log(torch.cosh(2.0 * normalized_error))
        ).squeeze(-1)
        return (1.0 - self.logcosh_weight) * l2_exp + self.logcosh_weight * logcosh


@register_reward
class TrackLinearVelocityXErrorIntegralL2(_BaseVelocityErrorIntegralL2):

    def __init__(
        self,
        command_names: Sequence[str] = ("lin_vel_x",),
        *args, **kwargs,
    ) -> None:
        
        super().__init__(
            command_names=command_names,
            *args, **kwargs
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        return self._compute_integral(
            task_context,
            task_context.state.base_lin_vel_body[:, 0:1]
        )


@register_reward
class TrackLinearVelocityYErrorIntegralL2(_BaseVelocityErrorIntegralL2):

    def __init__(
        self,
        command_names: Sequence[str] = ("lin_vel_y",),
        *args, **kwargs,
    ) -> None:
        
        super().__init__(
            command_names=command_names,
            *args, **kwargs
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        return self._compute_integral(
            task_context, task_context.state.base_lin_vel_body[:, 1:2]
        )


@register_reward
class TrackAngularVelocityZErrorIntegralL2(_BaseVelocityErrorIntegralL2):

    def __init__(
        self,
        command_names: Sequence[str] = ("ang_vel_z",),
        *args, **kwargs,
    ) -> None:
        
        super().__init__(
            command_names=command_names,
            *args, **kwargs
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        return self._compute_integral(
            task_context, task_context.state.base_ang_vel_body[:, 2:3]
        )
