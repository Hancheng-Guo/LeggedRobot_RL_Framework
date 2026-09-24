import math
import torch
from collections.abc import Sequence

from envs.simulators.utils.context import ModelContext
from envs.tasks.managers.reward.terms.base import BaseRewardTerm
from envs.tasks.managers.reward.terms.registry import register_reward
from envs.tasks.managers.reward.terms.utils import command_vector
from envs.tasks.utils.context import TaskContext


@register_reward
class FootStateDurationCommandWeighedExp(BaseRewardTerm):

    def __init__(
        self,
        num_envs: int,
        model_context: ModelContext,
        command_names: Sequence[str] = ("lin_vel_x", "lin_vel_y", "ang_vel_z"),
        sigma: float = 1.0,
        *args, **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.foot_geom_ids = model_context.foot_geom_ids
        self.sigma = sigma
        self.command_names = command_names

        self.last_foot_state = torch.zeros(
            (num_envs, self.foot_geom_ids.numel()),
            dtype=torch.bool,
            device=self.context.device,
        )
        self.duration = torch.zeros(
            (num_envs,),
            dtype=self.context.dtype,
            device=self.context.device,
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        landed = task_context.state.foot_ground_contact
        unchanged = (landed == self.last_foot_state).all(dim=-1)
        self.duration = torch.where(
            unchanged,
            self.duration + task_context.step_dt,
            torch.zeros_like(self.duration),
        )
        self.last_foot_state.copy_(landed)

        command_norm = torch.linalg.norm(
            command_vector(task_context, self.command_names),
            dim=-1,
        )
        scaled_duration = self.duration / self.sigma
        return torch.exp(-command_norm * scaled_duration)


    def reset(
        self,
        env_ids:
        torch.Tensor | None = None
    ) -> None:
        
        if env_ids is None:
            self.duration.zero_()
            self.last_foot_state.zero_()
        else:
            self.duration[env_ids] = 0.0
            self.last_foot_state[env_ids] = False


@register_reward
class FootStateDurationCubicCommandWeighedExp(BaseRewardTerm):

    def __init__(
        self,
        num_envs: int,
        model_context: ModelContext,
        command_names: Sequence[str] = ("lin_vel_x", "lin_vel_y", "ang_vel_z"),
        sigma: float = 1.0,
        *args, **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.foot_geom_ids = model_context.foot_geom_ids
        self.sigma = sigma
        self.command_names = command_names

        self.last_foot_state = torch.zeros(
            (num_envs, self.foot_geom_ids.numel()),
            dtype=torch.bool,
            device=self.context.device,
        )
        self.duration = torch.zeros(
            (num_envs,),
            dtype=self.context.dtype,
            device=self.context.device,
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        landed = task_context.state.foot_ground_contact
        unchanged = (landed == self.last_foot_state).all(dim=-1)
        self.duration = torch.where(
            unchanged,
            self.duration + task_context.step_dt,
            torch.zeros_like(self.duration),
        )
        self.last_foot_state.copy_(landed)

        command_norm = torch.linalg.norm(
            command_vector(task_context, self.command_names),
            dim=-1,
        )
        scaled_duration = self.duration / self.sigma
        return torch.exp(
            -command_norm * scaled_duration ** 3
        )
    

    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:

        if env_ids is None:
            self.duration.zero_()
            self.last_foot_state.zero_()
        else:
            self.duration[env_ids] = 0.0
            self.last_foot_state[env_ids] = False


@register_reward
class FootStateSwitch(BaseRewardTerm):

    def __init__(
        self,
        num_envs: int,
        model_context: ModelContext,
        hold_time: float = 0.2,
        *args, **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        if hold_time <= 0:
            raise ValueError("'hold_time' must be a positive float.")

        self.foot_geom_ids = model_context.foot_geom_ids
        self.hold_time = hold_time

        self.last_foot_state = torch.zeros(
            (num_envs, self.foot_geom_ids.numel()),
            dtype=torch.bool,
            device=self.context.device,
        )
        self.duration = torch.zeros(
            (num_envs, ),
            dtype=self.context.dtype,
            device=self.context.device,
        )
        self.initialized = torch.zeros(
            (num_envs, ),
            dtype=torch.bool,
            device=self.context.device,
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        landed = task_context.state.foot_ground_contact
        changed = (landed != self.last_foot_state).any(dim=-1)
        within_time_limit = self.duration < self.hold_time
        illegal_switch = changed * within_time_limit * self.initialized

        self.duration = torch.where(
            changed,
            torch.zeros_like(self.duration),
            self.duration + task_context.step_dt,
        )
        self.last_foot_state.copy_(landed)
        self.initialized = torch.where(
            changed,
            torch.ones_like(self.initialized),
            self.initialized,
        )

        return illegal_switch.to(dtype=self.context.dtype)
    

    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:

        if env_ids is None:
            self.duration.zero_()
            self.last_foot_state.zero_()
            self.initialized.zero_()
        else:
            self.duration[env_ids] = 0.0
            self.last_foot_state[env_ids] = False
            self.initialized[env_ids] = False


@register_reward
class FootSlidingVelocityL2(BaseRewardTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.foot_geom_ids = model_context.foot_geom_ids


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        landed = task_context.state.foot_ground_contact
        foot_xvel = task_context.state.geom_xvel[:, self.foot_geom_ids, :]
        foot_velocity = foot_xvel[..., 3:5]
        return ((foot_velocity * landed.unsqueeze(-1)).square()).mean(dim=(-1, -2))


@register_reward
class FootLiftHeightDiffCommandWeightedExp(BaseRewardTerm):

    def __init__(
        self,
        model_context: ModelContext,
        target_height: float,
        height_std: float = 0.03,
        command_std: float = 0.5,
        command_names: Sequence[str] = ("lin_vel_x", "lin_vel_y", "ang_vel_z"),
        *args, **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        if height_std <= 0.0:
            raise ValueError("'height_std' must be positive.")
        if command_std <= 0.0:
            raise ValueError("'command_std' must be positive.")

        self.foot_geom_ids = model_context.foot_geom_ids
        self.target_height = target_height
        self.height_std = height_std
        self.command_std = command_std
        self.command_names = command_names


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:

        command_norm = torch.linalg.norm(
            command_vector(task_context, self.command_names),
            dim=-1,
        )
        
        foot_height = task_context.state.geom_xpos[:, self.foot_geom_ids, 2]
        swinging = ~task_context.state.foot_ground_contact
        height_reward = torch.exp(
            -((foot_height - self.target_height) / self.height_std).square()
        )
        command_gate = 1.0 - torch.exp(-command_norm / self.command_std)
        return (height_reward * command_gate.unsqueeze(-1) * swinging).mean(dim=-1)


@register_reward
class FootLiftHeightDiffCommandGatedL2(BaseRewardTerm):

    def __init__(
        self,
        model_context: ModelContext,
        target_height: float,
        height_std: float | None = None,
        command_names: Sequence[str] = ("lin_vel_x", "lin_vel_y", "ang_vel_z"),
        *args, **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        if (
            not isinstance(target_height, (int, float))
            or isinstance(target_height, bool)
            or not math.isfinite(target_height)
            or target_height <= 0.0
        ):
            raise ValueError("'target_height' must be positive and finite.")
        if height_std is not None and (
            not isinstance(height_std, (int, float))
            or isinstance(height_std, bool)
            or not math.isfinite(height_std)
            or height_std <= 0.0
        ):
            raise ValueError("'height_std' must be positive and finite.")
        if not command_names or any(
            not isinstance(name, str) or not name
            for name in command_names
        ):
            raise ValueError("'command_names' must contain valid names.")

        self.foot_geom_ids = model_context.foot_geom_ids
        self.target_height = target_height
        self.height_std = height_std if height_std is not None else target_height
        self.command_names = command_names


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:

        foot_height = task_context.state.geom_xpos[:, self.foot_geom_ids, 2]
        command_norm = torch.linalg.norm(
            command_vector(task_context, self.command_names),
            dim=-1,
        ).unsqueeze(-1)
        command_gate = torch.where(
            command_norm <= 0.1,
            torch.zeros_like(command_norm),
            torch.ones_like(command_norm)
        )
        swinging = ~task_context.state.foot_ground_contact
        target_height = self.target_height * command_gate * swinging
        height_diff = foot_height - target_height
        height_diff_norm = height_diff / self.height_std

        return height_diff_norm.square().mean(dim=-1)


@register_reward
class QuadrupedalFootVelocityDiffL2(BaseRewardTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.foot_geom_ids = model_context.foot_geom_ids
        if self.foot_geom_ids.numel() != 4:
            raise ValueError("QuadrupedalFootVelocityDiffL2 requires exactly four feet.")


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:

        foot_xvel = task_context.state.geom_xvel[:, self.foot_geom_ids, :]
        foot_velocity = foot_xvel[..., 3:6]
        diagonal_diff = foot_velocity[:, [0, 1]] - foot_velocity[:, [3, 2]]
        return diagonal_diff.square().mean(dim=(-1, -2))


@register_reward
class FootContactWithoutCommand(BaseRewardTerm):

    def __init__(
        self,
        command_names: Sequence[str] = ("lin_vel_x", "lin_vel_y", "ang_vel_z"),
        *args, **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.command_names = command_names


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        landed = task_context.state.foot_ground_contact
        command_norm = torch.linalg.norm(
            command_vector(task_context, self.command_names),
            dim=-1,
        )
        idle = command_norm <= 0.1

        return landed.to(dtype=self.context.dtype).mean(dim=-1) * idle
