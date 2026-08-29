import torch
from collections.abc import Sequence

from envs.simulators.utils.context import ModelContext
from envs.tasks.managers.reward.terms.base import BaseRewardTerm
from envs.tasks.managers.reward.terms.registry import register_reward
from envs.tasks.managers.reward.terms.utils import command_vector
from envs.tasks.utils.context import TaskContext


_FOOT_CONTACT_FORCE_THRESHOLD = 15.0


def _foot_landed(
    task_context: TaskContext,
    geom_foot_ids: torch.Tensor,
    ground_geom_ids: torch.Tensor,
) -> torch.Tensor:
    
    contact_geom_ids = task_context.state["contact_geom_ids"]
    contact_forces = task_context.state["contact_forces"]
    landed = torch.zeros(
        (contact_geom_ids.shape[0], geom_foot_ids.numel()),
        dtype=torch.bool,
        device=contact_geom_ids.device,
    )
    if contact_geom_ids.shape[1] == 0:
        return landed

    geom1 = contact_geom_ids[..., 0]
    geom2 = contact_geom_ids[..., 1]
    normal_force = contact_forces[..., 0].abs()
    forceful_contact = normal_force >= _FOOT_CONTACT_FORCE_THRESHOLD

    for index, geom_foot_id in enumerate(geom_foot_ids):
        foot_ground_contact = (
            ((geom1 == geom_foot_id) & torch.isin(geom2, ground_geom_ids))
            | ((geom2 == geom_foot_id) & torch.isin(geom1, ground_geom_ids))
        )
        landed[:, index] = (foot_ground_contact & forceful_contact).any(dim=-1)
    return landed


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

        self.geom_foot_ids = model_context.geom_foot_ids
        self.ground_geom_ids = model_context.geom_floor_ids
        self.sigma = sigma
        self.command_names = command_names

        self.last_foot_state = torch.zeros(
            (num_envs, self.geom_foot_ids.numel()),
            dtype=torch.bool,
            device=self.context.device,
        )
        self.duration = torch.zeros(
            (num_envs, self.geom_foot_ids.numel()),
            dtype=self.context.dtype,
            device=self.context.device,
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        landed = _foot_landed(
            task_context,
            self.geom_foot_ids,
            self.ground_geom_ids,
        )
        unchanged = landed == self.last_foot_state
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
        return torch.exp(-command_norm.unsqueeze(-1) * scaled_duration).mean(dim=-1)


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

        self.geom_foot_ids = model_context.geom_foot_ids
        self.ground_geom_ids = model_context.geom_floor_ids
        self.sigma = sigma
        self.command_names = command_names

        self.last_foot_state = torch.zeros(
            (num_envs, self.geom_foot_ids.numel()),
            dtype=torch.bool,
            device=self.context.device,
        )
        self.duration = torch.zeros(
            (num_envs, self.geom_foot_ids.numel()),
            dtype=self.context.dtype,
            device=self.context.device,
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        landed = _foot_landed(
            task_context,
            self.geom_foot_ids,
            self.ground_geom_ids,
        )
        unchanged = landed == self.last_foot_state
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
            -command_norm.unsqueeze(-1) * scaled_duration ** 3
        ).mean(dim=-1)
    

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
class FootSlidingVelocityL2(BaseRewardTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.geom_foot_ids = model_context.geom_foot_ids
        self.ground_geom_ids = model_context.geom_floor_ids


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        landed = _foot_landed(
            task_context,
            self.geom_foot_ids,
            self.ground_geom_ids,
        )
        foot_xvel = task_context.state["geom_xvel"][:, self.geom_foot_ids, :]
        foot_velocity = foot_xvel[..., 3:5]
        return torch.sum(
            (foot_velocity * landed.unsqueeze(-1)).square(),
            dim=(-1, -2)
        )


@register_reward
class FootLiftHeightVelocityWeightedExp(BaseRewardTerm):

    def __init__(
        self,
        model_context: ModelContext,
        target_height: float,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.geom_foot_ids = model_context.geom_foot_ids
        self.target_height = target_height


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        foot_height = task_context.state["geom_xpos"][:, self.geom_foot_ids, 2]
        foot_speed = torch.linalg.norm(
            task_context.state["geom_xvel"][:, self.geom_foot_ids, 3:5],
            dim=-1,
        )
        return torch.sum(
            torch.exp(
                -(foot_height - self.target_height).square() * foot_speed
            ),
            dim=-1,
        )


@register_reward
class QuadrupedalFootVelocityDiffL2(BaseRewardTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.geom_foot_ids = model_context.geom_foot_ids
        if self.geom_foot_ids.numel() != 4:
            raise ValueError("QuadrupedalFootVelocityDiffL2 requires exactly four feet.")


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:

        foot_xvel = task_context.state["geom_xvel"][:, self.geom_foot_ids, :]
        foot_velocity = foot_xvel[..., 3:6]
        diagonal_diff = foot_velocity[:, [0, 1]] - foot_velocity[:, [3, 2]]
        return diagonal_diff.square().sum(dim=(-1, -2))


@register_reward
class FootContactWithoutCommand(BaseRewardTerm):

    def __init__(
        self,
        model_context: ModelContext,
        command_names: Sequence[str] = ("lin_vel_x", "lin_vel_y", "ang_vel_z"),
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.geom_foot_ids = model_context.geom_foot_ids
        self.ground_geom_ids = model_context.geom_floor_ids
        self.command_names = command_names


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        landed = _foot_landed(
            task_context,
            self.geom_foot_ids,
            self.ground_geom_ids,
        )
        command_norm = torch.linalg.norm(
            command_vector(task_context, self.command_names),
            dim=-1,
        )
        idle = command_norm <= 0.1
        reward = landed.sum(dim=-1) * idle
        return reward.to(dtype=self.context.dtype)
