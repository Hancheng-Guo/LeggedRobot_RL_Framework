import torch

from envs.simulators.utils.context import ModelContext
from envs.tasks.managers.reward.terms.base import BaseRewardTerm
from envs.tasks.managers.reward.terms.registry import register_reward
from envs.tasks.utils.context import TaskContext


@register_reward
class BaseLinearVelocityZL2(BaseRewardTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.qvel_id = int(model_context.base_lin_vel_qvel_ids[2].item())

    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        return task_context.state["qvel"][:, self.qvel_id].square()


@register_reward
class BaseLinearVelocityZL2XySpeedWeighted(BaseRewardTerm):

    def __init__(
        self,
        model_context: ModelContext,
        min_speed: float = 0.5,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)
        
        self.z_qvel_id = int(model_context.base_lin_vel_qvel_ids[2].item())
        self.xy_qvel_ids = model_context.base_lin_vel_qvel_ids[:2]
        self.min_speed = min_speed


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        z_velocity_l2 = task_context.state["qvel"][:, self.z_qvel_id].square()
        xy_speed = torch.linalg.norm(
            task_context.state["qvel"][:, self.xy_qvel_ids],
            dim=-1,
        ).clamp_min(self.min_speed)
        return z_velocity_l2 / xy_speed


@register_reward
class BaseHeightL2(BaseRewardTerm):

    def __init__(
        self,
        model_context: ModelContext,
        target_height: float,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.height_qpos_id = int(model_context.base_pos_qpos_ids[2].item())
        self.target_height = target_height


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        height = task_context.state["qpos"][:, self.height_qpos_id]
        return (height - self.target_height).square()


@register_reward
class BaseHeightL2XySpeedWeighted(BaseRewardTerm):

    def __init__(
        self,
        model_context: ModelContext,
        target_height: float,
        min_speed: float = 0.5,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.height_qpos_id = int(model_context.base_pos_qpos_ids[2].item())
        self.target_height = target_height
        self.xy_qvel_ids = model_context.base_lin_vel_qvel_ids[:2]
        self.min_speed = min_speed


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        height = task_context.state["qpos"][:, self.height_qpos_id]
        height_error_l2 = (height - self.target_height).square()
        xy_speed = torch.linalg.norm(
            task_context.state["qvel"][:, self.xy_qvel_ids],
            dim=-1,
        ).clamp_min(self.min_speed)
        return height_error_l2 / xy_speed


@register_reward
class BaseAngularVelocityXyL2(BaseRewardTerm):

    def __init__(
        self,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        return torch.mean(
            task_context.state["base_ang_vel_body"][:, :2].square(),
            dim=-1,
        )


@register_reward
class ProjectedGravityXyL2(BaseRewardTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.qpos_ids = model_context.base_quat_qpos_ids
        gravity = model_context.gravity
        gravity_norm = gravity.norm()
        if gravity_norm <= torch.finfo(gravity.dtype).eps:
            raise ValueError(
                "ProjectedGravityXyL2 requires a non-zero gravity vector."
            )
        self.gravity = gravity / gravity_norm


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        quaternion = task_context.state["qpos"][:, self.qpos_ids]
        quaternion = quaternion / quaternion.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(quaternion.dtype).eps)

        w = quaternion[:, 0:1]
        xyz = quaternion[:, 1:4]
        gravity = self.gravity.expand_as(xyz)
        projected_gravity = (
            gravity * (2.0 * w.square() - 1.0)
            - 2.0 * w * torch.cross(xyz, gravity, dim=-1)
            + 2.0 * xyz * (xyz * gravity).sum(dim=-1, keepdim=True)
        )
        return torch.mean(projected_gravity[:, :2].square(), dim=-1)
