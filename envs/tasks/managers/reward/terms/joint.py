import torch

from envs.simulators.utils.context import ModelContext
from envs.tasks.managers.reward.terms.base import BaseRewardTerm
from envs.tasks.managers.reward.terms.registry import register_reward
from envs.tasks.utils.context import TaskContext


@register_reward
class JointVelocityL2(BaseRewardTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.qvel_ids = model_context.joint_qvel_ids


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:

        return torch.mean(
            task_context.state["qvel"][:, self.qvel_ids].square(),
            dim=-1,
        )


@register_reward
class JointPositionL2(BaseRewardTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.qpos_ids = model_context.joint_qpos_ids
        self.default_position = model_context.joint_default_pos


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        position = task_context.state["qpos"][:, self.qpos_ids]
        error = position - self.default_position
        return torch.sum(error.square(), dim=-1)


@register_reward
class JointLimitViolationL1(BaseRewardTerm):

    def __init__(
        self,
        model_context: ModelContext,
        lower_limits: list[float] | None = None,
        upper_limits: list[float] | None = None,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.qpos_ids = model_context.joint_qpos_ids
        if lower_limits is None:
            self.lower_limits = model_context.joint_pos_limits[:, 0]
        else:
            self.lower_limits = torch.as_tensor(
                lower_limits,
                dtype=self.context.dtype,
                device=self.context.device,
            )
        if upper_limits is None:
            self.upper_limits = model_context.joint_pos_limits[:, 1]
        else:
            self.upper_limits = torch.as_tensor(
                upper_limits,
                dtype=self.context.dtype,
                device=self.context.device,
            )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        position = task_context.state["qpos"][:, self.qpos_ids]
        upper_excess = (position - self.upper_limits).clamp_min(0.0)
        lower_excess = (self.lower_limits - position).clamp_min(0.0)
        return torch.sum(upper_excess + lower_excess, dim=-1)


@register_reward
class JointPowerL1(BaseRewardTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.qvel_ids = model_context.joint_qvel_ids


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        joint_velocity = task_context.state["qvel"][:, self.qvel_ids]
        actuator_force = task_context.state["actuator_force"]
        return torch.sum((actuator_force * joint_velocity).abs(), dim=-1)
