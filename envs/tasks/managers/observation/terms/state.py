import torch

from envs.simulators.utils.context import ModelContext
from envs.tasks.managers.observation.terms.base import BaseObservationTerm
from envs.tasks.managers.observation.terms.registry import register_observation
from envs.tasks.utils.context import TaskContext


@register_observation
class BaseAngularVelocity(BaseObservationTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.qvel_ids = model_context.base_ang_vel_qvel_ids

    def compute(self, task_context: TaskContext) -> torch.Tensor:
        return task_context.state["qvel"][:, self.qvel_ids]


@register_observation
class ProjectedGravity(BaseObservationTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.qpos_ids = model_context.base_quat_qpos_ids
        gravity = model_context.gravity
        gravity_norm = gravity.norm()
        if gravity_norm <= torch.finfo(gravity.dtype).eps:
            raise ValueError(
                "ProjectedGravity requires a non-zero gravity vector."
            )
        self.gravity = gravity / gravity_norm

    def compute(self, task_context: TaskContext) -> torch.Tensor:
        quaternion = task_context.state["qpos"][:, self.qpos_ids]
        quaternion = quaternion / quaternion.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(quaternion.dtype).eps)

        w = quaternion[:, 0:1]
        xyz = quaternion[:, 1:4]
        gravity = self.gravity.expand_as(xyz)

        return (
            gravity * (2.0 * w.square() - 1.0)
            - 2.0 * w * torch.cross(xyz, gravity, dim=-1)
            + 2.0 * xyz * (xyz * gravity).sum(dim=-1, keepdim=True)
        )


@register_observation
class JointPosition(BaseObservationTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.qpos_ids = model_context.joint_qpos_ids

    def compute(self, task_context: TaskContext) -> torch.Tensor:
        return task_context.state["qpos"][:, self.qpos_ids]


@register_observation
class JointVelocity(BaseObservationTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.qvel_ids = model_context.joint_qvel_ids

    def compute(self, task_context: TaskContext) -> torch.Tensor:
        return task_context.state["qvel"][:, self.qvel_ids]
