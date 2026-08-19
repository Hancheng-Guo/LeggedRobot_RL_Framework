import torch

from envs.tasks.managers.observation.terms.base import BaseObservationTerm
from envs.tasks.managers.observation.terms.registry import register_observation
from envs.tasks.utils.context import TaskContext


@register_observation
class BaseAngularVelocity(BaseObservationTerm):

    def compute(self, task_context: TaskContext) -> torch.Tensor:
        return task_context.state["qvel"][:, 3:6]


@register_observation
class ProjectedGravity(BaseObservationTerm):

    def compute(self, task_context: TaskContext) -> torch.Tensor:
        quaternion = task_context.state["qpos"][:, 3:7]
        quaternion = quaternion / quaternion.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(quaternion.dtype).eps)

        w = quaternion[:, 0:1]
        xyz = quaternion[:, 1:4]
        gravity = torch.zeros_like(xyz)
        gravity[:, 2] = -1.0

        return (
            gravity * (2.0 * w.square() - 1.0)
            - 2.0 * w * torch.cross(xyz, gravity, dim=-1)
            + 2.0 * xyz * (xyz * gravity).sum(dim=-1, keepdim=True)
        )


@register_observation
class JointPosition(BaseObservationTerm):

    def compute(self, task_context: TaskContext) -> torch.Tensor:
        return task_context.state["qpos"][:, 7:]


@register_observation
class JointVelocity(BaseObservationTerm):

    def compute(self, task_context: TaskContext) -> torch.Tensor:
        return task_context.state["qvel"][:, 6:]

