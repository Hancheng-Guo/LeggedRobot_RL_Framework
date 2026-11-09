import torch

from .base import BaseTerminationTerm
from .registry import register_termination
from envs.tasks.utils import TaskContext


@register_termination
class TerrainBounds(BaseTerminationTerm):
    """Optionally end an episode when the root or a foot leaves the random patch."""

    def compute(
        self,
        task_context: TaskContext,
    ) -> torch.Tensor:

        in_bounds = task_context.state.terrain_in_bounds
        if in_bounds is None:
            return torch.zeros(task_context.state.qpos.shape[0], dtype=torch.bool,
                               device=task_context.state.qpos.device)
        return ~in_bounds
