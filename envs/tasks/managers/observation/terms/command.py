import torch

from envs.tasks.managers.observation.terms.base import BaseObservationTerm
from envs.tasks.managers.observation.terms.registry import register_observation
from envs.tasks.utils.context import TaskContext


@register_observation
class Command(BaseObservationTerm):

    def __init__(
        self,
        command_dim: int,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)
        
        self.output_dim = command_dim


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        if not task_context.command:
            return task_context.action.new_empty(
                (task_context.action.shape[0], 0)
            )

        return torch.cat(tuple(task_context.command.values()), dim=-1)
