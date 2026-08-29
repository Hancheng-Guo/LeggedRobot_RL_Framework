import torch

from envs.tasks.managers.observation.terms.base import BaseObservationTerm
from envs.tasks.managers.observation.terms.registry import register_observation
from envs.tasks.utils.context import TaskContext


@register_observation
class LastAction(BaseObservationTerm):

    def __init__(
        self,
        action_dim: int,
        *args,
        **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)
        
        self.output_dim = action_dim


    def compute(self, task_context: TaskContext) -> torch.Tensor:
        return task_context.action
