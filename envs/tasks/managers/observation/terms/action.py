import torch

from envs.tasks.managers.observation.terms.base import BaseObservationTerm
from envs.tasks.managers.observation.terms.registry import register_observation
from envs.tasks.utils.context import TaskContext


@register_observation
class LastAction(BaseObservationTerm):

    def compute(self, task_context: TaskContext) -> torch.Tensor:
        return task_context.action

