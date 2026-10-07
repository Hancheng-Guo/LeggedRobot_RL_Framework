import torch

from .base import BaseObservationTerm
from .registry import register_observation
from envs.tasks.utils import TaskContext


@register_observation
class QuadrupedalGaitPhase(BaseObservationTerm):

    def __init__(
        self,
        *args, **kwargs
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.output_dim = 4


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        phase = task_context.task_state.get("quadrupedal_gait_phase")
        if phase is None:
            raise ValueError(
                "QuadrupedalGaitPhase requires task state "
                "'quadrupedal_gait_phase'."
            )
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
