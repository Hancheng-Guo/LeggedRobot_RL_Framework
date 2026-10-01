import torch

from .base import BaseTerminationTerm
from .registry import register_termination
from envs.simulators.utils import ModelContext
from envs.tasks.utils import TaskContext


@register_termination
class BaseHeight(BaseTerminationTerm):

    def __init__(
        self,
        model_context: ModelContext,
        min_height: float,
        *args,
        **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.height_qpos_id = model_context.base_pos_qpos_ids[2]
        self.min_height = min_height


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        return (
            task_context.state.qpos[:, self.height_qpos_id]
            < self.min_height
        )
