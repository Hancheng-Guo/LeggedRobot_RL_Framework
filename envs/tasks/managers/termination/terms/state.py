import torch

from envs.simulators.utils.context import ModelContext
from envs.tasks.managers.termination.terms.base import BaseTerminationTerm
from envs.tasks.managers.termination.terms.registry import register_termination
from envs.tasks.utils.context import TaskContext


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
            task_context.state["qpos"][:, self.height_qpos_id]
            < self.min_height
        )
