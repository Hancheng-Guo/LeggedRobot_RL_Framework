import torch

from .base import BaseTerminationTerm
from .registry import register_termination
from envs.simulators.utils import ModelContext
from envs.tasks.utils import TaskContext, validate_height_reference, base_height, mask_ground_signal


@register_termination
class BaseHeight(BaseTerminationTerm):

    def __init__(
        self,
        model_context: ModelContext,
        min_height: float,
        height_reference: str = "ground",
        *args,
        **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)
        self.height_reference = validate_height_reference(height_reference)

        self.height_qpos_id = int(model_context.base_pos_qpos_ids[2].item())
        self.min_height = min_height


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:

        height = base_height(task_context, self.height_qpos_id, self.height_reference)
        value = height < self.min_height
        return mask_ground_signal(task_context, value, self.height_reference)
