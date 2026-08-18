import torch

from envs.tasks.managers.action.terms.base import BaseActionTerm
from envs.tasks.managers.action.terms.registry import register_action
from envs.simulators.utils.context import ModelContext


@register_action
class LinearMap(BaseActionTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        
        self.input_dim = self.output_dim

        ctrl_range = model_context.actuator_ctrl_range
        self.center = 0.5 * (ctrl_range[:, 0] + ctrl_range[:, 1])
        self.scale = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])


    def process(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:

        return self.center + value * self.scale
        