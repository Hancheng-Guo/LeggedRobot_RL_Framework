import torch

from envs.tasks.managers.action.terms.base import BaseActionTerm
from envs.tasks.managers.action.terms.registry import register_action
from envs.simulators.utils.context import ModelContext


@register_action
class HardClamp(BaseActionTerm):

    def __init__(
        self,
        min_value: float,
        max_value: float,
        *args, **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        
        self.input_dim = self.output_dim

        self.min_value = min_value
        self.max_value = max_value


    def process(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:

        return value.clamp(self.min_value, self.max_value)