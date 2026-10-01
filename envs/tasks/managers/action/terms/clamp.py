import torch

from .base import BaseActionTerm
from .registry import register_action


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
