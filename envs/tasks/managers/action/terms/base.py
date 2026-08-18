from abc import ABC, abstractmethod
import torch

from envs.simulators.utils.context import ModelContext


class BaseActionTerm(ABC):

    def __init__(
        self,
        output_dim: int,
        *args, **kwargs,
    ) -> None:

        self.input_dim: int
        self.output_dim = output_dim


    @abstractmethod
    def process(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:
        pass
