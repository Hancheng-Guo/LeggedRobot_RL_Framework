from envs.tasks.utils import StateOwner
from abc import ABC, abstractmethod
import torch

from utils import RuntimeContext


class BaseActionTerm(StateOwner, ABC):

    def __init__(
        self,
        name: str,
        output_dim: int,
        context: RuntimeContext,
        *args, **kwargs,
    ) -> None:

        super().__init__(name=name)

        self.input_dim: int
        self.output_dim = output_dim
        self.context = context


    @abstractmethod
    def process(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        pass
