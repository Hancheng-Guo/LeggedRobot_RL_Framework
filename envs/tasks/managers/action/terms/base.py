from abc import ABC, abstractmethod
import torch

from app.utils.context import RuntimeContext


class BaseActionTerm(ABC):

    def __init__(
        self,
        output_dim: int,
        context: RuntimeContext,
        *args, **kwargs,
    ) -> None:

        self.input_dim: int
        self.output_dim = output_dim
        self.context = context


    @abstractmethod
    def process(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:
        pass


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        pass
