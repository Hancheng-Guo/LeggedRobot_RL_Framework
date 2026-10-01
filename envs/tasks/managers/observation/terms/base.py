import torch
from abc import ABC, abstractmethod


from envs.tasks.utils import TaskContext
from utils import RuntimeContext


class BaseObservationTerm(ABC):

    def __init__(
        self,
        context: RuntimeContext,
        scale: float = 1.0,
        *args, **kwargs,
    ) -> None:

        if not isinstance(scale, (int, float)):
            raise TypeError("'scale' of observation term must be numeric.")

        self.context = context
        self.scale = float(scale)
        self.output_dim: int


    @abstractmethod
    def compute(
        self,
        task_context: TaskContext,
    ) -> torch.Tensor:
        raise NotImplementedError


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        pass
