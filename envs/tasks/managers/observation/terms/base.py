from abc import ABC, abstractmethod

import torch

from app.utils.context import RuntimeContext
from envs.simulators.utils.context import ModelContext
from envs.tasks.utils.context import TaskContext


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


    @abstractmethod
    def compute(
        self,
        task_context: TaskContext,
    ) -> torch.Tensor:
        pass


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        pass
