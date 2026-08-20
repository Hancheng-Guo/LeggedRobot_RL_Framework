import torch
from abc import ABC, abstractmethod

from app.utils.context import RuntimeContext
from envs.tasks.utils.context import TaskContext


class BaseRewardTerm(ABC):

    def __init__(
        self,
        context: RuntimeContext,
        weight: float = 1.0,
        *args, **kwargs,
    ) -> None:

        if not isinstance(weight, (int, float)):
            raise TypeError("'weight' of reward term must be numeric.")

        self.context = context
        self.weight = float(weight)


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
