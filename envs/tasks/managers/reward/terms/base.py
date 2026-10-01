import torch
from abc import ABC, abstractmethod

from envs.tasks.utils import TaskContext
from utils import RuntimeContext


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
        raise NotImplementedError


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        pass
