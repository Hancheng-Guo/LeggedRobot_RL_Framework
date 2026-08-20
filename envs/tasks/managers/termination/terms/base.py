import torch
from abc import ABC, abstractmethod

from app.utils.context import RuntimeContext
from envs.tasks.utils.context import TaskContext


class BaseTerminationTerm(ABC):

    def __init__(
        self,
        context: RuntimeContext,
        *args,
        **kwargs,
    ) -> None:
        self.context = context


    @abstractmethod
    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        pass


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        pass
