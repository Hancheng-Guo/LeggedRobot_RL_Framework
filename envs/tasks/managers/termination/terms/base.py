import torch
from abc import ABC, abstractmethod

from envs.tasks.utils import TaskContext
from utils import RuntimeContext


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
        raise NotImplementedError


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        pass
