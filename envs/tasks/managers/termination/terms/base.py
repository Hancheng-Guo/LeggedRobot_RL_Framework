import torch
from abc import ABC, abstractmethod

from envs.tasks.utils import TaskContext, StateOwner
from utils import RuntimeContext


class BaseTerminationTerm(StateOwner, ABC):

    def __init__(
        self,
        name: str,
        context: RuntimeContext,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(name=name)

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
