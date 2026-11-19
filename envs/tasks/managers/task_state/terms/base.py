import torch
from abc import ABC, abstractmethod
from typing import Any

from envs.tasks.utils import TaskContext, StateOwner
from utils import RuntimeContext


class BaseTaskStateTerm(StateOwner, ABC):

    def __init__(
        self,
        name: str,
        num_envs: int,
        context: RuntimeContext,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(name=name)
        
        self.num_envs = num_envs
        self.context = context


    @property
    @abstractmethod
    def value(self) -> torch.Tensor:
        raise NotImplementedError


    @abstractmethod
    def update(
        self,
        task_context: TaskContext
    ) -> None:
        raise NotImplementedError


    @abstractmethod
    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        raise NotImplementedError
