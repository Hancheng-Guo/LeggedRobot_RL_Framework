from abc import ABC, abstractmethod

import torch

from app.utils.context import RuntimeContext
from envs.simulators.utils.context import ModelContext
from envs.tasks.base import TaskContext


class BaseCommandTerm(ABC):

    def __init__(
        self,
        num_envs: int,
        context: RuntimeContext,
        model_context: ModelContext,
        *args,
        **kwargs,
    ) -> None:

        self.num_envs = num_envs
        self.context = context
        self.model_context = model_context

        self.command = torch.zeros(
            (self.num_envs, 1),
            dtype=self.context.dtype,
            device=self.context.device,
        )
        self.last_command = torch.zeros_like(self.command)


    @abstractmethod
    def update(
        self,
        task_context: TaskContext,
    ) -> None:
        pass


    @abstractmethod
    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        pass