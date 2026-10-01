from abc import ABC, abstractmethod

import torch

from envs.simulators.utils import ModelContext
from envs.tasks.utils import TaskContext
from utils import RuntimeContext


class BaseCommandTerm(ABC):

    def __init__(
        self,
        num_envs: int,
        context: RuntimeContext,
        model_context: ModelContext,
        dim: int = 1,
        *args, **kwargs,
    ) -> None:

        if (
            not isinstance(dim, int)
            or isinstance(dim, bool)
            or dim <= 0
        ):
            raise ValueError("'command_dim' must be a positive integer.")

        self.num_envs = num_envs
        self.context = context
        self.model_context = model_context
        self.dim = dim

        self.command = torch.zeros(
            (self.num_envs, self.dim),
            dtype=self.context.dtype,
            device=self.context.device,
        )


    @abstractmethod
    def update(
        self,
        task_context: TaskContext,
    ) -> None:
        raise NotImplementedError


    @abstractmethod
    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        raise NotImplementedError
