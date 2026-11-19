import torch
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from envs.simulators.utils import ModelContext
from envs.tasks.managers.curriculum import CurriculumSpaceConfigs
from envs.tasks.utils import TaskContext, StateOwner
from utils import RuntimeContext


class BaseCommandTerm(StateOwner, ABC):

    RESET_STATE_FIELDS = ("command",)

    def __init__(
        self,
        name: str,
        num_envs: int,
        context: RuntimeContext,
        model_context: ModelContext,
        dim: int = 1,
        *args, **kwargs,
    ) -> None:

        super().__init__(name=name)

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


    @classmethod
    def get_curriculum_references(
        cls,
        params: Mapping[str, Any],
    ) -> set[str]:
        return set()


    @classmethod
    def get_curriculum_space_config(
        cls, name: str, params: Mapping[str, Any],
    ) -> CurriculumSpaceConfigs:
        return {}


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
