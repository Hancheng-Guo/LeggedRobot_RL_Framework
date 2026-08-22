from abc import ABC, abstractmethod

import torch

from app.utils.context import RuntimeContext
from envs.simulators.utils.context import ModelContext


class BaseCurriculumTerm(ABC):

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


    @abstractmethod
    def update(
        self,
        *args, **kwargs
    ) -> dict[str, torch.Tensor]:
        pass


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        pass
