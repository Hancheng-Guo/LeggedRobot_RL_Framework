import torch
from abc import ABC, abstractmethod

from envs.simulators.utils import ModelContext
from utils import RuntimeContext


class BaseCurriculumTerm(ABC):

    def __init__(
        self,
        num_envs: int,
        context: RuntimeContext,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:
        
        self.num_envs = num_envs
        self.context = context
        self.model_context = model_context


    @abstractmethod
    def update(
        self,
        *args, **kwargs
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


    def collect_metrics(self) -> dict[str, torch.Tensor]:
        return {}


    def get_metric_names(self) -> set[str]:
        """Declare supported metrics even before their first values exist."""
        return set()


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        pass
