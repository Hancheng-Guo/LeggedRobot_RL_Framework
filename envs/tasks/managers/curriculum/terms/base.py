import torch
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

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


    def checkpoint_state_dict(self) -> dict[str, Any]:
        return {"type": type(self).__name__}


    def validate_checkpoint_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("Curriculum term checkpoint must be a mapping.")
        if state.get("type") != type(self).__name__:
            raise ValueError("Curriculum checkpoint term type does not match configuration.")


    def load_checkpoint_state_dict(self, state: Mapping[str, Any]) -> None:
        self.validate_checkpoint_state_dict(state)
