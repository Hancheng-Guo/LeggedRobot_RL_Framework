import torch
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from envs.simulators.utils import ModelContext
from envs.tasks.utils import StateOwner
from utils import RuntimeContext


class BaseCurriculumTerm(StateOwner, ABC):

    def __init__(
        self,
        name: str,
        num_envs: int,
        context: RuntimeContext,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:

        super().__init__(name=name)
        
        self.num_envs = num_envs
        self.context = context
        self.model_context = model_context
        self.training = True
        self.persistent_assignments = False
        self.freeze_assignments = False


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


    def set_training_mode(self, training: bool) -> None:
        self.training = training
        self.freeze_assignments = not training and self.persistent_assignments


    def finalize_episodes(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        pass


    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {"type": type(self).__name__}


    def playback_state_dict(self, env_index: int) -> dict[str, Any]:
        """Project term-owned checkpoint state to a single playback environment."""
        return self.checkpoint_state_dict()


    def resample(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        pass


    def validate_checkpoint_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("Curriculum term checkpoint must be a mapping.")
        if state.get("type") != type(self).__name__:
            raise ValueError("Curriculum checkpoint term type does not match configuration.")
