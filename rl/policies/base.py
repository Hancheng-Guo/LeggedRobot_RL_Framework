import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.utils.context import RuntimeContext
from utils.component import Component


@dataclass(slots=True)
class PolicyActionOutput:
    action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor


@dataclass(slots=True)
class PolicyEvaluation:
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


RecurrentState = dict[str, torch.Tensor]


@runtime_checkable
class RecurrentPolicy(Protocol):

    @property
    def is_recurrent(self) -> bool: ...

    def get_recurrent_state(
        self,
        batch_size: int | None = None,
    ) -> RecurrentState: ...

    def reset_recurrent_state(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None: ...

    def evaluate_recurrent_sequences(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        initial_state: RecurrentState,
        reset_mask: torch.Tensor,
    ) -> tuple[PolicyEvaluation, RecurrentState]: ...


class BasePolicy(torch.nn.Module, ABC):

    def __init__(
        self,
        context: RuntimeContext
    ) -> None:
        
        super().__init__()
        
        self.context = context
        self._pending_checkpoint_state: Mapping[str, Any] | None = None


    @abstractmethod
    def config_update(
        self,
        component: Component,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError


    @abstractmethod
    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> PolicyActionOutput:
        raise NotImplementedError


    @abstractmethod
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> PolicyEvaluation:
        raise NotImplementedError


    @abstractmethod
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


    def set_train_mode(self) -> None:
        self.train()


    def set_eval_mode(self) -> None:
        self.eval()


    def checkpoint_state_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "state_dict": self.state_dict(),
        }


    def prepare_checkpoint_load(
        self,
        state: Mapping[str, Any],
    ) -> None:
        self._pending_checkpoint_state = state


    def load_checkpoint_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        self.load_state_dict(state["state_dict"])
        self._pending_checkpoint_state = None


    def save_module_artifacts(self, directory: Path) -> list[Path]:
        """Save portable named-module artifacts owned by this policy."""
        return []


    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
