from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

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


class BasePolicy(torch.nn.Module, ABC):

    def __init__(self, context: RuntimeContext) -> None:
        super().__init__()
        self.context = context


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


    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
