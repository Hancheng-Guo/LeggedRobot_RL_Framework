from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass
from torch import Tensor

from app.utils.context import RuntimeContext


@dataclass
class PolicyOutput:

    action: Tensor
    log_prob: Tensor | None = None
    value: Tensor | None = None


class BaseAlgorithm(ABC):

    def __init__(
        self,
        context: RuntimeContext
    ) -> None:
        self.context = context
        self.model = None


    @abstractmethod
    def config_update(self, *args, **kwargs):
        pass


    @abstractmethod
    def act(
        self,
        observation: Any,
        deterministic: bool | None = None
    ) -> Any:
        pass


    @abstractmethod
    def eval(self) -> None:
        pass


    @abstractmethod
    def update(self) -> None:
        pass


class OnPolicyAlgorithm(BaseAlgorithm):

    @abstractmethod
    def compute_returns(
        self,
        last_obs,
    ) -> None:
        pass


    @abstractmethod
    def process_transition(
        self,
        obs,
        policy_output,
        reward,
        terminated,
        truncated,
        next_obs,
        info,
    ) -> None:
        pass


class OffPolicyAlgorithm(BaseAlgorithm):

    @abstractmethod
    def sample_batch(self) -> None:
        pass
