from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass
from torch import Tensor


@dataclass
class PolicyOutput:

    action: Tensor
    log_prob: Tensor | None = None
    value: Tensor | None = None


class BaseAlgorithm(ABC):

    def __init__(self) -> None:
        self.model = None


    @abstractmethod
    def act(self, observation: Any) -> Any:
        pass


    @abstractmethod
    def eval(self) -> None:
        pass


    @abstractmethod
    def update(self) -> None:
        pass


class OnPolicyAlgorithm(BaseAlgorithm):

    @abstractmethod
    def compute_returns(self) -> None:
        pass


class OffPolicyAlgorithm(BaseAlgorithm):

    @abstractmethod
    def sample_batch(self) -> None:
        pass
