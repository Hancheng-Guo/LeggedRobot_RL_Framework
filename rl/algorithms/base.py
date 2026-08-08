from abc import ABC, abstractmethod
from typing import Any


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
