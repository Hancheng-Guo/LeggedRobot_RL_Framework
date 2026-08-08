from typing import Any
from abc import ABC, abstractmethod


class BaseRunner(ABC):

    def __init__(self) -> None:
        self.max_iterations = None
        self.rollout_length = None
        self.algorithm = None
        self.environment = None
        self.stage_manager = None
        self.callbacks = []


    @abstractmethod
    def train(self) -> None:
        pass


    @abstractmethod
    def test(self) -> None:
        pass


    @abstractmethod
    def play(self) -> None:
        pass


    @abstractmethod
    def close(self) -> None:
        pass
