from typing import Any
from abc import ABC, abstractmethod

from app.utils.context import RuntimeContext
from runners.callbacks.stage import StageCallback


class BaseRunner(ABC):

    def __init__(
        self,
        context: RuntimeContext,
    ) -> None:
        self.max_iterations = None
        self.rollout_length = None
        self.algorithm = None
        self.environment = None
        self.stage_manager = None
        self.callbacks = []


    @abstractmethod
    def config_update(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def stage_update(
        self,
        stage_callback: StageCallback
    ):
        pass

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
