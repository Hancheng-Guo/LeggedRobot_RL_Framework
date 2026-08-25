from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from app.utils.context import RuntimeContext
from runners.callbacks.base import BaseCallback
from runners.callbacks.stage import StageCallback


class BaseRunner(ABC):

    stop_callback: list[BaseCallback]

    def __init__(
        self,
        context: RuntimeContext,
    ) -> None:

        self.context = context
        self.environment = None
        self.algorithm = None

        self.max_iterations = None
        self.rollout_length = None
        self.callbacks = []


    @abstractmethod
    def config_update(
        self,
        max_iterations: int | None = None,
        rollout_length: int | None = None,
        callbacks: Sequence[str | Mapping[str, Any]] | None = None,
        *args, **kwargs
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def stage_update(
        self,
        stage_callback: StageCallback | None,
    ):
        raise NotImplementedError

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError


    @abstractmethod
    def test(
        self,
        num_episodes: int = 1000,
    ) -> None:
        raise NotImplementedError


    @abstractmethod
    def play(
        self,
        num_steps: int = 5000
    ) -> None:
        raise NotImplementedError


    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError
