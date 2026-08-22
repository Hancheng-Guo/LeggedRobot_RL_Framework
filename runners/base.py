from abc import ABC, abstractmethod

from app.utils.context import RuntimeContext
from runners.callbacks.stage import StageCallback


class BaseRunner(ABC):

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
        callback_names: list[str] | None = None,
        *args, **kwargs
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def stage_update(
        self,
        stage_callback: StageCallback
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
