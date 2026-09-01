from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from app.utils.context import RuntimeContext
from envs.base import BaseEnv
from rl.algorithms.base import OnPolicyAlgorithm
from runners.callbacks.base import BaseCallback
from runners.callbacks.stage import StageCallback
from utils.component import Component


class BaseRunner(ABC):

    def __init__(
        self,
        context: RuntimeContext,
    ) -> None:

        self.context = context
        self.environment: BaseEnv
        self.algorithm: OnPolicyAlgorithm

        self.current_iteration: int

        self.max_iterations: int
        self.rollout_length: int
        self.callbacks: list[BaseCallback] = []
        self.stop_callback: list[BaseCallback] = []


    @abstractmethod
    def config_update(
        self,
        component: Component, 
        max_iterations: int | None = None,
        rollout_length: int | None = None,
        rollout_length_history_size: int | None = None,
        callbacks: Sequence[str | Mapping[str, Any]] | None = None,
        stage_index: int | None = None,
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
