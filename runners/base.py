from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.utils.context import RuntimeContext
from envs.base import BaseEnv
from rl.algorithms.base import OnPolicyAlgorithm
from runners.callbacks.base import BaseCallback
from runners.callbacks.stage import StageCallback
from utils.component import Component


if TYPE_CHECKING:
    from runners.utils.frames import VideoFormat, VideoFormats


class BaseRunner(ABC):

    def __init__(
        self,
        context: RuntimeContext,
    ) -> None:

        self.context = context
        self.environment: BaseEnv
        self.algorithm: OnPolicyAlgorithm

        self.current_iteration = -1
        self.stage_index: int | None = None

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
        num_steps: int = 5000,
        formats: VideoFormat | VideoFormats = "gif",
    ) -> None:
        raise NotImplementedError


    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


    @abstractmethod
    def save(
        self,
        path: Path | None = None,
    ) -> Path:
        raise NotImplementedError


    @abstractmethod
    def prepare_checkpoint_load(self, path: Path) -> dict[str, Any]:
        """Read metadata needed to construct modules before state loading."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support checkpoint loading."
        )


    @abstractmethod
    def load(self, load_optimizer: bool = False) -> None:
        """Restore checkpoint state after runner configuration."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support checkpoint loading."
        )
