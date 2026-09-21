import torch
import numpy as np
from pathlib import Path
from typing import Any
from abc import ABC, abstractmethod

from envs.simulators.utils.context import ModelContext
from app.utils.context import RuntimeContext


class BaseSimulator(ABC):

    SUPPORTED_RENDER_MODES = frozenset(("human", "rgb_array"))
    SUPPORTS_CONCURRENT_INSTANCES = True

    def __init__(
        self,
        context: RuntimeContext
    ) -> None:

        self.context = context
        self.num_envs: int
        self.model_path: Path
        self.sim_dt: float
        self.frame_skip: int
        self.render_mode: str | None
        self.model_context: ModelContext


    @abstractmethod
    def config_update(
        self,
        num_envs: int | None = None,
        model_path: Path | str | None = None,
        sim_dt: float | None = None,
        frame_skip: int | None = None,
        render_mode: str | None = None,
        *args, **kwargs
    ) -> None:
        raise NotImplementedError


    @abstractmethod
    def _build_model_context(self) -> None:
        raise NotImplementedError


    @property
    def control_dt(self) -> float:

        return (
            self.sim_dt *
            self.frame_skip
        )


    @property
    def playback_env_index(self) -> int:
        return 0


    def _tensor(
        self,
        values: Any
    ) -> torch.Tensor:

        return torch.as_tensor(
            values,
            dtype=self.context.dtype,
            device=self.context.device,
        )


    def _index_tensor(
        self,
        values: Any
    ) -> torch.Tensor:

        return torch.as_tensor(
            values,
            dtype=torch.long,
            device=self.context.device,
        )


    @abstractmethod
    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        raise NotImplementedError


    @abstractmethod
    def step(
        self,
        action: torch.Tensor,
    ) -> None:
        raise NotImplementedError


    @abstractmethod
    def render(
        self,
    ) -> np.ndarray | None:
        raise NotImplementedError


    @abstractmethod
    def close(
        self,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_state(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError
