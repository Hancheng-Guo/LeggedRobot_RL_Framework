import torch
import numpy as np
from pathlib import Path
from typing import Any
from abc import ABC, abstractmethod

from .utils import ModelContext, SimulatorState
from .terrain import TerrainConfig
from utils import RuntimeContext


class BaseSimulator(ABC):

    SUPPORTED_RENDER_MODES = frozenset(("human", "rgb_array"))
    SUPPORTS_CONCURRENT_INSTANCES = True
    FOOT_CONTACT_FORCE_THRESHOLD: float = 15.0

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
        self.terrain_config = TerrainConfig()


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


    @property
    def active_render_mode(self) -> str | None:
        return self.render_mode


    def has_playback_render_mode(self) -> bool:
        return self.render_mode is not None


    def get_ground_heights(
        self,
        xy: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample world-frame ground z for [env, sample, xy] query points.

        A subset query uses rows in the same order as ``env_ids``. Backends
        override this when their ground is not the z=0 plane.
        """
        if xy.ndim != 3 or xy.shape[-1] != 2:
            raise ValueError("Ground queries must have shape [env, sample, 2].")
        expected = self.num_envs if env_ids is None else env_ids.numel()
        if xy.shape[0] != expected:
            raise ValueError("Ground query rows must match the selected environments.")
        if not self.terrain_config.is_plane:
            raise NotImplementedError("Ground sampling requires a terrain backend.")
        return torch.zeros(xy.shape[:2], dtype=xy.dtype, device=xy.device)


    def prepare_playback(
        self,
        render_mode: str | None = None
    ) -> None:
        """Prepare simulator-specific playback resources, if needed."""


    def is_playback_open(self) -> bool:
        return True


    def terminate_playback(self) -> None:
        """Release simulator-specific playback resources, if any."""


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
    ) -> SimulatorState:
        """Return simulator state for all or selected environments.

        Implementations may return views backed by reusable internal buffers.
        A returned tensor is guaranteed to remain valid only until the next
        state read that reuses the same buffer. Callers must clone values that
        need to outlive that boundary.
        """
        raise NotImplementedError
