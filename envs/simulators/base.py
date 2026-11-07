import torch
import numpy as np
from pathlib import Path
from typing import Any
from abc import ABC, abstractmethod
from collections.abc import Sequence

from .terrain import TerrainPatch, TerrainSpawnGeometry
from .utils import ModelContext, SimulatorState
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
        return torch.zeros(xy.shape[:2], dtype=xy.dtype, device=xy.device)


    def get_terrain_spawn_geometry(self) -> TerrainSpawnGeometry:
        """Report physical reset bounds; the environment chooses safe spawns."""
        raise NotImplementedError("This backend does not expose terrain reset geometry.")


    def apply_terrain(
        self,
        env_ids_tensor: torch.Tensor,
        patches: Sequence[TerrainPatch],
    ) -> None:
        """Install supplied geometry without sampling or resetting robot state."""
        raise NotImplementedError("This backend cannot apply geometric terrain yet.")


    def set_root_positions(
        self,
        env_ids_tensor: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Apply world positions chosen by the environment, preserving joints."""
        raise NotImplementedError("This backend cannot apply supplied root positions yet.")


    def capture_reset_state(self, env_ids_tensor: torch.Tensor) -> object:
        """Capture backend-owned physical state for a recoverable reset failure."""
        raise NotImplementedError("This backend cannot snapshot a terrain reset yet.")


    def restore_reset_state(self, state: object) -> None:
        raise NotImplementedError("This backend cannot restore a terrain reset yet.")


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
