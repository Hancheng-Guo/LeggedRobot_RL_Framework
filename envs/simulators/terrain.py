"""Backend-neutral terrain geometry and physical application interfaces.

Sampling configuration, allocation and spawn policy belong to task randomization
terms. This module contains no task dependencies or simulator engine imports.
"""
from __future__ import annotations

import math
import torch
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Protocol

from utils import RuntimeContext


@dataclass(frozen=True)
class TerrainGeometry:
    """Only collision buffer dimensions and bounds; no sampling policy."""
    patch_size: tuple[float, float]
    grid_shape: tuple[int, int]  # rows (y), columns (x)
    height_bound: float

    def __post_init__(self):
        if (
            len(self.patch_size) != 2
            or any(not math.isfinite(v) or v <= 0 for v in self.patch_size)
            or len(self.grid_shape) != 2
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in self.grid_shape)
            or not math.isfinite(2 * self.height_bound)
            or self.height_bound <= 0
        ):
            raise ValueError("Invalid terrain collision geometry dimensions or bounds.")


@dataclass(frozen=True)
class TerrainSpawnGeometry:
    """Backend-measured reset pose and collision AABB relative to the root.

    These are physical measurements. Safe-area validation, clearance and random
    spawn choices belong to the task randomization term.
    """
    origin: tuple[float, float, float]
    root_position: tuple[float, float, float]
    collision_lower: tuple[float, float, float]
    collision_upper: tuple[float, float, float]


@dataclass(frozen=True)
class TerrainPatch:
    """Local metre coordinates, centred at zero; rows=y, columns=x.

    Backends must use the supplied vertices, split each cell on its lower-left
    to upper-right diagonal, and query those same triangles. No extrapolation
    outside the patch is defined. Sampling and spawn rules belong to the task.
    """

    x: tuple[float, ...]
    y: tuple[float, ...]
    heights: tuple[tuple[float, ...], ...]


class TerrainBackend(Protocol):
    context: RuntimeContext
    num_envs: int

    def get_terrain_spawn_geometry(self) -> TerrainSpawnGeometry: ...
    def apply_terrain(self, env_ids_tensor: torch.Tensor, patches: Sequence[TerrainPatch]) -> None: ...
    def set_root_positions(self, env_ids_tensor: torch.Tensor, positions: torch.Tensor) -> None: ...
    def get_ground_heights(self, xy: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor: ...
    def capture_reset_state(self, env_ids_tensor: torch.Tensor) -> object: ...
    def restore_reset_state(self, state: object) -> None: ...
