from __future__ import annotations

import torch
import numpy as np
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class IsaacSimResetState:
    base_position: tuple[float, float, float] | None = None
    base_orientation: tuple[float, float, float, float] | None = None
    joint_positions: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class IsaacSimModelMetadata:
    """Backend-independent description of one Isaac Sim articulation."""

    body_names: tuple[str, ...]
    dof_names: tuple[str, ...]
    joint_default_pos: torch.Tensor
    joint_pos_limits: torch.Tensor
    actuator_ctrl_range: torch.Tensor
    actuator_default_ctrl: torch.Tensor
    gravity: torch.Tensor
    body_prim_paths: tuple[str, ...]
    base_body_prim_path: str
    foot_body_prim_paths: tuple[str, ...]
    floor_prim_paths: tuple[str, ...]


class IsaacSimBackend(Protocol):
    metadata: IsaacSimModelMetadata

    @property
    def playback_env_index(self) -> int: ...

    def configure(
        self,
        *,
        num_envs: int,
        model_path: Path,
        ros_package_paths: Sequence[Mapping[str, str]],
        sim_dt: float,
        frame_skip: int,
        render_mode: str | None,
        env_spacing: float,
        robot_prim_path: str,
        base_body_prim_path: str | None,
        foot_body_prim_paths: Sequence[str],
        floor_prim_paths: Sequence[str],
        foot_contact_force_threshold: float,
        camera_prim_path: str | None,
        camera_resolution: Sequence[int],
        merge_fixed_joints: bool,
        allow_self_collision: bool,
        joint_stiffness: float | Mapping[str, float] | None,
        joint_damping: float | Mapping[str, float] | None,
        reset_state: IsaacSimResetState,
    ) -> None: ...

    def reset(self, env_ids: torch.Tensor | None = None) -> None: ...

    def step(self, action: torch.Tensor, frame_skip: int) -> None: ...

    def get_state(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> Mapping[str, torch.Tensor]: ...

    def render(self) -> np.ndarray | None: ...

    def close(self) -> None: ...
