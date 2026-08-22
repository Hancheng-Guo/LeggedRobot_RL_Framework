import torch
from typing import Any
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelContext:

    nq: int
    nv: int
    nu: int
    na: int

    gravity: torch.Tensor

    # base_names: tuple[str, ...]
    # base_ids: torch.Tensor
    base_pos_qpos_ids: torch.Tensor
    base_quat_qpos_ids: torch.Tensor
    base_ang_vel_qvel_ids: torch.Tensor

    body_names: tuple[str | None, ...]

    # joint_names: tuple[str, ...]
    # joint_ids: torch.Tensor
    joint_qpos_ids: torch.Tensor
    joint_qvel_ids: torch.Tensor
    # joint_pos_limits: torch.Tensor

    # actuator_names: tuple[str, ...]
    # actuator_ids: torch.Tensor
    actuator_ctrl_range: torch.Tensor

    geom_names: tuple[str | None, ...]
    geom_body_ids: torch.Tensor
    # geom_ids: torch.Tensor

    # foot_body_ids: torch.Tensor
    # foot_geom_ids: torch.Tensor

    # floor_geom_ids: torch.Tensor
    # fatal_body_ids: torch.Tensor
