import torch
from typing import Any
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelContext:

    nq: int
    nv: int
    nu: int
    na: int

#     joint_names: tuple[str, ...]
#     joint_ids: torch.Tensor
#     joint_qpos_ids: torch.Tensor
#     joint_qvel_ids: torch.Tensor
#     joint_pos_limits: torch.Tensor

#     actuator_names: tuple[str, ...]
#     actuator_ids: torch.Tensor
    actuator_ctrl_range: torch.Tensor

#     body_names: tuple[str, ...]
#     body_ids: torch.Tensor

#     geom_names: tuple[str, ...]
#     geom_ids: torch.Tensor

#     foot_body_ids: torch.Tensor
#     foot_geom_ids: torch.Tensor

#     floor_geom_ids: torch.Tensor
#     fatal_body_ids: torch.Tensor