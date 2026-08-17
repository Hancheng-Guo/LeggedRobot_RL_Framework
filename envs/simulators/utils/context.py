from typing import Any
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelContext:

    models: tuple[Any, ...]
    nq: int
    nv: int
    nu: int
    na: int

    body_names: tuple[str | None, ...]
    joint_names: tuple[str | None, ...]
    actuator_names: tuple[str | None, ...]
    # actuator_ctrlrange:

    # model.jnt_range          # joint position range
    # model.jnt_limited        # 是否启用 joint limit

# @dataclass(frozen=True)
# class ModelContext:

#     joint_names: tuple[str, ...]
#     joint_ids: torch.Tensor
#     joint_qpos_ids: torch.Tensor
#     joint_qvel_ids: torch.Tensor
#     joint_pos_limits: torch.Tensor

#     actuator_names: tuple[str, ...]
#     actuator_ids: torch.Tensor
#     actuator_ctrl_limits: torch.Tensor

#     body_names: tuple[str, ...]
#     body_ids: torch.Tensor

#     geom_names: tuple[str, ...]
#     geom_ids: torch.Tensor

#     foot_body_ids: torch.Tensor
#     foot_geom_ids: torch.Tensor

#     floor_geom_ids: torch.Tensor
#     fatal_body_ids: torch.Tensor