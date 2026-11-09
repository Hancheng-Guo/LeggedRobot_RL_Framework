from dataclasses import dataclass, fields
from collections.abc import Iterator

import torch


@dataclass
class SimulatorState:
    """Normalized simulator state shared by every backend.

    Tensor storage may be owned and reused by the simulator. Call ``clone``
    before retaining a state beyond a later read from the same buffer lane.
    Fields are intentionally exposed only through typed attribute access.
    """

    qpos: torch.Tensor
    qvel: torch.Tensor
    qacc: torch.Tensor
    ctrl: torch.Tensor
    geom_xpos: torch.Tensor
    geom_xvel: torch.Tensor
    actuator_force: torch.Tensor
    base_lin_vel_body: torch.Tensor
    base_ang_vel_body: torch.Tensor
    contact_geom_ids: torch.Tensor
    contact_forces: torch.Tensor
    foot_ground_contact: torch.Tensor  # [num_envs, num_feet]
    foot_contact_normal_force: torch.Tensor  # [num_envs, num_feet]
    base_ground_height: torch.Tensor | None = None  # [env], world ground z below the root.
    foot_ground_heights: torch.Tensor | None = None  # [env, foot], ModelContext foot order.
    ground_valid: torch.Tensor | None = None  # [env], root/foot ground samples are defined.
    terrain_in_bounds: torch.Tensor | None = None  # [env], root and feet inside the random patch.

    def as_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._items())

    def _items(self) -> Iterator[tuple[str, torch.Tensor]]:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                yield field.name, value

    def clone(self) -> "SimulatorState":
        return type(self)(**{
            name: value.clone()
            for name, value in self._items()
        })

    def select(self, env_ids: torch.Tensor) -> "SimulatorState":
        return type(self)(**{
            name: value[env_ids]
            for name, value in self._items()
        })
