from dataclasses import dataclass, fields

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
    foot_ground_contact: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }

    def _items(self):
        return (
            (field.name, getattr(self, field.name))
            for field in fields(self)
        )

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
