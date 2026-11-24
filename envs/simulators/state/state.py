import torch
from dataclasses import dataclass, field, fields
from collections.abc import Iterator


@dataclass
class SimulatorState:
    """Normalized simulator state shared by every backend.

    Tensor storage may be owned and reused by the simulator. Call ``clone``
    before retaining a state beyond a later read from the same buffer lane.
    Core fields use typed attributes and configured terms publish named tensors.
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
    foot_ground_heights: torch.Tensor | None = None  # [env, num_feet], ModelContext foot order.
    ground_valid: torch.Tensor | None = None  # [env], root/foot ground samples are defined.
    terrain_in_bounds: torch.Tensor | None = None  # [env], root and feet inside the random patch.
    term_values: dict[str, torch.Tensor | None] = field(default_factory=dict, init=False)


    def __getattr__(
        self,
        name: str,
    ) -> torch.Tensor | None:
        """Fall back to term outputs when normal attribute lookup fails."""
        try:
            return vars(self)["term_values"][name]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__} has no attribute '{name}'."
            ) from None


    def as_dict(self) -> dict[str, torch.Tensor]:
        """Export available tensors, omitting fields whose value is None."""
        values = dict(self._core_items())
        attributes = vars(self)
        values.update({
            name: value for name, value in self.term_values.items()
            if value is not None and name not in attributes
        })
        return values


    def _core_items(self) -> Iterator[tuple[str, torch.Tensor]]:
        for field in fields(self):
            if field.name == "term_values":
                continue
            value = getattr(self, field.name)
            if value is not None:
                yield field.name, value


    def clone(self) -> "SimulatorState":
        state = type(self)(**{
            name: value.clone()
            for name, value in self._core_items()
        })
        state.term_values = {
            name: (None if value is None else value.clone())
            for name, value in self.term_values.items()
        }
        return state


    def select(
        self,
        env_ids: torch.Tensor,
    ) -> "SimulatorState":
        state = type(self)(**{
            name: value[env_ids]
            for name, value in self._core_items()
        })
        state.term_values = {
            name: (None if value is None else value[env_ids])
            for name, value in self.term_values.items()
        }
        return state
