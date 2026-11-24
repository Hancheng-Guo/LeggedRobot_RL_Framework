import torch
from math import isfinite, sqrt

from .base import BaseSimulatorStateTerm
from .registry import register_simulator_state_term
from ..state import SimulatorState
from ...utils import ModelContext, GROUND_RAY_QUERY, GroundRayQueryProvider
from utils import RuntimeContext


@register_simulator_state_term
class FootTerrainScan(BaseSimulatorStateTerm):

    required_capabilities = frozenset((GROUND_RAY_QUERY,))
    MAX_POINTS_PER_FOOT = 4096


    def __init__(
        self,
        name: str,
        context: RuntimeContext,
        model_context: ModelContext,
        simulator: GroundRayQueryProvider,
        *,
        spacing: float = 0.12,
        layers: int = 4,
        max_distance: float = 2.0,
    ) -> None:

        super().__init__(name)

        self._model_context = model_context
        self._simulator = simulator

        spacing = self._positive_number(spacing, "spacing")
        if type(layers) is not int or layers < 0:
            raise ValueError(
                "foot_terrain_scan.layers must be a non-negative integer."
            )
        self.max_distance = self._positive_number(max_distance, "max_distance")
        self.point_count = 1 + 3 * layers * (layers + 1)
        if self.point_count > self.MAX_POINTS_PER_FOOT:
            raise ValueError(
                "foot_terrain_scan.layers exceeds the per-foot point budget."
            )

        points = []
        for r in range(-layers, layers + 1):
            for q in range(-layers, layers + 1):
                layer = max(abs(q), abs(r), abs(q + r))
                if layer <= layers:
                    points.append((layer, r, q))
        self.axial_coordinates = tuple((q, r) for _, r, q in sorted(points))
        self.point_indices = {
            coordinate: index
            for index, coordinate in enumerate(self.axial_coordinates)
        }
        self.offsets = torch.tensor(
            [(spacing * (q + r / 2), spacing * sqrt(3) * r / 2)
             for q, r in self.axial_coordinates],
            dtype=context.dtype, device=context.device,
        )
        if not bool(self.offsets.isfinite().all()):
            raise ValueError(
                "foot_terrain_scan.spacing exceeds the simulator dtype range."
            )

        foot_count = model_context.foot_geom_ids.numel()
        if foot_count == 0:
            raise ValueError(
                "foot_terrain_scan requires at least one configured foot."
            )
        gravity_norm = model_context.gravity.norm()
        if not bool(gravity_norm.isfinite() & (gravity_norm > 0.)):
            raise ValueError(
                "foot_terrain_scan requires finite, nonzero gravity."
            )
        self._up = -model_context.gravity / gravity_norm
        shape = (foot_count, self.point_count)
        self.output_shapes = {
            "foot_terrain_scan_heights": shape,
            "foot_terrain_scan_valid": shape,
        }


    @staticmethod
    def _positive_number(
        value: object,
        name: str,
    ) -> float:

        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(
                f"foot_terrain_scan.{name} must be positive and finite."
            )
        try:
            result = float(value)
        except OverflowError:
            raise ValueError(
                f"foot_terrain_scan.{name} must be positive and finite."
            ) from None
        if not isfinite(result) or result <= 0.:
            raise ValueError(
                f"foot_terrain_scan.{name} must be positive and finite."
            )
        return result


    def compute(
        self,
        state: SimulatorState,
        env_ids: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:

        feet = state.geom_xpos[:, self._model_context.foot_geom_ids]
        if feet.size(0) == 0:
            shape = (*feet.shape[:2], self.point_count)
            return {
                "foot_terrain_scan_heights": feet.new_empty(shape),
                "foot_terrain_scan_valid": feet.new_empty(shape, dtype=torch.bool),
            }

        base_position = state.qpos[:, self._model_context.base_pos_qpos_ids]
        quaternion = state.qpos[:, self._model_context.base_quat_qpos_ids]
        norm = quaternion.norm(dim=-1, keepdim=True)
        valid_pose = (
            base_position.isfinite().all(dim=-1)
            & norm[:, 0].isfinite()
            & (norm[:, 0] > torch.finfo(quaternion.dtype).eps)
        )
        quaternion = torch.where(
            valid_pose[:, None],
            quaternion / torch.where(valid_pose[:, None], norm, 1.),
            0.,
        )
        w, x, y, z = quaternion.unbind(dim=-1)
        # Columns of the base-to-world rotation for a scalar-first quaternion.
        base_x = torch.stack(
            (1 - 2 * (y*y + z*z), 2 * (x*y + w*z), 2 * (x*z - w*y)), dim=-1,
        )
        base_y = torch.stack(
            (2 * (x*y - w*z), 1 - 2 * (x*x + z*z), 2 * (y*z + w*x)), dim=-1,
        )
        base_up = torch.stack(
            (2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2 * (x*x + y*y)), dim=-1,
        )
        base_up = torch.where(valid_pose[:, None], base_up, 0.)

        valid_foot = feet.isfinite().all(dim=-1) & valid_pose[:, None]
        feet = torch.where(valid_foot[..., None], feet, 0.)
        base_position = torch.where(valid_pose[:, None], base_position, 0.)
        # Project each foot onto base z=0 before adding the in-plane offsets.
        foot_base_z = (
            (feet - base_position[:, None]) * base_up[:, None]
        ).sum(dim=-1)
        feet_on_base_plane = feet - foot_base_z[..., None] * base_up[:, None]
        dx, dy = self.offsets.unbind(dim=-1)
        offsets = (
            dx[None, :, None] * base_x[:, None]
            + dy[None, :, None] * base_y[:, None]
        )
        origins = feet_on_base_plane[:, :, None] + offsets[:, None]
        distance = self._simulator.query_ground_rays(
            origins, -base_up, self.max_distance, env_ids
        )
        if distance.shape != origins.shape[:-1]:
            raise ValueError(
                "Ground ray query returned an unexpected distance shape."
            )
        valid = (
            valid_foot[..., None] & distance.isfinite()
            & (distance >= 0.) & (distance <= self.max_distance)
        )
        dz = -torch.where(valid, distance, 0.) - foot_base_z[..., None]
        height = (
            (offsets * self._up).sum(dim=-1)[:, None]
            + dz * (base_up * self._up).sum(dim=-1)[:, None, None]
        )
        valid = valid & height.isfinite()
        return {
            "foot_terrain_scan_heights": torch.where(valid, height, 0.),
            "foot_terrain_scan_valid": valid,
        }
