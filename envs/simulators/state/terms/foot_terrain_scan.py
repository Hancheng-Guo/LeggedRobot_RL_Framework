import torch
from math import cos, isfinite, radians, sqrt
from typing import Any

from .base import BaseSimulatorStateTerm
from .registry import register_simulator_state_term
from ..state import SimulatorState
from ...utils import ModelContext, GROUND_RAY_QUERY, ScanRayProvider
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
        simulator: ScanRayProvider,
        *,
        mount_names: list[str],
        reference_height: float,
        hit_mode: str = "first_surface",
        max_angle_deg: float = 90.,
        spacing: float = 0.12,
        layers: int = 4,
        max_distance: float = 2.0,
    ) -> None:

        super().__init__(name)

        self._model_context = model_context
        self._simulator = simulator

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

        self.max_distance = self._positive_number(max_distance, "max_distance")
        self.ref_height = self._positive_number(reference_height, "reference_height")
        if hit_mode not in ("first_surface", "visible_ground", "ground_only"):
            raise ValueError("Unknown foot_terrain_scan.hit_mode.")
        self.hit_mode = hit_mode
        angle = self._positive_number(max_angle_deg, "max_angle_deg")
        if angle > 180.:
            raise ValueError("foot_terrain_scan.max_angle_deg must be at most 180.")
        self._min_cos = cos(radians(angle))

        self._build_hex_layout(context, spacing, layers)
        self._resolve_mounts(mount_names, foot_count)

        self._measurement_contract = {
            "geometry": "base_reference_plane_v1",
            "projection": "gravity_up",
            "up_world": self._up.tolist(),
            "feet": tuple(
                model_context.geom_names[i]
                for i in model_context.foot_geom_ids.tolist()
            ),
            "mount_names": tuple(mount_names),
            "mount_positions": self._mount_positions.tolist(),
            "mount_axes": self._mount_axes.tolist(),
            "axial_coordinates": self.axial_coordinates,
            "spacing": spacing,
            "layers": layers,
            "reference_height": self.ref_height,
            "max_angle_deg": angle,
            "hit_mode": hit_mode,
            "max_distance": self.max_distance,
        }
        shape = (foot_count, self.point_count)
        self.output_shapes = {
            "foot_terrain_scan_heights": shape,
            "foot_terrain_scan_valid": shape,
        }


    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "scan": self._measurement_contract,
        }


    def playback_state_dict(
        self,
        env_index: int,
    ) -> dict[str, Any]:
        return self.checkpoint_state_dict()


    def _build_hex_layout(
        self,
        context: RuntimeContext,
        spacing: float,
        layers: int,
    ) -> None:

        spacing = self._positive_number(spacing, "spacing")
        if type(layers) is not int or layers < 0:
            raise ValueError(
                "foot_terrain_scan.layers must be a non-negative integer."
            )
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


    def _resolve_mounts(
        self,
        mount_names: list[str],
        foot_count: int,
    ) -> None:

        if (
            not isinstance(mount_names, (list, tuple))
            or len(mount_names) != foot_count
            or not all(isinstance(name, str) and name for name in mount_names)
            or len(set(mount_names)) != foot_count
        ):
            raise ValueError(
                "foot_terrain_scan.mount_names requires one unique mount per foot."
            )
        self._mount_positions, self._mount_axes = self._simulator.resolve_scan_mounts(
            tuple(mount_names)
        )
        for values in (self._mount_positions, self._mount_axes):
            if values.shape != (foot_count, 3) or not values.isfinite().all():
                raise ValueError("Invalid foot_terrain_scan mount geometry.")
        if not torch.allclose(
            self._mount_axes.norm(dim=-1), torch.ones_like(self._mount_axes[:, 0])
        ):
            raise ValueError("Expected unit mount axes.")


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


    def _get_base_pose(
        self,
        state: SimulatorState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_pos_world = state.qpos[:, self._model_context.base_pos_qpos_ids]
        quaternion = state.qpos[:, self._model_context.base_quat_qpos_ids]
        quaternion_norm = quaternion.norm(dim=-1, keepdim=True)
        base_pose_valid = (
            base_pos_world.isfinite().all(dim=-1)
            & quaternion_norm[:, 0].isfinite()
            & (quaternion_norm[:, 0] > torch.finfo(quaternion.dtype).eps)
        )
        quaternion = torch.where(
            base_pose_valid[:, None],
            quaternion / torch.where(base_pose_valid[:, None], quaternion_norm, 1.),
            0.,
        )
        w, x, y, z = quaternion.unbind(dim=-1)
        # Columns of the base-to-world rotation for a scalar-first quaternion.
        base_x_world = torch.stack(
            (1 - 2 * (y*y + z*z), 2 * (x*y + w*z), 2 * (x*z - w*y)), dim=-1,
        )
        base_y_world = torch.stack(
            (2 * (x*y - w*z), 1 - 2 * (x*x + z*z), 2 * (y*z + w*x)), dim=-1,
        )
        base_z_world = torch.stack(
            (2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2 * (x*x + y*y)), dim=-1,
        )
        base_to_world = torch.stack((base_x_world, base_y_world, base_z_world), dim=-1)
        base_pos_world = torch.where(base_pose_valid[:, None], base_pos_world, 0.)
        return base_pos_world, base_to_world, base_pose_valid


    def compute(
        self,
        state: SimulatorState,
        env_ids: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:

        foot_pos_world = state.geom_xpos[:, self._model_context.foot_geom_ids]
        if foot_pos_world.size(0) == 0:
            shape = (*foot_pos_world.shape[:2], self.point_count)
            return {
                "foot_terrain_scan_heights": foot_pos_world.new_empty(shape),
                "foot_terrain_scan_valid": foot_pos_world.new_empty(shape, dtype=torch.bool),
            }

        base_pos_world, base_to_world, base_pose_valid = self._get_base_pose(state)
        foot_pose_valid = foot_pos_world.isfinite().all(dim=-1) & base_pose_valid[:, None]
        foot_pos_world = torch.where(foot_pose_valid[..., None], foot_pos_world, 0.)

        # Project the foot onto base z = -reference_height, then aim from the mount.
        foot_pos_base = (foot_pos_world - base_pos_world[:, None]) @ base_to_world
        target_xy_base = foot_pos_base[:, :, None, :2] + self.offsets
        target_z_base = torch.full_like(target_xy_base[..., :1], -self.ref_height)
        target_points_base = torch.cat((target_xy_base, target_z_base), -1)

        mount_to_target = target_points_base - self._mount_positions[:, None]
        mount_to_target_norm = mount_to_target.norm(dim=-1, keepdim=True)
        ray_directions_base = mount_to_target / mount_to_target_norm.clamp_min(
            torch.finfo(mount_to_target.dtype).tiny
        )
        within_fov = (
            ray_directions_base * self._mount_axes[:, None]
        ).sum(-1) >= (self._min_cos - 1e-7)
        ray_enabled = foot_pose_valid[..., None] & (mount_to_target_norm[..., 0] > 0) & within_fov

        ray_directions_world = torch.einsum("nij,nfkj->nfki", base_to_world, ray_directions_base)
        ray_directions_world = torch.where(ray_enabled[..., None], ray_directions_world, 0.)
        ray_origins_world = base_pos_world[:, None] + self._mount_positions @ base_to_world.transpose(1, 2)
        hit_distance, ground_hit = self._simulator.query_surface_rays(
            origins=ray_origins_world,
            directions=ray_directions_world,
            max_distance=self.max_distance,
            env_ids=env_ids,
            include_robot=self.hit_mode != "ground_only",
        )
        valid_hit = ray_enabled & (
            hit_distance.isfinite() & (hit_distance >= 0.) & (hit_distance <= self.max_distance)
        )
        hit_points_world = ray_origins_world[:, :, None] + (
            torch.where(valid_hit, hit_distance, 0.)[..., None] * ray_directions_world
        )
        height_difference = ((foot_pos_world[:, :, None] - hit_points_world) * self._up).sum(-1)
        valid_measurement = valid_hit & height_difference.isfinite()

        return {
            "foot_terrain_scan_heights": torch.where(valid_measurement, height_difference, 0.),
            "foot_terrain_scan_valid": (
                valid_measurement & ground_hit if self.hit_mode == "visible_ground"
                else valid_measurement
            ),
        }
