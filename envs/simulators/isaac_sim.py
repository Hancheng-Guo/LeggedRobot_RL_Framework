from __future__ import annotations

import torch
import numpy as np
from math import isfinite
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from .base import BaseSimulator
from .friction import FrictionBackend
from .terrain import (
    StaticTerrainBackend, TerrainBackend, TerrainGeometry,
    TerrainPatch, TerrainRegionKey, TerrainSpawnGeometry,
)
from .isaac_sim_backend import DEFAULT_TERRAIN_LAYOUT, IsaacSimBackend, IsaacSimResetState
from .state import SimulatorState, SimulatorStateManager
from .utils import ModelContext, GROUND_RAY_QUERY
from utils import Component, RuntimeContext, update_attributes


def _default_backend_factory(context: RuntimeContext) -> IsaacSimBackend:
    try:
        from envs.simulators.isaac_sim_runtime import IsaacSimRuntime
    except ModuleNotFoundError as error:
        if error.name and (
            error.name == "isaacsim" or error.name.startswith("isaacsim.")
        ):
            raise ModuleNotFoundError(
                "Isaac Sim is an optional dependency. Install the packages "
                "listed in requirements/isaacsim.txt with the quadruped-rl "
                "interpreter before selecting simulator type 'isaac_sim'."
            ) from error
        raise
    return IsaacSimRuntime(context)


class IsaacSimSimulator(BaseSimulator):
    """Isaac Sim implementation of the framework simulator contract.

    Isaac Sim is imported lazily through :class:`IsaacSimRuntime`. This keeps
    normal MuJoCo runs and test collection independent of the optional Kit
    runtime.
    """

    SUPPORTS_CONCURRENT_INSTANCES = False

    def __init__(
        self,
        context: RuntimeContext,
        backend_factory: Callable[[RuntimeContext], IsaacSimBackend] | None = None,
    ) -> None:

        super().__init__(context)

        self._backend_factory = backend_factory or _default_backend_factory
        self._backend: IsaacSimBackend | None = None

        self.num_envs: int
        self.model_path: Path
        self.sim_dt: float
        self.frame_skip: int
        self.render_mode: str | None = None
        self.train_render_mode: str | None = None
        self.env_spacing: float = 2.0
        self.robot_prim_path: str = "/Robot"
        self.base_body_prim_path: str | None = None
        self.foot_body_prim_paths: tuple[str, ...] = ()
        self.floor_prim_paths: tuple[str, ...] = ()
        self.leg_joint_names: tuple[str, ...] = ()
        self.foot_contact_force_threshold: float = self.FOOT_CONTACT_FORCE_THRESHOLD
        self.camera_prim_path: str | None = None
        self.camera_resolution: tuple[int, int] = (640, 480)
        self.reset_state = IsaacSimResetState()
        self.terrain_geometry: TerrainGeometry | None = None
        self._friction_enabled = False
        self._terrain_layout = DEFAULT_TERRAIN_LAYOUT


    def config_update(
        self,
        component: Component,
        num_envs: int | None = None,
        model_path: Path | str | None = None,
        ros_package_paths: Sequence[Mapping[str, str]] | None = None,
        sim_dt: float | None = None,
        frame_skip: int | None = None,
        render_mode: str | None = None,
        train_render_mode: str | None = None,
        env_spacing: float | None = None,
        robot_prim_path: str | None = None,
        base_body_prim_path: str | None = None,
        foot_body_prim_paths: Sequence[str] | None = None,
        floor_prim_paths: Sequence[str] | None = None,
        leg_joint_names: Sequence[str] | None = None,
        foot_contact_force_threshold: float | None = None,
        camera_prim_path: str | None = None,
        camera_resolution: Sequence[int] | None = None,
        merge_fixed_joints: bool | None = None,
        allow_self_collision: bool | None = None,
        joint_stiffness: float | Mapping[str, float] | None = None,
        joint_damping: float | Mapping[str, float] | None = None,
        reset_state: Mapping[str, object] | None = None,
        terrain_geometry: TerrainGeometry | None = None,
        friction_enabled: bool | None = None,
        terrain_atlas: Mapping[str, object] | None = None,
        simulator_state_manager_config: Mapping[str, object] | None = None,
    ) -> None:

        state_manager_config = self._get_simulator_state_manager_config(
            simulator_state_manager_config,
            replace_config=model_path is not None or component.simulator is not None,
        )
        unavailable_capabilities = {}
        effective_floor_paths = (
            self.floor_prim_paths if floor_prim_paths is None
            else floor_prim_paths
        )
        if effective_floor_paths and terrain_geometry is None:
            unavailable_capabilities[GROUND_RAY_QUERY] = (
                "Isaac ground ray queries support the default ground plane and "
                "the shared terrain atlas. Custom floor_prim_paths are unsupported."
            )
        SimulatorStateManager.validate_config(
            unavailable_capabilities=unavailable_capabilities,
            **state_manager_config,
        )
        backend_configuration_changed = any(
            value is not None
            for value in (
                model_path,
                ros_package_paths,
                sim_dt,
                frame_skip,
                render_mode,
                train_render_mode,
                env_spacing,
                robot_prim_path,
                base_body_prim_path,
                foot_body_prim_paths,
                floor_prim_paths,
                foot_contact_force_threshold,
                camera_prim_path,
                camera_resolution,
                merge_fixed_joints,
                allow_self_collision,
                joint_stiffness,
                joint_damping,
                reset_state,
                terrain_atlas,
            )
        ) or terrain_geometry != self.terrain_geometry
        if (
            self._backend is not None
            and component.simulator is None
            and not backend_configuration_changed
            and (num_envs is None or num_envs == self.num_envs)
        ):
            backend = self._require_backend()
            if bool(friction_enabled) != self._friction_enabled:
                if friction_enabled:
                    self._require_friction_backend().prepare_friction()
                else:
                    disable_friction = getattr(backend, "disable_friction", None)
                    if disable_friction is None:
                        raise NotImplementedError(
                            "This Isaac runtime cannot release friction materials."
                        )
                    disable_friction()
                self._friction_enabled = bool(friction_enabled)
            if simulator_state_manager_config is not None:
                self._build_simulator_state_manager(state_manager_config)
            return

        if self._backend is not None:
            raise RuntimeError(
                "Reconfiguring the Isaac Sim backend would restart SimulationApp "
                "in the same process, which is not supported now."
            )

        if num_envs is not None and num_envs <= 0:
            raise ValueError("'num_envs' must be greater than 0.")
        if sim_dt is not None and sim_dt <= 0.0:
            raise ValueError("'sim_dt' must be greater than 0.")
        if frame_skip is not None and frame_skip <= 0:
            raise ValueError("'frame_skip' must be greater than 0.")
        if env_spacing is not None and env_spacing <= 0.0:
            raise ValueError("'env_spacing' must be greater than 0.")
        if isinstance(floor_prim_paths, (str, bytes)):
            raise TypeError("'floor_prim_paths' must be a sequence of paths.")
        if floor_prim_paths is not None:
            invalid_floor_paths = [
                path
                for path in floor_prim_paths
                if (
                    not path.startswith("/")
                    or path == "/"
                    or path.endswith("/")
                )
            ]
            if invalid_floor_paths:
                raise ValueError(
                    "Every floor Prim path must be absolute; invalid path(s): "
                    f"{invalid_floor_paths}."
                )
        if (
            foot_contact_force_threshold is not None
            and foot_contact_force_threshold < 0.0
        ):
            raise ValueError(
                "'foot_contact_force_threshold' must be non-negative."
            )
        if camera_resolution is not None:
            if (
                len(camera_resolution) != 2
                or any(int(size) <= 0 for size in camera_resolution)
            ):
                raise ValueError(
                    "'camera_resolution' must contain two positive integers."
                )
        parsed_reset_state = (
            None
            if reset_state is None
            else self._parse_reset_state(reset_state)
        )
        normalized_robot_path = (
            None
            if robot_prim_path is None
            else self._normalize_relative_prim_path(robot_prim_path)
        )
        normalized_base_body_path = (
            None
            if base_body_prim_path is None
            else self._normalize_relative_prim_path(base_body_prim_path)
        )
        normalized_foot_body_paths = (
            None
            if foot_body_prim_paths is None
            else tuple(
                self._normalize_relative_prim_path(path)
                for path in foot_body_prim_paths
            )
        )

        update_attributes(
            self,
            num_envs=num_envs,
            model_path=None if model_path is None else Path(model_path),
            sim_dt=sim_dt,
            frame_skip=frame_skip,
            env_spacing=env_spacing,
            robot_prim_path=normalized_robot_path,
            foot_body_prim_paths=normalized_foot_body_paths,
            floor_prim_paths=(
                tuple(floor_prim_paths)
                if floor_prim_paths is not None
                else None
            ),
            leg_joint_names=(
                tuple(leg_joint_names)
                if leg_joint_names is not None
                else None
            ),
            foot_contact_force_threshold=foot_contact_force_threshold,
            camera_resolution=(
                tuple(int(size) for size in camera_resolution)
                if camera_resolution is not None
                else None
            ),
        )

        if model_path is not None:
            self.base_body_prim_path = normalized_base_body_path
            self.camera_prim_path = camera_prim_path
            self.reset_state = parsed_reset_state or IsaacSimResetState()
        else:
            if normalized_base_body_path is not None:
                self.base_body_prim_path = normalized_base_body_path
            if camera_prim_path is not None:
                self.camera_prim_path = camera_prim_path
            if parsed_reset_state is not None:
                self.reset_state = parsed_reset_state

        for mode in (render_mode, train_render_mode):
            if mode is not None and mode not in self.SUPPORTED_RENDER_MODES:
                raise ValueError(f"Unsupported render mode: {mode!r}.")
        self.render_mode = render_mode
        self.train_render_mode = train_render_mode
        self.terrain_geometry = terrain_geometry
        self._friction_enabled = bool(friction_enabled)
        if terrain_atlas is not None:
            self._terrain_layout = self._parse_terrain_layout(terrain_atlas)

        self._validate_required_configuration()
        # Backend rebuilding is disabled until same-process SimulationApp restart
        # is verified with a fully configured Isaac environment.
        # if self._backend is not None:
        #     self._backend.close()
        self._backend = self._backend_factory(self.context)
        self._backend.configure(
            num_envs=self.num_envs,
            model_path=self.model_path.resolve(),
            ros_package_paths=tuple(
                {
                    package_name: str(Path(package_path).resolve())
                    for package_name, package_path in mapping.items()
                }
                for mapping in (ros_package_paths or ())
            ),
            sim_dt=self.sim_dt,
            frame_skip=self.frame_skip,
            render_mode=self.train_render_mode,
            env_spacing=self.env_spacing,
            robot_prim_path=self.robot_prim_path,
            base_body_prim_path=self.base_body_prim_path,
            foot_body_prim_paths=self.foot_body_prim_paths,
            floor_prim_paths=self.floor_prim_paths,
            foot_contact_force_threshold=self.foot_contact_force_threshold,
            camera_prim_path=self.camera_prim_path,
            camera_resolution=self.camera_resolution,
            merge_fixed_joints=bool(merge_fixed_joints),
            allow_self_collision=bool(allow_self_collision),
            joint_stiffness=joint_stiffness,
            joint_damping=joint_damping,
            reset_state=self.reset_state,
            terrain_geometry=terrain_geometry,
            terrain_layout=self._terrain_layout,
            **({"friction_enabled": True} if self._friction_enabled else {}),
        )
        self._build_model_context()
        self._build_simulator_state_manager(state_manager_config)


    @staticmethod
    def _parse_terrain_layout(
        config: Mapping[str, object]
    ) -> tuple[int, tuple[int, int], float]:

        if (
            not isinstance(config, Mapping)
            or set(config) - {"variants", "region_tiles", "border_width"}
        ):
            raise ValueError("Invalid Isaac terrain_atlas fields.")

        default_variants, default_tiles, default_width = DEFAULT_TERRAIN_LAYOUT
        variants = config.get("variants", default_variants)
        tiles = config.get("region_tiles", default_tiles)
        width = config.get("border_width", default_width)

        if type(variants) is not int or variants <= 0:
            raise ValueError("terrain_atlas.variants must be a positive integer.")
        if (
            not isinstance(tiles, (list, tuple))
            or len(tiles) != 2
            or any(type(value) is not int or value <= 0 for value in tiles)
        ):
            raise ValueError(
                "terrain_atlas.region_tiles must contain positive x and y tile counts."
            )
        if (
            isinstance(width, bool)
            or not isinstance(width, (int, float))
            or not isfinite(width)
            or width < 0
        ):
            raise ValueError(
                "terrain_atlas.border_width must be a finite non-negative length in metres."
            )

        return variants, (tiles[0], tiles[1]), float(width)


    @staticmethod
    def _normalize_relative_prim_path(path: str) -> str:
        normalized = "/" + path.strip("/")
        if normalized == "/":
            raise ValueError("Relative Prim paths must not be empty.")
        return normalized


    @staticmethod
    def _parse_reset_state(
        configuration: Mapping[str, object],
    ) -> IsaacSimResetState:
        
        supported_fields = {
            "base_position",
            "base_orientation",
            "joint_positions",
        }
        unknown_fields = configuration.keys() - supported_fields
        if unknown_fields:
            raise ValueError(
                f"Unsupported reset state field(s): {sorted(unknown_fields)}."
            )

        def _vector(name: str, size: int) -> tuple[float, ...] | None:
            values = configuration.get(name)
            if values is None:
                return None
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise TypeError(f"reset_state.{name} must be a sequence.")
            if len(values) != size:
                raise ValueError(
                    f"reset_state.{name} must contain {size} values."
                )
            return tuple(float(value) for value in values)

        base_position = _vector("base_position", 3)
        base_orientation = _vector("base_orientation", 4)
        joint_positions = configuration.get("joint_positions", {})

        if base_orientation is not None and not any(base_orientation):
            raise ValueError("reset_state.base_orientation must be non-zero.")
        if not isinstance(joint_positions, Mapping):
            raise TypeError("reset_state.joint_positions must be a mapping.")

        return IsaacSimResetState(
            base_position=cast(
                tuple[float, float, float] | None,
                base_position,
            ),
            base_orientation=cast(
                tuple[float, float, float, float] | None,
                base_orientation,
            ),
            joint_positions={
                str(name): float(position)
                for name, position in joint_positions.items()
            },
        )


    def _validate_required_configuration(self) -> None:

        required = ("num_envs", "model_path", "sim_dt", "frame_skip")
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise ValueError(
                f"Missing required Isaac Sim configuration: {missing}."
            )
        
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        
        if self.model_path.suffix.lower() not in {
            ".usd", ".usda", ".usdc", ".xml", ".urdf",
        }:
            raise ValueError(
                "Isaac Sim requires a USD model (.usd, .usda, or .usdc) "
                "or a robot description (.urdf or MJCF .xml)."
            )


    def _build_model_context(self) -> None:

        backend = self._require_backend()
        metadata = backend.metadata
        body_names = metadata.body_names
        
        try:
            base_id = metadata.body_prim_paths.index(
                metadata.base_body_prim_path
            )
        except ValueError as error:
            raise ValueError(
                "Unknown base body Prim path "
                f"{metadata.base_body_prim_path!r}."
            ) from error

        unknown_feet = (
            set(metadata.foot_body_prim_paths)
            - set(metadata.body_prim_paths)
        )
        if unknown_feet:
            raise ValueError(
                f"Unknown foot body Prim path(s): {sorted(unknown_feet)}."
            )

        missing_leg_names = set(self.leg_joint_names) - set(metadata.dof_names)
        if missing_leg_names:
            raise ValueError(f"Unknown actuated leg joint(s): {sorted(missing_leg_names)}.")

        dof_count = len(metadata.dof_names)
        floor_geom_names = tuple(
            path.rsplit("/", 1)[-1]
            for path in metadata.floor_prim_paths
        )
        geom_names = body_names + floor_geom_names

        self.model_context = ModelContext(
            nq=7 + dof_count,
            nv=6 + dof_count,
            nu=dof_count,
            na=0,
            body_names=body_names,
            gravity=self._tensor(metadata.gravity),
            base_id=base_id,
            base_pos_qpos_ids=self._index_tensor((0, 1, 2)),
            base_quat_qpos_ids=self._index_tensor((3, 4, 5, 6)),
            base_lin_vel_qvel_ids=self._index_tensor((0, 1, 2)),
            base_ang_vel_qvel_ids=self._index_tensor((3, 4, 5)),
            joint_qpos_ids=torch.arange(
                7,
                7 + dof_count,
                dtype=torch.long,
                device=self.context.device,
            ),
            leg_joint_qpos_ids=self._index_tensor(
                tuple(
                    7 + index for index, name in enumerate(metadata.dof_names)
                    if name in self.leg_joint_names
                )
            ),
            joint_qvel_ids=torch.arange(
                6,
                6 + dof_count,
                dtype=torch.long,
                device=self.context.device,
            ),
            joint_default_pos=self._tensor(metadata.joint_default_pos),
            joint_pos_limits=self._tensor(metadata.joint_pos_limits),
            actuator_ctrl_range=self._tensor(metadata.actuator_ctrl_range),
            actuator_default_ctrl=self._tensor(metadata.actuator_default_ctrl),
            geom_names=geom_names,
            geom_body_ids=self._index_tensor(
                tuple(range(len(body_names)))
                + (-1,) * len(floor_geom_names)
            ),
            foot_geom_ids=self._index_tensor(
                tuple(
                    metadata.body_prim_paths.index(path)
                    for path in metadata.foot_body_prim_paths
                )
            ),
            floor_geom_ids=torch.arange(
                len(body_names),
                len(body_names) + len(floor_geom_names),
                dtype=torch.long,
                device=self.context.device,
            ),
        )


    def resolve_scan_mounts(
        self,
        names: tuple[str, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._require_backend().resolve_scan_mounts(names)


    def query_surface_rays(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        max_distance: float,
        env_ids: torch.Tensor | None,
        *,
        include_robot: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        return self._require_backend().query_scene_rays(
            origins, directions, max_distance, env_ids,
            include_ground=True, include_robot=include_robot,
        )


    def query_ground_rays(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        max_distance: float,
        env_ids: torch.Tensor | None,
    ) -> torch.Tensor:

        query = self._require_backend().query_scene_rays
        ground, _ = query(
            origins, directions, max_distance, env_ids,
            include_ground=True, include_robot=False,
        )
        return ground


    def query_robot_rays(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        max_distance: float,
        env_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        distances, _ = self._require_backend().query_scene_rays(
            origins, directions, max_distance, env_ids,
            include_ground=False, include_robot=True,
        )
        return distances


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        self._require_backend().reset(env_ids)


    def get_terrain_spawn_geometry(self) -> TerrainSpawnGeometry:
        return self._require_terrain_backend().get_terrain_spawn_geometry()


    def get_terrain_layout(self) -> tuple[int, tuple[int, int], float]:
        return cast(
            StaticTerrainBackend, self._require_backend()
        ).get_terrain_layout()


    def get_terrain_patch(
        self,
        key: TerrainRegionKey,
    ) -> TerrainPatch:
        return cast(
            StaticTerrainBackend, self._require_backend()
        ).get_terrain_patch(key)


    def select_terrain(
        self,
        env_ids_tensor: torch.Tensor,
        keys: Sequence[TerrainRegionKey],
    ) -> None:
        cast(
            StaticTerrainBackend, self._require_backend()
        ).select_terrain(env_ids_tensor, keys)


    def get_ground_bounds(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.terrain_geometry is None:
            return None
        return self._require_terrain_backend().get_ground_bounds(env_ids)


    def get_ground_heights(
        self,
        xy: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.terrain_geometry is None:
            return super().get_ground_heights(xy, env_ids)
        return self._require_terrain_backend().get_ground_heights(xy, env_ids)


    def set_root_positions(
        self,
        env_ids_tensor: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        self._require_terrain_backend().set_root_positions(env_ids_tensor, positions)


    def capture_reset_state(
        self,
        env_ids_tensor: torch.Tensor,
    ) -> object:
        return self._require_terrain_backend().capture_reset_state(env_ids_tensor)


    def restore_reset_state(
        self,
        state: object,
    ) -> None:
        self._require_terrain_backend().restore_reset_state(state)


    def _require_terrain_backend(self) -> TerrainBackend:
        if self.terrain_geometry is None:
            raise ValueError("Isaac terrain collision geometry is not configured.")
        return cast(TerrainBackend, self._require_backend())


    def prepare_friction(self) -> None:
        self._require_friction_backend().prepare_friction()


    def capture_friction(
        self,
        env_ids: Sequence[int],
    ) -> object:
        return self._require_friction_backend().capture_friction(env_ids)


    def apply_friction(
        self,
        env_ids: Sequence[int],
        factors: Sequence[float],
    ) -> None:
        self._require_friction_backend().apply_friction(env_ids, factors)


    def restore_friction(
        self,
        snapshot: object,
    ) -> None:
        self._require_friction_backend().restore_friction(snapshot)


    def _require_friction_backend(self) -> FrictionBackend:
        backend = self._require_backend()
        if not isinstance(backend, FrictionBackend):
            raise NotImplementedError(
                "This Isaac runtime does not support friction randomization."
            )
        return backend


    def step(
        self,
        action: torch.Tensor
    ) -> None:
        expected_shape = (self.num_envs, self.model_context.nu)
        if tuple(action.shape) != expected_shape:
            raise ValueError(
                f"Action must have shape {expected_shape}, got {tuple(action.shape)}."
            )
        self._require_backend().step(action, self.frame_skip)


    def get_state(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> SimulatorState:
        return self.simulator_state_manager.assemble(
            env_ids, **self._require_backend().get_state(env_ids),
        )


    def render(self) -> np.ndarray | None:
        return self._require_backend().render()


    def prepare_playback(
        self,
        render_mode: str | None = None
    ) -> None:
        mode = render_mode if render_mode is not None else self.render_mode
        if mode is None:
            return
        backend = self._require_backend()
        if mode == backend.render_mode:
            return
        backend.activate_playback(mode)


    def terminate_playback(self) -> None:
        backend = self._require_backend()
        backend.terminate_playback()


    @property
    def active_render_mode(self) -> str | None:
        return self._require_backend().render_mode


    @property
    def playback_env_index(self) -> int:
        return self._require_backend().playback_env_index


    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None


    def _require_backend(self) -> IsaacSimBackend:
        if self._backend is None:
            raise RuntimeError("Isaac Sim backend has not been configured.")
        return self._backend
