from __future__ import annotations

import torch
import numpy as np
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from app.utils.context import RuntimeContext
from envs.simulators.base import BaseSimulator
from envs.simulators.isaac_sim_backend import IsaacSimBackend, IsaacSimResetState
from envs.simulators.utils.context import ModelContext
from utils.component import Component
from utils.param import update_attributes


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
        self.env_spacing: float = 2.0
        self.robot_prim_path: str = "/Robot"
        self.base_body_prim_path: str | None = None
        self.foot_body_prim_paths: tuple[str, ...] = ()
        self.floor_prim_paths: tuple[str, ...] = ()
        self.foot_contact_force_threshold: float = 15.0
        self.camera_prim_path: str | None = None
        self.camera_resolution: tuple[int, int] = (640, 480)
        self.reset_state = IsaacSimResetState()


    def config_update(
        self,
        component: Component,
        num_envs: int | None = None,
        model_path: Path | str | None = None,
        ros_package_paths: Sequence[Mapping[str, str]] | None = None,
        sim_dt: float | None = None,
        frame_skip: int | None = None,
        render_mode: str | None = None,
        env_spacing: float | None = None,
        robot_prim_path: str | None = None,
        base_body_prim_path: str | None = None,
        foot_body_prim_paths: Sequence[str] | None = None,
        floor_prim_paths: Sequence[str] | None = None,
        foot_contact_force_threshold: float | None = None,
        camera_prim_path: str | None = None,
        camera_resolution: Sequence[int] | None = None,
        merge_fixed_joints: bool | None = None,
        allow_self_collision: bool | None = None,
        joint_stiffness: float | Mapping[str, float] | None = None,
        joint_damping: float | Mapping[str, float] | None = None,
        reset_state: Mapping[str, object] | None = None,
    ) -> None:

        backend_configuration_changed = any(
            value is not None
            for value in (
                model_path,
                ros_package_paths,
                sim_dt,
                frame_skip,
                render_mode,
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
            )
        )
        if (
            self._backend is not None
            and component.simulator is None
            and not backend_configuration_changed
            and (num_envs is None or num_envs == self.num_envs)
        ):
            return

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
        if render_mode is not None:
            if render_mode not in self.SUPPORTED_RENDER_MODES:
                raise ValueError(f"Unsupported render mode: {render_mode!r}.")
            self.render_mode = render_mode

        self._validate_required_configuration()
        if self._backend is not None:
            self._backend.close()
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
            render_mode=self.render_mode,
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
        )
        self._build_model_context()


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


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:

        self._require_backend().reset(env_ids)


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
    ) -> dict[str, torch.Tensor]:
        
        state = dict(self._require_backend().get_state(env_ids))
        required = {
            "qpos",
            "qvel",
            "qacc",
            "ctrl",
            "geom_xpos",
            "geom_xvel",
            "actuator_force",
            "base_lin_vel_body",
            "base_ang_vel_body",
            "contact_geom_ids",
            "contact_forces",
            "foot_ground_contact",
        }
        missing = required - state.keys()
        if missing:
            raise RuntimeError(
                f"Isaac Sim backend omitted state field(s): {sorted(missing)}."
            )
        return state


    def render(self) -> np.ndarray | None:
        return self._require_backend().render()


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
