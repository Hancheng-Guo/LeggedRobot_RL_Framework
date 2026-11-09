import os


if os.name == "nt":
    os.environ.setdefault("MUJOCO_GL", "wgl")


import mujoco
import torch
import numpy as np
from collections.abc import Sequence
from mujoco import viewer
from dataclasses import dataclass, fields
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any, Generic, TypeVar
from copy import copy

from .base import BaseSimulator
from .terrain import TerrainGeometry, TerrainPatch, TerrainSpawnGeometry
from .mujoco_terrain import MujocoTerrain
from .utils import ModelContext, SimulatorState
from utils import Component, RuntimeContext, update_attributes


BufferType = TypeVar("BufferType", np.ndarray, torch.Tensor)
_TERRAIN_UNCHANGED = object()


@dataclass
class _MujocoResetSnapshot:
    env_ids_tuple: tuple[int, ...]
    models: tuple[Any, ...]
    datas: tuple[Any, ...]
    hfields: tuple[np.ndarray, ...]
    grids: tuple[np.ndarray, ...]


@dataclass
class MujocoFixedStateBuffers(Generic[BufferType]):
    qpos: BufferType
    qvel: BufferType
    qacc: BufferType
    ctrl: BufferType
    geom_xpos: BufferType
    actuator_force: BufferType
    base_lin_vel_body: BufferType
    base_ang_vel_body: BufferType
    geom_xvel: BufferType

    def iter_fields(self):
        return (
            (field.name, getattr(self, field.name))
            for field in fields(self)
        )


class MujocoSimulator(BaseSimulator):

    def __init__(
        self,
        context: RuntimeContext,
    ) -> None:

        super().__init__(context)

        self.num_envs: int
        self.model_path: Path
        self.sim_dt: float
        self.frame_skip: int
        self.step_workers: int = 1
        self.foot_geom_names: tuple[str, ...]
        self.floor_geom_names: tuple[str, ...]
        self.leg_joint_names: tuple[str, ...] = ()
        self.foot_contact_force_threshold: float = self.FOOT_CONTACT_FORCE_THRESHOLD
        self.render_mode: str | None = None
        self.reset_keyframe: str | None = None
        self.reset_keyframe_id: int = -1
        self._terrain: MujocoTerrain | None = None
        self.terrain_geometry: TerrainGeometry | None = None
        self._terrain_render_dirty = False

        self.models: list[mujoco.MjModel] = []  # pyright: ignore[reportAttributeAccessIssue]
        self.datas: list[mujoco.MjData] = []    # pyright: ignore[reportAttributeAccessIssue]
        self.viewer = None
        self.renderer = None
        self._render_camera = None
        self._step_executor: ThreadPoolExecutor | None = None
        self._full_state_numpy_buffers: MujocoFixedStateBuffers[np.ndarray] | None = None
        self._full_state_tensor_buffers: MujocoFixedStateBuffers[torch.Tensor] | None = None
        self._reset_state_numpy_buffers: MujocoFixedStateBuffers[np.ndarray] | None = None
        self._reset_state_tensor_buffers: MujocoFixedStateBuffers[torch.Tensor] | None = None
        self._state_staging_tensors: dict[int, torch.Tensor] = {}
        self._playback_render_mode: str | None = None
        self._playback_render_mode_changed = False


    def config_update(
        self,
        component: Component,
        num_envs: int | None = None,
        model_path: Path | str | None = None,
        sim_dt: float | None = None,
        frame_skip: int | None = None,
        step_workers: int | None = None,
        render_mode: str | None = None,
        foot_geom_names: list[str] | tuple[str, ...] | None = None,
        floor_geom_names: list[str] | tuple[str, ...] | None = None,
        leg_joint_names: list[str] | tuple[str, ...] | None = None,
        foot_contact_force_threshold: float | None = None,
        reset_keyframe: str | None = None,
        terrain_geometry: TerrainGeometry | None | object = _TERRAIN_UNCHANGED,
    ) -> None:

        if terrain_geometry is not _TERRAIN_UNCHANGED:
            if terrain_geometry is not None and not isinstance(terrain_geometry, TerrainGeometry):
                raise TypeError("terrain_geometry must be a TerrainGeometry or None.")
            self.terrain_geometry = terrain_geometry
        elif model_path is not None:
            self.terrain_geometry = None

        if num_envs is not None and num_envs <= 0:
            raise ValueError(
                "'num_envs' should be a int greater than 0."
            )
        if step_workers is not None and step_workers <= 0:
            raise ValueError("'step_workers' must be greater than 0.")
        if (
            foot_contact_force_threshold is not None
            and foot_contact_force_threshold < 0.0
        ):
            raise ValueError(
                "'foot_contact_force_threshold' must be non-negative."
            )
        
        update_attributes(
            self,
            num_envs=num_envs,
            model_path=None if model_path is None else Path(model_path),
            sim_dt=sim_dt,
            frame_skip=frame_skip,
            step_workers=step_workers,
            foot_contact_force_threshold=foot_contact_force_threshold,
            foot_geom_names=(
                tuple(foot_geom_names)
                if foot_geom_names is not None
                else None
            ),
            floor_geom_names=(
                tuple(floor_geom_names)
                if floor_geom_names is not None
                else None
            ),
            leg_joint_names=(
                tuple(leg_joint_names)
                if leg_joint_names is not None
                else None
            ),
        )

        self._shutdown_step_executor()
        if self.step_workers > 1:
            self._step_executor = ThreadPoolExecutor(
                max_workers=min(self.step_workers, self.num_envs),
                thread_name_prefix="mujoco-step",
            )

        # A simulator config that supplies a model path owns the complete reset
        # configuration. Incremental updates (for example, changing num_envs)
        # keep the previously selected keyframe.
        if model_path is not None:
            self.reset_keyframe = reset_keyframe
        elif reset_keyframe is not None:
            self.reset_keyframe = reset_keyframe

        if render_mode is not None:
            if render_mode not in self.SUPPORTED_RENDER_MODES:
                raise ValueError(f"Unsupported render mode: {render_mode!r}.")
            self.render_mode = render_mode

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
            self._render_camera = None

        self._terrain_render_dirty = False
        if self.terrain_geometry is None:
            self._terrain = None
            self.models = [
                mujoco.MjModel.from_xml_path(str(self.model_path))  # pyright: ignore[reportAttributeAccessIssue]
                for _ in range(self.num_envs)
            ]
        else:
            self._terrain = MujocoTerrain(
                self.terrain_geometry,
                self.num_envs,
            )
            self.models = self._terrain.build_models(self.model_path, self.floor_geom_names)
        for model in self.models:
            model.opt.timestep = self.sim_dt

        self.datas = [
            mujoco.MjData(model)    # pyright: ignore[reportAttributeAccessIssue]
            for model in self.models
        ]
        self.reset_keyframe_id = -1
        if self.reset_keyframe is not None:
            reset_keyframe_id = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
                self.models[0],
                mujoco.mjtObj.mjOBJ_KEY,            # pyright: ignore[reportAttributeAccessIssue]
                self.reset_keyframe,
            )
            if reset_keyframe_id == -1:
                raise ValueError(
                    f"Keyframe '{self.reset_keyframe}' was not found."
                )
            self.reset_keyframe_id = reset_keyframe_id

        self._build_model_context()
        self._initialize_state_buffers()


    def _build_model_context(self) -> None:

        model = self.models[0]
        if model.na != 0:
            raise NotImplementedError(
                "MuJoCo actuator activation states are not supported; "
                f"expected model.na == 0, got {model.na}."
            )
        
        body_names = self._object_names(
            model,
            mujoco.mjtObj.mjOBJ_BODY,  # pyright: ignore[reportAttributeAccessIssue]
            model.nbody,
        )
        geom_names = self._object_names(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,  # pyright: ignore[reportAttributeAccessIssue]
            model.ngeom,
        )
        base_id, base_qpos_adr, base_qvel_adr = (
            self._find_base_joint_info(model)
        )
        actuator_joint_ids = self._find_actuated_joint_ids(model)
        actuator_default_ctrl = (
            np.zeros(model.nu, dtype=model.qpos0.dtype)
            if self.reset_keyframe_id == -1
            else model.key_ctrl[self.reset_keyframe_id, :]
        )
        joint_qpos_ids = model.jnt_qposadr[actuator_joint_ids]
        actuator_joint_names = self._object_names(
            model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt  # pyright: ignore[reportAttributeAccessIssue]
        )
        actuated_names = tuple(actuator_joint_names[int(i)] for i in actuator_joint_ids)
        missing_leg_names = set(self.leg_joint_names) - set(actuated_names)
        if missing_leg_names:
            raise ValueError(f"Unknown actuated leg joint(s): {sorted(missing_leg_names)}.")
        leg_joint_qpos_ids = joint_qpos_ids[
            [name in self.leg_joint_names for name in actuated_names]
        ]
        joint_default_pos = (
            model.qpos0[joint_qpos_ids]
            if self.reset_keyframe_id == -1
            else model.key_qpos[self.reset_keyframe_id, joint_qpos_ids]
        )
        joint_pos_limits = self._joint_pos_limits(
            model,
            actuator_joint_ids,
        )

        self.model_context = ModelContext(

            nq = model.nq,
            nv = model.nv,
            nu = model.nu,
            na = model.na,

            body_names=body_names,
            gravity=self._tensor(model.opt.gravity),

            # base_names=names(mujoco.mjtObj.mjOBJ_BODY, model.nbody),
            base_id=int(model.jnt_bodyid[base_id]),
            base_pos_qpos_ids=self._index_tensor(
                np.arange(base_qpos_adr, base_qpos_adr + 3)
            ),
            base_quat_qpos_ids=self._index_tensor(
                np.arange(base_qpos_adr + 3, base_qpos_adr + 7)
            ),
            base_lin_vel_qvel_ids=self._index_tensor(
                np.arange(base_qvel_adr, base_qvel_adr + 3)
            ),
            base_ang_vel_qvel_ids=self._index_tensor(
                np.arange(base_qvel_adr + 3, base_qvel_adr + 6)
            ),
            
            # joint_names=names(mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
            joint_qpos_ids=self._index_tensor(joint_qpos_ids),
            joint_qvel_ids=self._index_tensor(
                model.jnt_dofadr[actuator_joint_ids]
            ),
            joint_default_pos=self._tensor(joint_default_pos),
            joint_pos_limits=self._tensor(joint_pos_limits),
            leg_joint_qpos_ids=self._index_tensor(leg_joint_qpos_ids),

            # actuator_names=names(mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
            actuator_ctrl_range = self._tensor(model.actuator_ctrlrange),
            actuator_default_ctrl = self._tensor(actuator_default_ctrl),

            geom_names=geom_names,
            geom_body_ids=self._index_tensor(model.geom_bodyid),
            foot_geom_ids=self._geom_ids_by_name(
                geom_names,
                requested_names=self.foot_geom_names,
                default_suffix="foot",
            ),
            floor_geom_ids=self._geom_ids_by_name(
                geom_names,
                requested_names=(
                    self.floor_geom_names
                    if self._terrain is None
                    else self.floor_geom_names + self._terrain.peripheral_geom_names
                ),
            ),
        )


    def _object_names(
        self,
        model,
        object_type,
        count: int,
    ) -> tuple[str | None, ...]:
        
        return tuple(
            mujoco.mj_id2name(model, object_type, index)  # pyright: ignore[reportAttributeAccessIssue]
            for index in range(count)
        )


    def _geom_ids_by_name(
        self,
        geom_names: tuple[str | None, ...],
        requested_names: tuple[str, ...],
        default_suffix: str | None = None,
    ) -> torch.Tensor:
        
        geom_name_to_id = {
            name: geom_id
            for geom_id, name in enumerate(geom_names)
            if name is not None
        }

        if requested_names:
            unknown_names = set(requested_names) - geom_name_to_id.keys()
            if unknown_names:
                raise ValueError(
                    f"Unknown geom name(s): {sorted(unknown_names)}."
                )
            ids = [geom_name_to_id[name] for name in requested_names]
        elif default_suffix is not None:
            ids = [
                geom_id
                for geom_id, name in enumerate(geom_names)
                if name is not None and name.lower().endswith(default_suffix)
            ]
        else:
            ids = []

        return self._index_tensor(np.asarray(ids, dtype=np.int64))


    def _find_base_joint_info(
        self,
        model,
    ) -> tuple[int, int, int]:
        
        free_joint_ids = np.flatnonzero(
            model.jnt_type == mujoco.mjtJoint.mjJNT_FREE    # pyright: ignore[reportAttributeAccessIssue]
        )
        if free_joint_ids.size != 1:
            raise ValueError(
                "The model must contain exactly one free joint to build "
                "base observations."
            )
        base_joint_id = int(free_joint_ids[0])
        base_qpos_adr = int(model.jnt_qposadr[base_joint_id])
        base_qvel_adr = int(model.jnt_dofadr[base_joint_id])

        return base_joint_id, base_qpos_adr, base_qvel_adr


    def _find_actuated_joint_ids(
        self,
        model,
    ) -> np.ndarray:
        
        joint_transmission_types = {
            int(mujoco.mjtTrn.mjTRN_JOINT),         # pyright: ignore[reportAttributeAccessIssue]
            int(mujoco.mjtTrn.mjTRN_JOINTINPARENT), # pyright: ignore[reportAttributeAccessIssue]
        }
        if any(
            int(transmission_type) not in joint_transmission_types
            for transmission_type in model.actuator_trntype
        ):
            raise ValueError(
                "Every actuator must use a joint transmission to build "
                "joint observations."
            )

        # explain the mapping between actuators and joints
        actuator_joint_ids = model.actuator_trnid[:, 0].astype(
            np.int64,
            copy=False,
        )
        if np.any(actuator_joint_ids < 0):
            raise ValueError(
                "Every actuator must reference a joint to build joint observations."
            )   # '-1' means that the actuator does not reference a joint correctly

        # check all actuated joints are hinge or slide joints, which are one-dimensional joints
        actuator_joint_types = model.jnt_type[actuator_joint_ids]
        scalar_joint_types = {
            int(mujoco.mjtJoint.mjJNT_HINGE),   # pyright: ignore[reportAttributeAccessIssue]
            int(mujoco.mjtJoint.mjJNT_SLIDE),   # pyright: ignore[reportAttributeAccessIssue]
        }
        if any(
            int(joint_type) not in scalar_joint_types
            for joint_type in actuator_joint_types
        ):
            raise ValueError(
                "Actuated joints must be hinge or slide joints to build "
                "one-dimensional joint observations."
            )

        return actuator_joint_ids


    def _joint_pos_limits(
        self,
        model,
        actuator_joint_ids: np.ndarray,
    ) -> np.ndarray:
        
        joint_pos_limits = model.jnt_range[actuator_joint_ids].copy()
        unlimited_joints = ~model.jnt_limited[actuator_joint_ids].astype(bool)
        joint_pos_limits[unlimited_joints, 0] = -np.inf
        joint_pos_limits[unlimited_joints, 1] = np.inf

        return joint_pos_limits


    def _viewer_lock(self, env_id: int):
        if self.viewer is not None and env_id == self.playback_env_index:
            return self.viewer.lock()
        return nullcontext()


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        env_ids_list = self._validated_env_ids(env_ids)
        for env_id in env_ids_list:
            model = self.models[env_id]
            data = self.datas[env_id]
            with self._viewer_lock(env_id):
                if self.reset_keyframe_id == -1:
                    mujoco.mj_resetData(model, data)  # pyright: ignore[reportAttributeAccessIssue]
                else:
                    mujoco.mj_resetDataKeyframe(model, data, self.reset_keyframe_id)  # pyright: ignore[reportAttributeAccessIssue]
                mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]


    def apply_terrain(
        self,
        env_ids_tensor: torch.Tensor,
        patches: Sequence[TerrainPatch],
    ) -> None:

        if self._terrain is None:
            raise ValueError("Configure terrain collision geometry before applying patches.")
        env_ids_list = self._validated_env_ids(env_ids_tensor)
        if len(env_ids_list) != len(patches):
            raise ValueError("Terrain patches must match the selected environments.")

        # Validate every supplied grid before changing any model or render state.
        for patch in patches:
            self._terrain.validate_patch(patch)
        old_heights = [self.models[env_id].hfield_data.copy() for env_id in env_ids_list]
        old_grids = [self._terrain.grids[env_id] for env_id in env_ids_list]

        try:
            for env_id, patch in zip(env_ids_list, patches):
                with self._viewer_lock(env_id):
                    self._terrain.grids[env_id] = self._terrain.install(self.models[env_id], patch)
            self._terrain_changed(env_ids_list)

        except Exception as error:
            for env_id, heights, grid in zip(env_ids_list, old_heights, old_grids):
                with self._viewer_lock(env_id):
                    self.models[env_id].hfield_data[:] = heights
                    self._terrain.grids[env_id] = grid
            try:
                self._terrain_changed(env_ids_list)
            except Exception as rollback_error:
                raise ExceptionGroup(
                    "Terrain application and visual rollback failed",
                    [error, rollback_error]
                ) from error
            raise


    def _terrain_changed(self, env_ids_list: Sequence[int]) -> None:
        if self.playback_env_index in env_ids_list:
            self._terrain_render_dirty = True
            if self.viewer is not None and self._terrain is not None:
                self.viewer.update_hfield(self._terrain.hfield_id)


    def capture_reset_state(self, env_ids_tensor: torch.Tensor) -> object:
        env_ids_tuple = tuple(self._validated_env_ids(env_ids_tensor))
        return _MujocoResetSnapshot(
            env_ids_tuple=env_ids_tuple,
            models=tuple(self.models[env_id] for env_id in env_ids_tuple),
            datas=tuple(copy(self.datas[env_id]) for env_id in env_ids_tuple),
            hfields=tuple(self.models[env_id].hfield_data.copy() for env_id in env_ids_tuple),
            grids=(
                () if self._terrain is None
                else tuple(self._terrain.grids[env_id].copy() for env_id in env_ids_tuple)
            ),
        )


    def restore_reset_state(self, state: object) -> None:

        if not isinstance(state, _MujocoResetSnapshot):
            raise TypeError("Invalid MuJoCo reset snapshot.")
        if any(
            env_id >= self.num_envs
            or self.models[env_id] is not model
            for env_id, model in zip(state.env_ids_tuple, state.models)
        ):
            raise ValueError("Reset snapshot belongs to a different model configuration.")

        for index, (env_id, model, data, heights) in enumerate(
            zip(state.env_ids_tuple, state.models, state.datas, state.hfields)
        ):
            with self._viewer_lock(env_id):
                model.hfield_data[:] = heights
                mujoco.mj_copyData(self.datas[env_id], model, data)  # pyright: ignore[reportAttributeAccessIssue]
            if self._terrain is not None:
                self._terrain.grids[env_id] = state.grids[index]
        if self._terrain is not None:
            self._terrain_changed(state.env_ids_tuple)


    def get_terrain_spawn_geometry(self) -> TerrainSpawnGeometry:

        if self._terrain is None:
            raise ValueError("Terrain collision geometry is not configured.")
        model = self.models[0]
        data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
        if self.reset_keyframe_id < 0:
            mujoco.mj_resetData(model, data)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            mujoco.mj_resetDataKeyframe(    # pyright: ignore[reportAttributeAccessIssue]
                model,
                data,
                self.reset_keyframe_id
            )
        mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]

        robot_ids = np.flatnonzero((model.geom_bodyid != 0)
                                  & ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
        if robot_ids.size == 0:
            raise ValueError("Reset bounds require at least one robot collision shape.")

        rotations = data.geom_xmat[robot_ids].reshape(-1, 3, 3)
        boxes = model.geom_aabb[robot_ids]
        centres = data.geom_xpos[robot_ids] + np.einsum("nij,nj->ni", rotations, boxes[:, :3])
        radii = np.einsum("nij,nj->ni", np.abs(rotations), boxes[:, 3:])
        base_adr = self._find_base_joint_info(model)[1]
        base = data.qpos[base_adr:base_adr + 3]
        lower, upper = (centres - radii).min(axis=0) - base, (centres + radii).max(axis=0) - base
        return TerrainSpawnGeometry(
            tuple(self._terrain.origin.tolist()),
            tuple(base.tolist()),
            tuple(lower.tolist()),
            tuple(upper.tolist())
        )


    def set_root_positions(
        self,
        env_ids_tensor: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:

        env_ids_list = self._validated_env_ids(env_ids_tensor)
        if (
            positions.shape != (len(env_ids_list), 3)
            or not positions.is_floating_point()
            or not torch.isfinite(positions).all()
        ):
            raise ValueError(
                "Root positions must be finite floating-point data with shape [env, 3]."
            )

        values = positions.detach().cpu().double().numpy()
        base_adr = self._find_base_joint_info(self.models[0])[1]
        for env_id, position in zip(env_ids_list, values):
            with self._viewer_lock(env_id):
                self.datas[env_id].qpos[base_adr:base_adr + 3] = position
                mujoco.mj_forward(  # pyright: ignore[reportAttributeAccessIssue]
                    self.models[env_id],
                    self.datas[env_id]
                )


    def _validated_env_ids(
        self,
        env_ids: torch.Tensor | None,
    ) -> list[int]:

        if env_ids is None:
            return list(range(self.num_envs))
        if env_ids.ndim != 1 or env_ids.dtype != torch.long:
            raise ValueError("Environment IDs must be a one-dimensional long tensor.")
        env_ids_list = env_ids.tolist()
        if (
            len(set(env_ids_list)) != len(env_ids_list)
            or any(env_id < 0 or env_id >= self.num_envs for env_id in env_ids_list)
        ):
            raise ValueError("Environment IDs must be unique and in range.")
        return env_ids_list


    def get_ground_bounds(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:

        if self._terrain is None:
            return None
        env_ids_list = self._validated_env_ids(env_ids)
        origin = self._tensor(self._terrain.origin[:2])
        half_size = self._tensor(self._terrain.geometry.patch_size) / 2
        return (
            (origin - half_size).expand(len(env_ids_list), 2),
            (origin + half_size).expand(len(env_ids_list), 2),
        )


    def get_ground_heights(
        self,
        xy: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:

        env_ids_list = self._validated_env_ids(env_ids)
        if xy.ndim != 3 or xy.shape[-1] != 2 or xy.shape[0] != len(env_ids_list):
            raise ValueError("Ground query rows must match environments with shape [env, sample, 2].")
        if not xy.is_floating_point() or not torch.isfinite(xy).all():
            raise ValueError("Ground queries must contain finite floating-point coordinates.")

        if self._terrain is None:
            return super().get_ground_heights(xy, env_ids)
        points = xy.detach().cpu().double().numpy()
        result = np.empty(xy.shape[:2], dtype=np.float64)
        for row, env_id in enumerate(env_ids_list):
            result[row] = self._terrain.heights(env_id, points[row])
        return torch.as_tensor(result, dtype=xy.dtype, device=xy.device)


    def step(
        self,
        action: torch.Tensor,
    ) -> None:
        
        assert self.models is not None
        assert self.datas is not None

        action_np = action.detach().cpu().numpy()

        for env_id, data in enumerate(self.datas):
            data.ctrl[:] = action_np[env_id]

        if self._step_executor is None:
            for model, data in zip(self.models, self.datas):
                self._step_model(model, data)
        else:
            tuple(self._step_executor.map(
                self._step_model,
                self.models,
                self.datas,
            ))


    def _step_model(self, model, data) -> None:

        mujoco.mj_step(  # pyright: ignore[reportAttributeAccessIssue]
            model,
            data,
            nstep=self.frame_skip,
        )

    
    def render(self) -> np.ndarray | None:

        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            return self._human_render()
        if self.render_mode == "rgb_array":
            return self._rgb_array_render()
        raise RuntimeError(f"Invalid configured render mode: {self.render_mode!r}.")


    def prepare_playback(
        self,
        render_mode: str | None = None
    ) -> None:

        if render_mode is None:
            return
        if render_mode not in self.SUPPORTED_RENDER_MODES:
            raise ValueError(f"Unsupported render mode: {render_mode!r}.")

        self._playback_render_mode = self.render_mode
        self._playback_render_mode_changed = True
        self.render_mode = render_mode


    def terminate_playback(self) -> None:
        if self._playback_render_mode_changed:
            self.render_mode = self._playback_render_mode
            self._playback_render_mode = None
            self._playback_render_mode_changed = False


    def is_playback_open(self) -> bool:
        return self.viewer is None or bool(self.viewer.is_running())


    def _human_render(self) -> None:

        if self.viewer is None:
            self.viewer = viewer.launch_passive(
                self.models[self.playback_env_index],
                self.datas[self.playback_env_index],
            )
            with self.viewer.lock():
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING  # pyright: ignore[reportAttributeAccessIssue]
                self.viewer.cam.trackbodyid = self.model_context.base_id
        self.viewer.sync()


    def _rgb_array_render(self) -> np.ndarray:
        
        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self.models[self.playback_env_index],
            )
            self._render_camera = mujoco.MjvCamera()  # pyright: ignore[reportAttributeAccessIssue]
            mujoco.mjv_defaultFreeCamera(  # pyright: ignore[reportAttributeAccessIssue]
                self.models[self.playback_env_index], self._render_camera
            )
            self._render_camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING  # pyright: ignore[reportAttributeAccessIssue]
            self._render_camera.trackbodyid = self.model_context.base_id
            self._terrain_render_dirty = False

        if self._terrain_render_dirty:
            # MuJoCo 3.11 Renderer has no public heightfield-upload method.
            if self.renderer._gl_context is not None:
                self.renderer._gl_context.make_current()
            assert self._terrain is not None
            mujoco.mjr_uploadHField(    # pyright: ignore[reportAttributeAccessIssue]
                self.models[self.playback_env_index],
                self.renderer._mjr_context,
                self._terrain.hfield_id,
            )
            self._terrain_render_dirty = False

        self.renderer.update_scene(self.datas[self.playback_env_index], camera=self._render_camera)
        return self.renderer.render()


    def close(self) -> None:

        self._shutdown_step_executor()

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
            self._render_camera = None


    def _shutdown_step_executor(self) -> None:

        if self._step_executor is not None:
            self._step_executor.shutdown(wait=True)
            self._step_executor = None


    def get_state(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> SimulatorState:
        """Return views of the reusable full-state or reset-state buffers.

        Full and selected-environment reads use separate storage, so an
        auto-reset read does not overwrite the terminal full-state view from
        the same environment step. A later read of the same kind reuses and
        updates its buffer; clone any state that must be retained beyond it.
        """

        if env_ids is None:
            datas = self.datas
            models = self.models
            numpy_buffers = self._full_state_numpy_buffers
            tensor_buffers = self._full_state_tensor_buffers
        else:
            env_ids_list = env_ids.detach().cpu().tolist()
            datas = [
                self.datas[env_id]
                for env_id in env_ids_list
            ]
            models = [
                self.models[env_id]
                for env_id in env_ids_list
            ]
            numpy_buffers = self._reset_state_numpy_buffers
            tensor_buffers = self._reset_state_tensor_buffers

        if numpy_buffers is None or tensor_buffers is None:
            self._initialize_state_buffers()
            if env_ids is None:
                numpy_buffers = self._full_state_numpy_buffers
                tensor_buffers = self._full_state_tensor_buffers
            else:
                numpy_buffers = self._reset_state_numpy_buffers
                tensor_buffers = self._reset_state_tensor_buffers

        basic_state = self._get_basic_state(
            datas,
            numpy_buffers,
            tensor_buffers,
        )
        base_velocity_state = self._get_base_velocity_state(
            datas,
            numpy_buffers,
            tensor_buffers,
        )
        geom_xvel = self._get_geom_xvel(
            models,
            datas,
            numpy_buffers,
            tensor_buffers,
        )
        (
            contact_geom_ids,
            contact_forces
        ) = self._get_contact_state(models, datas)
        foot_contact_state = self._get_foot_contact_state(
            contact_geom_ids,
            contact_forces,
        )

        ground_state = self._get_ground_state(
            qpos=basic_state["qpos"],
            geom_xpos=basic_state["geom_xpos"],
            env_ids=env_ids,
        )
        return SimulatorState(
            **ground_state,
            **basic_state,
            **base_velocity_state,
            **foot_contact_state,
            contact_geom_ids=contact_geom_ids,
            contact_forces=contact_forces,
            geom_xvel=geom_xvel,
        )


    def _initialize_state_buffers(self) -> None:

        self._state_staging_tensors.clear()
        (
            self._full_state_numpy_buffers,
            self._full_state_tensor_buffers,
        ) = self._create_state_buffers()
        (
            self._reset_state_numpy_buffers,
            self._reset_state_tensor_buffers,
        ) = self._create_state_buffers()


    def _create_state_buffers(
        self,
    ) -> tuple[
        MujocoFixedStateBuffers[np.ndarray],
        MujocoFixedStateBuffers[torch.Tensor],
    ]:

        model = self.models[0]
        buffer_capacity = len(self.models)
        shapes = {
            "qpos": (buffer_capacity, model.nq),
            "qvel": (buffer_capacity, model.nv),
            "qacc": (buffer_capacity, model.nv),
            "ctrl": (buffer_capacity, model.nu),
            "geom_xpos": (buffer_capacity, model.ngeom, 3),
            "actuator_force": (buffer_capacity, model.nu),
            "base_lin_vel_body": (buffer_capacity, 3),
            "base_ang_vel_body": (buffer_capacity, 3),
            "geom_xvel": (buffer_capacity, model.ngeom, 6),
        }
        numpy_buffers = MujocoFixedStateBuffers(**{
            name: np.empty(shape, dtype=np.float64)
            for name, shape in shapes.items()
        })
        staging_tensors = {
            name: torch.from_numpy(values)
            for name, values in numpy_buffers.iter_fields()
        }
        tensor_buffers = MujocoFixedStateBuffers(**{
            name: (
                staging_tensors[name]
                if (
                    self.context.device.type == "cpu"
                    and self.context.dtype == torch.float64
                )
                else torch.empty(
                    values.shape,
                    dtype=self.context.dtype,
                    device=self.context.device,
                )
            )
            for name, values in numpy_buffers.iter_fields()
        })
        self._state_staging_tensors.update({
            id(values): staging_tensors[name]
            for name, values in numpy_buffers.iter_fields()
        })
        return numpy_buffers, tensor_buffers


    def _publish_state_fields(
        self,
        names: tuple[str, ...],
        numpy_buffers: MujocoFixedStateBuffers[np.ndarray],
        tensor_buffers: MujocoFixedStateBuffers[torch.Tensor],
        batch_size: int,
    ) -> dict[str, torch.Tensor]:

        for name in names:
            numpy_buffer = getattr(numpy_buffers, name)
            tensor_buffer = getattr(tensor_buffers, name)
            if tensor_buffer.data_ptr() != numpy_buffer.ctypes.data:
                tensor_buffer[:batch_size].copy_(
                    self._state_staging_tensors[
                        id(numpy_buffer)
                    ][:batch_size]
                )
        return {
            name: getattr(tensor_buffers, name)[:batch_size]
            for name in names
        }

    def _get_foot_contact_state(
        self,
        contact_geom_ids: torch.Tensor,
        contact_forces: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        
        geom1 = contact_geom_ids[..., 0].unsqueeze(-1)
        geom2 = contact_geom_ids[..., 1].unsqueeze(-1)
        foot_ids = self.model_context.foot_geom_ids.view(1, 1, -1)
        floor_ids = self.model_context.floor_geom_ids
        foot_ground_pair = (
            ((geom1 == foot_ids) & torch.isin(geom2, floor_ids))
            | ((geom2 == foot_ids) & torch.isin(geom1, floor_ids))
        )
        normal_force = contact_forces[..., 0].abs().unsqueeze(-1)
        if self._terrain is not None:
            # Heightfields split support across triangle contacts. Thresholding
            # each triangle separately can mark a supported foot as airborne.
            normal_force = (normal_force * foot_ground_pair).sum(dim=1, keepdim=True)
            foot_ground_pair = foot_ground_pair.any(dim=1, keepdim=True)
        forceful_contact = (
            (normal_force > 0.0)
            & (normal_force >= self.foot_contact_force_threshold)
        )
        valid_contact = foot_ground_pair & forceful_contact
        foot_ground_contact = valid_contact.any(dim=1)
        foot_contact_normal_force = (normal_force * valid_contact).sum(dim=1)

        return {
            "foot_ground_contact": foot_ground_contact,
            "foot_contact_normal_force": foot_contact_normal_force,
        }


    def _get_basic_state(
        self,
        datas,
        numpy_buffers: MujocoFixedStateBuffers[np.ndarray] | None = None,
        tensor_buffers: MujocoFixedStateBuffers[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:

        if numpy_buffers is None or tensor_buffers is None:
            if (
                self._reset_state_numpy_buffers is None
                or self._reset_state_tensor_buffers is None
            ):
                self._initialize_state_buffers()
            numpy_buffers = self._reset_state_numpy_buffers
            tensor_buffers = self._reset_state_tensor_buffers
        assert numpy_buffers is not None
        assert tensor_buffers is not None

        names = (
            "qpos",
            "qvel",
            "qacc",
            "ctrl",
            "geom_xpos",
            "actuator_force",
        )
        for env_index, data in enumerate(datas):
            for name in names:
                np.copyto(
                    getattr(numpy_buffers, name)[env_index],
                    getattr(data, name),
                )

        return self._publish_state_fields(
            names,
            numpy_buffers,
            tensor_buffers,
            len(datas),
        )


    def _get_base_velocity_state(
        self,
        datas,
        numpy_buffers: MujocoFixedStateBuffers[np.ndarray] | None = None,
        tensor_buffers: MujocoFixedStateBuffers[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:

        base_id = self.model_context.base_id
        lin_vel_ids = self.model_context.base_lin_vel_qvel_ids.cpu().numpy()
        ang_vel_ids = self.model_context.base_ang_vel_qvel_ids.cpu().numpy()

        if numpy_buffers is None or tensor_buffers is None:
            if (
                self._reset_state_numpy_buffers is None
                or self._reset_state_tensor_buffers is None
            ):
                self._initialize_state_buffers()
            numpy_buffers = self._reset_state_numpy_buffers
            tensor_buffers = self._reset_state_tensor_buffers
        assert numpy_buffers is not None
        assert tensor_buffers is not None

        for env_index, data in enumerate(datas):
            rotation_body_to_world = data.xmat[base_id].reshape(3, 3)
            rotation_world_to_body = rotation_body_to_world.T
            np.matmul(
                rotation_world_to_body,
                data.qvel[lin_vel_ids],
                out=numpy_buffers.base_lin_vel_body[env_index],
            )
            # Free-joint angular qvel is already expressed in the local body frame.
            np.copyto(
                numpy_buffers.base_ang_vel_body[env_index],
                data.qvel[ang_vel_ids]
            )

        return self._publish_state_fields(
            ("base_lin_vel_body", "base_ang_vel_body"),
            numpy_buffers,
            tensor_buffers,
            len(datas),
        )


    def _get_geom_xvel(
        self,
        models,
        datas,
        numpy_buffers: MujocoFixedStateBuffers[np.ndarray] | None = None,
        tensor_buffers: MujocoFixedStateBuffers[torch.Tensor] | None = None,
    ) -> torch.Tensor:

        if numpy_buffers is None or tensor_buffers is None:
            if (
                self._reset_state_numpy_buffers is None
                or self._reset_state_tensor_buffers is None
            ):
                self._initialize_state_buffers()
            numpy_buffers = self._reset_state_numpy_buffers
            tensor_buffers = self._reset_state_tensor_buffers
        assert numpy_buffers is not None
        assert tensor_buffers is not None
        
        geom_xvel = numpy_buffers.geom_xvel
        for env_index, (model, data) in enumerate(zip(models, datas)):
            for geom_id in range(model.ngeom):
                mujoco.mj_objectVelocity(  # pyright: ignore[reportAttributeAccessIssue]
                    model,
                    data,
                    mujoco.mjtObj.mjOBJ_GEOM,  # pyright: ignore[reportAttributeAccessIssue]
                    geom_id,
                    geom_xvel[env_index, geom_id],
                    0,
                )

        return self._publish_state_fields(
            ("geom_xvel",),
            numpy_buffers,
            tensor_buffers,
            len(datas),
        )["geom_xvel"]


    def _get_contact_state(
        self,
        models,
        datas,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        contact_pairs = []
        contact_force_arrays = []

        for model, data in zip(models, datas):
            pairs = np.empty((data.ncon, 2), dtype=np.int64)
            forces = np.empty((data.ncon, 6), dtype=np.float64)

            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                pairs[contact_index] = (contact.geom1, contact.geom2)
                mujoco.mj_contactForce(  # pyright: ignore[reportAttributeAccessIssue]
                    model,
                    data,
                    contact_index,
                    forces[contact_index],
                )

            contact_pairs.append(pairs)
            contact_force_arrays.append(forces)

        max_contacts = max(
            (pairs.shape[0] for pairs in contact_pairs),
            default=0,
        )

        contact_geom_ids = torch.full(
            (len(datas), max_contacts, 2),
            -1,
            dtype=torch.long,
            device=self.context.device,
        )

        contact_forces = torch.zeros(
            (len(datas), max_contacts, 6),
            dtype=self.context.dtype,
            device=self.context.device,
        )
        
        for env_index, pairs in enumerate(contact_pairs):
            if pairs.shape[0] > 0:
                contact_geom_ids[env_index, :pairs.shape[0]] = (
                    torch.as_tensor(
                        pairs,
                        dtype=torch.long,
                        device=self.context.device,
                    )
                )
                contact_forces[env_index, :pairs.shape[0]] = (
                    torch.as_tensor(
                        contact_force_arrays[env_index],
                        dtype=self.context.dtype,
                        device=self.context.device,
                    )
                )

        return contact_geom_ids, contact_forces
