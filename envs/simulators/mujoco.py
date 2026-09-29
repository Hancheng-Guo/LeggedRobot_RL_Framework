import os


if os.name == "nt":
    os.environ.setdefault("MUJOCO_GL", "wgl")


import mujoco
import torch
import numpy as np
from mujoco import viewer
from dataclasses import dataclass, fields
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, TypeVar

from envs.simulators.base import BaseSimulator
from envs.simulators.utils.context import ModelContext
from envs.simulators.utils.state import SimulatorState
from utils.component import Component
from app.utils.context import RuntimeContext
from utils.param import update_attributes


BufferType = TypeVar("BufferType", np.ndarray, torch.Tensor)


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
        self.foot_contact_force_threshold: float = self.FOOT_CONTACT_FORCE_THRESHOLD
        self.render_mode: str | None = None
        self.reset_keyframe: str | None = None
        self.reset_keyframe_id: int = -1

        self.models: list[mujoco.MjModel] = []  # pyright: ignore[reportAttributeAccessIssue]
        self.datas: list[mujoco.MjData] = []    # pyright: ignore[reportAttributeAccessIssue]
        self.viewer = None
        self.renderer = None
        self._step_executor: ThreadPoolExecutor | None = None
        self._full_state_numpy_buffers: MujocoFixedStateBuffers[np.ndarray] | None = None
        self._full_state_tensor_buffers: MujocoFixedStateBuffers[torch.Tensor] | None = None
        self._reset_state_numpy_buffers: MujocoFixedStateBuffers[np.ndarray] | None = None
        self._reset_state_tensor_buffers: MujocoFixedStateBuffers[torch.Tensor] | None = None
        self._state_staging_tensors: dict[int, torch.Tensor] = {}


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
        foot_contact_force_threshold: float | None = None,
        reset_keyframe: str | None = None,
    ) -> None:

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

        self.models = [
            mujoco.MjModel.from_xml_path(   # pyright: ignore[reportAttributeAccessIssue]
                str(self.model_path)
            )
            for _ in range(self.num_envs)
        ]
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
                requested_names=self.floor_geom_names,
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


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs,
                dtype=torch.long,
            )

        for env_id in env_ids.tolist():
            model = self.models[env_id]
            data = self.datas[env_id]

            if self.reset_keyframe_id == -1:
                mujoco.mj_resetData(model, data)    # pyright: ignore[reportAttributeAccessIssue]
            else:
                mujoco.mj_resetDataKeyframe(        # pyright: ignore[reportAttributeAccessIssue]
                    model,
                    data,
                    self.reset_keyframe_id
                )

            mujoco.mj_forward(model, data)      # pyright: ignore[reportAttributeAccessIssue]


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


    def _human_render(self) -> None:

        if self.viewer is None:
            self.viewer = viewer.launch_passive(
                self.models[0],
                self.datas[0],
            )
        self.viewer.sync()


    def _rgb_array_render(self) -> np.ndarray:
        
        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self.models[0],
            )

        self.renderer.update_scene(self.datas[0])
        return self.renderer.render()


    def close(self) -> None:

        self._shutdown_step_executor()

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


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
            indices = env_ids.detach().cpu().tolist()
            datas = [
                self.datas[i]
                for i in indices
            ]
            models = [
                self.models[i]
                for i in indices
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

        return SimulatorState(
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
        normal_force = contact_forces[..., 0].abs()
        forceful_contact = (
            (normal_force > 0.0)
            & (normal_force >= self.foot_contact_force_threshold)
        )
        valid_contact = foot_ground_pair & forceful_contact.unsqueeze(-1)
        foot_ground_contact = valid_contact.any(dim=1)
        foot_contact_normal_force = (
            normal_force.unsqueeze(-1) * valid_contact
        ).sum(dim=1)

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
            np.matmul(
                rotation_world_to_body,
                data.qvel[ang_vel_ids],
                out=numpy_buffers.base_ang_vel_body[env_index],
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
