import mujoco
import torch
import warnings
import numpy as np
from mujoco import viewer
from pathlib import Path
from collections.abc import Callable

from envs.simulators.base import BaseSimulator
from envs.simulators.utils.context import ModelContext
from utils.component import Component
from app.utils.context import RuntimeContext
from utils.param import update_attributes


class MujocoSimulator(BaseSimulator):

    def __init__(
        self,
        context: RuntimeContext,
    ) -> None:

        self.context = context

        self.num_envs: int
        self.model_path: Path
        self.sim_dt: float
        self.frame_skip: int
        self.geom_foot_names: tuple[str, ...]
        self.geom_floor_names: tuple[str, ...]
        self.render_mode: str | None = None
        self.reset_keyframe: str | None = None
        self.reset_keyframe_id: int = -1

        self.models: list[mujoco.MjModel] = []  # pyright: ignore[reportAttributeAccessIssue]
        self.datas: list[mujoco.MjData] = []    # pyright: ignore[reportAttributeAccessIssue]
        self.viewer = None
        self.renderer = None

        self._RENDER_TYPE_MAP: dict[
            str,
            Callable[[], np.ndarray | None],
        ] = {
            "human": self._human_render,
            "rgb_array": self._rgb_array_render
        }


    def config_update(
        self,
        component: Component,
        num_envs: int | None = None,
        model_path: Path | str | None = None,
        sim_dt: float | None = None,
        frame_skip: int | None = None,
        render_mode: str | None = None,
        geom_foot_names: list[str] | tuple[str, ...] | None = None,
        geom_floor_names: list[str] | tuple[str, ...] | None = None,
        reset_keyframe: str | None = None,
    ) -> None:

        if num_envs is not None and num_envs <= 0:
            raise ValueError(
                "'num_envs' should be a int greater than 0."
            )
        
        update_attributes(
            self,
            num_envs=num_envs,
            model_path=None if model_path is None else Path(model_path),
            sim_dt=sim_dt,
            frame_skip=frame_skip,
            geom_foot_names=(
                tuple(geom_foot_names)
                if geom_foot_names is not None
                else None
            ),
            geom_floor_names=(
                tuple(geom_floor_names)
                if geom_floor_names is not None
                else None
            ),
        )

        # A simulator config that supplies a model path owns the complete reset
        # configuration. Incremental updates (for example, changing num_envs)
        # keep the previously selected keyframe.
        if model_path is not None:
            self.reset_keyframe = reset_keyframe
        elif reset_keyframe is not None:
            self.reset_keyframe = reset_keyframe

        if render_mode is not None:
            if render_mode in self._RENDER_TYPE_MAP:
                self.render_mode = render_mode
            else:
                warnings.warn(
                    f"Unsupported render mode: {render_mode!r}."
                )
                self.render_mode = None

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


    def _build_model_context(self) -> None:

        model = self.models[0]
        
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
            base_pos_qpos_ids=self._indices(
                np.arange(base_qpos_adr, base_qpos_adr + 3)
            ),
            base_quat_qpos_ids=self._indices(
                np.arange(base_qpos_adr + 3, base_qpos_adr + 7)
            ),
            base_lin_vel_qvel_ids=self._indices(
                np.arange(base_qvel_adr, base_qvel_adr + 3)
            ),
            base_ang_vel_qvel_ids=self._indices(
                np.arange(base_qvel_adr + 3, base_qvel_adr + 6)
            ),
            
            # joint_names=names(mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
            joint_qpos_ids=self._indices(joint_qpos_ids),
            joint_qvel_ids=self._indices(
                model.jnt_dofadr[actuator_joint_ids]
            ),
            joint_default_pos=self._tensor(joint_default_pos),
            joint_pos_limits=self._tensor(joint_pos_limits),

            # actuator_names=names(mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
            actuator_ctrl_range = self._tensor(model.actuator_ctrlrange),
            actuator_default_ctrl = self._tensor(actuator_default_ctrl),

            geom_names=geom_names,
            geom_body_ids=self._indices(model.geom_bodyid),
            geom_foot_ids=self._geom_ids_by_name(
                geom_names,
                requested_names=self.geom_foot_names,
                default_suffix="foot",
            ),
            geom_floor_ids=self._geom_ids_by_name(
                geom_names,
                requested_names=self.geom_floor_names,
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


    def _tensor(
        self,
        value: np.ndarray,
    ) -> torch.Tensor:
        
        return torch.as_tensor(
            value,
            dtype=self.context.dtype,
            device=self.context.device,
        )


    def _indices(
        self,
        object_indices: np.ndarray,
    ) -> torch.Tensor:
        
        return torch.as_tensor(
            object_indices,
            dtype=torch.long,
            device=self.context.device,
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

        return self._indices(np.asarray(ids, dtype=np.int64))


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

        for _ in range(self.frame_skip):
            for model, data in zip(self.models, self.datas):
                mujoco.mj_step(  # pyright: ignore[reportAttributeAccessIssue]
                    model,
                    data,
                )

    
    def render(self) -> np.ndarray | None:

        if self.render_mode is None:
            return None

        render_type = self._RENDER_TYPE_MAP[self.render_mode]
        render_result = render_type()

        return render_result


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

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


    def get_state(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:

        if env_ids is None:
            datas = self.datas
            models = self.models
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

        basic_state = self._get_basic_state(datas)
        base_velocity_state = self._get_base_velocity_state(datas)
        geom_xvel = self._get_geom_xvel(models, datas)
        (
            contact_geom_ids,
            contact_forces
        ) = self._get_contact_state(models, datas)

        return (
            basic_state |
            base_velocity_state |
            {
                "contact_geom_ids": contact_geom_ids,
                "contact_forces": contact_forces,
                "geom_xvel": geom_xvel,
            }
        )


    def _get_basic_state(
        self,
        datas,
    ) -> dict[str, torch.Tensor]:
        
        qpos = []
        qvel = []
        qacc = []
        ctrl = []
        geom_xpos = []
        actuator_force = []

        for data in datas:
            qpos.append(data.qpos)
            qvel.append(data.qvel)
            qacc.append(data.qacc)
            ctrl.append(data.ctrl)
            geom_xpos.append(data.geom_xpos)
            actuator_force.append(data.actuator_force)

        return {
            "qpos": self._tensor(np.asarray(qpos)),
            "qvel": self._tensor(np.asarray(qvel)),
            "qacc": self._tensor(np.asarray(qacc)),
            "ctrl": self._tensor(np.asarray(ctrl)),
            "geom_xpos": self._tensor(np.asarray(geom_xpos)),
            "actuator_force": self._tensor(np.asarray(actuator_force)),
        }


    def _get_base_velocity_state(
        self,
        datas,
    ) -> dict[str, torch.Tensor]:

        base_id = self.model_context.base_id
        lin_vel_ids = self.model_context.base_lin_vel_qvel_ids.cpu().numpy()
        ang_vel_ids = self.model_context.base_ang_vel_qvel_ids.cpu().numpy()

        base_lin_vel_body = []
        base_ang_vel_body = []
        for data in datas:
            rotation_body_to_world = data.xmat[base_id].reshape(3, 3)
            rotation_world_to_body = rotation_body_to_world.T
            base_lin_vel_body.append(
                rotation_world_to_body @ data.qvel[lin_vel_ids]
            )
            base_ang_vel_body.append(
                rotation_world_to_body @ data.qvel[ang_vel_ids]
            )

        return {
            "base_lin_vel_body": self._tensor(np.asarray(base_lin_vel_body)),
            "base_ang_vel_body": self._tensor(np.asarray(base_ang_vel_body)),
        }


    def _get_geom_xvel(
        self,
        models,
        datas,
    ) -> torch.Tensor:
        
        geom_xvel = []
        for model, data in zip(models, datas):

            per_env = []
            for geom_id in range(model.ngeom):

                velocity = np.zeros(6, dtype=np.float64)
                mujoco.mj_objectVelocity(  # pyright: ignore[reportAttributeAccessIssue]
                    model,
                    data,
                    mujoco.mjtObj.mjOBJ_GEOM,  # pyright: ignore[reportAttributeAccessIssue]
                    geom_id,
                    velocity,
                    0,
                )
                per_env.append(velocity)

            geom_xvel.append(
                torch.as_tensor(
                    np.asarray(per_env),
                    dtype=self.context.dtype,
                )
            )

        return torch.stack(geom_xvel).to(self.context.device)


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
