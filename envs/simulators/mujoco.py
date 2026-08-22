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
        self.render_mode: str | None = None

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
        )

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
        self._build_model_context()


    def _build_model_context(self) -> None:

        model = self.models[0]

        def names(object_type, count: int) -> tuple[str | None, ...]:
            return tuple(
                mujoco.mj_id2name(model, object_type, index)  # pyright: ignore[reportAttributeAccessIssue]
                for index in range(count)
            )

        def tensors(object_type: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(
                object_type,
                dtype=self.context.dtype,
                device=self.context.device,
            )

        def indices(object_indices: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(
                object_indices,
                dtype=torch.long,
                device=self.context.device,
            )

        # find the free joint in the model, which represents the base of the robot
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

        # check all transmission type of actuators are JOINT or JOINTINPARENT.
        # if transmission type is TENDON, the actuator is connected to a tendon,
        # which does not have a direct mapping to a joint
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

        self.model_context = ModelContext(
            nq = model.nq,
            nv = model.nv,
            nu = model.nu,
            na = model.na,
            gravity=tensors(model.opt.gravity),
            # base_names=names(mujoco.mjtObj.mjOBJ_BODY, model.nbody),  # pyright: ignore[reportAttributeAccessIssue]
            base_pos_qpos_ids=indices(
                np.arange(base_qpos_adr, base_qpos_adr + 3)
            ),
            base_quat_qpos_ids=indices(
                np.arange(base_qpos_adr + 3, base_qpos_adr + 7)
            ),
            base_ang_vel_qvel_ids=indices(
                np.arange(base_qvel_adr + 3, base_qvel_adr + 6)
            ),
            body_names=names(
                mujoco.mjtObj.mjOBJ_BODY,  # pyright: ignore[reportAttributeAccessIssue]
                model.nbody,
            ),
            # joint_names=names(mujoco.mjtObj.mjOBJ_JOINT, model.njnt),  # pyright: ignore[reportAttributeAccessIssue]
            joint_qpos_ids=indices(
                model.jnt_qposadr[actuator_joint_ids]
            ),
            joint_qvel_ids=indices(
                model.jnt_dofadr[actuator_joint_ids]
            ),
            # actuator_names=names(mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),  # pyright: ignore[reportAttributeAccessIssue]
            actuator_ctrl_range = tensors(model.actuator_ctrlrange),
            geom_names=names(
                mujoco.mjtObj.mjOBJ_GEOM,  # pyright: ignore[reportAttributeAccessIssue]
                model.ngeom,
            ),
            geom_body_ids=indices(model.geom_bodyid),
        )


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
            mujoco.mj_resetData(    # pyright: ignore[reportAttributeAccessIssue]
                self.models[env_id],
                self.datas[env_id],
            )

            mujoco.mj_forward(      # pyright: ignore[reportAttributeAccessIssue]
                self.models[env_id],
                self.datas[env_id],
            )


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
        else:
            indices = env_ids.detach().cpu().tolist()
            datas = [
                self.datas[i]
                for i in indices
            ]

        qpos = torch.stack([
            torch.as_tensor(
                data.qpos,
                dtype=self.context.dtype,
            )
            for data in datas
        ]).to(self.context.device)

        qvel = torch.stack([
            torch.as_tensor(
                data.qvel,
                dtype=self.context.dtype,
            )
            for data in datas
        ]).to(self.context.device)

        qacc = torch.stack([
            torch.as_tensor(
                data.qacc,
                dtype=self.context.dtype,
            )
            for data in datas
        ]).to(self.context.device)

        ctrl = torch.stack([
            torch.as_tensor(
                data.ctrl,
                dtype=self.context.dtype,
            )
            for data in datas
        ]).to(self.context.device)

        contact_pairs = [
            np.asarray(
                [
                    (data.contact[index].geom1, data.contact[index].geom2)
                    for index in range(data.ncon)
                ],
                dtype=np.int64,
            ).reshape(-1, 2)
            for data in datas
        ]
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
        for env_index, pairs in enumerate(contact_pairs):
            if pairs.shape[0] > 0:
                contact_geom_ids[env_index, :pairs.shape[0]] = (
                    torch.as_tensor(
                        pairs,
                        dtype=torch.long,
                        device=self.context.device,
                    )
                )

        # time = torch.tensor(
        #     [data.time for data in datas],
        #     dtype=self.context.dtype,
        #     device=self.context.device,
        # )

        # body_pos = torch.stack([
        #     torch.as_tensor(
        #         data.xpos,
        #         dtype=self.context.dtype,
        #     )
        #     for data in datas
        # ]).to(self.context.device)

        # body_quat = torch.stack([
        #     torch.as_tensor(
        #         data.xquat,
        #         dtype=self.context.dtype,
        #     )
        #     for data in datas
        # ]).to(self.context.device)

        # body_vel = torch.stack([
        #     torch.as_tensor(
        #         data.cvel,
        #         dtype=self.context.dtype,
        #     )
        #     for data in datas
        # ]).to(self.context.device)

        # sensor_data = torch.stack([
        #     torch.as_tensor(
        #         data.sensordata,
        #         dtype=self.context.dtype,
        #     )
        #     for data in datas
        # ]).to(self.context.device)

        # actuator_force = torch.stack([
        #     torch.as_tensor(
        #         data.actuator_force,
        #         dtype=self.context.dtype,
        #     )
        #     for data in datas
        # ]).to(self.context.device)

        return {
            "qpos": qpos,
            "qvel": qvel,
            "qacc": qacc,
            "ctrl": ctrl,
            "contact_geom_ids": contact_geom_ids,
            # "time": time,
            # "body_pos": body_pos,
            # "body_quat": body_quat,
            # "body_vel": body_vel,
            # "sensor_data": sensor_data,
            # "actuator_force": actuator_force,
        }



    # def set_state(
    #     self,
    #     state: dict[str, Any],
    # ) -> None:

    #     assert self.model is not None
    #     assert self.data is not None

    #     self.data.qpos[:] = state["qpos"]
    #     self.data.qvel[:] = state["qvel"]

    #     mujoco.mj_forward(
    #         self.model,
    #         self.data,
    #     )


    # def get_joint_positions(self):

    #     assert self.data is not None

    #     return self.data.qpos.copy()


    # def get_joint_velocities(self):

    #     assert self.data is not None

    #     return self.data.qvel.copy()


    # def get_contacts(self):

    #     assert self.data is not None

    #     contacts = []

    #     for i in range(
    #         self.data.ncon
    #     ):
    #         contact = self.data.contact[i]
    #         contacts.append(contact)

    #     return contacts
