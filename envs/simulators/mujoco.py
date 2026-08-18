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

        self.model_context = ModelContext(
            nq = model.nq,
            nv = model.nv,
            nu = model.nu,
            na = model.na,
            # body_names=names(mujoco.mjtObj.mjOBJ_BODY, model.nbody),  # pyright: ignore[reportAttributeAccessIssue]
            # joint_names=names(mujoco.mjtObj.mjOBJ_JOINT, model.njnt),  # pyright: ignore[reportAttributeAccessIssue]
            # actuator_names=names(mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),  # pyright: ignore[reportAttributeAccessIssue]
            actuator_ctrl_range = tensors(model.actuator_ctrlrange),
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
