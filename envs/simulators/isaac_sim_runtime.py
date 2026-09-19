from __future__ import annotations

import torch
import numpy as np
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.utils.context import RuntimeContext
from envs.simulators.isaac_sim_backend import IsaacSimModelMetadata, IsaacSimResetState
from envs.simulators.isaac_sim_model import IsaacSimModelConverter


class IsaacSimRuntime:
    """Thin owner of the optional Isaac Sim/Kit runtime.

    All Isaac imports intentionally happen after ``SimulationApp`` starts and
    only when this backend is selected. The public simulator remains importable
    on machines that do not have Isaac Sim installed.
    """

    def __init__(
        self,
        context: RuntimeContext
    ) -> None:
        
        self.context = context
        self.metadata: IsaacSimModelMetadata
        self._app: Any = None
        self._world: Any = None
        self._articulation: Any = None
        self._body_view: Any = None
        self._camera: Any = None
        self._closed = False


    def configure(
        self,
        *,
        num_envs: int,
        model_path: Path,
        ros_package_paths: Sequence[Mapping[str, str]],
        sim_dt: float,
        frame_skip: int,
        render_mode: str | None,
        env_spacing: float,
        robot_prim_path: str,
        base_body_prim_path: str | None,
        foot_body_prim_paths: Sequence[str],
        floor_prim_paths: Sequence[str],
        foot_contact_force_threshold: float,
        camera_prim_path: str | None,
        camera_resolution: Sequence[int],
        merge_fixed_joints: bool,
        allow_self_collision: bool,
        joint_stiffness: float | Mapping[str, float] | None,
        joint_damping: float | Mapping[str, float] | None,
        reset_state: IsaacSimResetState,
    ) -> None:

        self.num_envs = int(num_envs)
        self.model_path = Path(model_path)
        self.sim_dt = float(sim_dt)
        self.frame_skip = int(frame_skip)
        self.render_mode = render_mode
        self.robot_prim_path = str(robot_prim_path)
        self.env_spacing = float(env_spacing)
        self._requested_base_body_prim_path = base_body_prim_path
        self._requested_foot_body_prim_paths = tuple(foot_body_prim_paths)
        self._requested_floor_prim_paths = tuple(floor_prim_paths)
        self._requested_camera_prim_path = camera_prim_path
        self.camera_resolution = tuple(camera_resolution)
        self.foot_contact_force_threshold = float(foot_contact_force_threshold)
        self.reset_state = reset_state

        self.base_body_prim_path: str
        self.foot_body_prim_paths: tuple[str, ...]
        self.floor_prim_paths: tuple[str, ...]
        self.camera_prim_path: str | None = None

        self._start_application()
        converter = IsaacSimModelConverter(
            self.context,
            ros_package_paths=ros_package_paths,
            merge_fixed_joints=merge_fixed_joints,
            allow_self_collision=allow_self_collision,
            joint_stiffness=joint_stiffness,
            joint_damping=joint_damping,
        )
        self.model_path = converter.convert_if_needed(self.model_path)
        self._build_scene()
        self._build_metadata()
        self._capture_default_state()
        self._ctrl = self.metadata.actuator_default_ctrl.repeat(
            self.num_envs,
            1,
        )
        self._previous_qvel: torch.Tensor | None = None
        self._last_frame_skip = 1


    def _start_application(self) -> None:
        
        try:
            from isaacsim.simulation_app import SimulationApp  # pyright: ignore[reportMissingImports]
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Isaac Sim 6.1 is required by the isaac_sim backend."
            ) from error

        headless = self.render_mode != "human"
        self._app = SimulationApp({"headless": headless})


    def _build_scene(self) -> None:
        # Isaac/Omniverse modules must be imported after SimulationApp starts.
        from isaacsim.core.api import World  # pyright: ignore[reportMissingImports]
        from isaacsim.core.cloner import GridCloner  # pyright: ignore[reportMissingImports]
        from isaacsim.core.prims import Articulation, RigidPrim  # pyright: ignore[reportMissingImports]
        from isaacsim.core.utils.stage import add_reference_to_stage  # pyright: ignore[reportMissingImports]
        from pxr import Usd, UsdGeom, UsdPhysics  # pyright: ignore[reportMissingImports]

        self._world = World(
            physics_dt=self.sim_dt,
            rendering_dt=self.sim_dt,
            backend="torch",
            device=str(self.context.device),
            physics_prim_path="/World/physicsScene",
        )
        stage = self._world.stage
        source_env_path = "/World/envs/env_0"
        source_robot_path = source_env_path + self._normalized_robot_prim_path()
        UsdGeom.Xform.Define(stage, source_env_path)
        add_reference_to_stage(
            usd_path=str(self.model_path),
            prim_path=source_robot_path,
        )
        self._select_physx_variant(stage, source_robot_path)
        articulation_root_relative_path = (
            self._find_articulation_root_relative_path(
                stage,
                source_robot_path,
                Usd,
                UsdPhysics,
            )
        )

        if self._requested_floor_prim_paths:
            missing_floor_paths = [
                path
                for path in self._requested_floor_prim_paths
                if not stage.GetPrimAtPath(path).IsValid()
            ]
            if missing_floor_paths:
                raise ValueError(
                    "Unknown floor Prim path(s): "
                    f"{missing_floor_paths}."
                )
            self.floor_prim_paths = self._requested_floor_prim_paths
        else:
            default_floor_path = "/World/GroundPlane"
            self._world.scene.add_default_ground_plane(
                prim_path=default_floor_path,
                name="GroundPlane",
            )
            self.floor_prim_paths = (default_floor_path,)

        cloner = GridCloner(spacing=self.env_spacing, stage=stage)
        cloner.define_base_env("/World/envs")
        env_paths = cloner.generate_paths("/World/envs/env", self.num_envs)
        cloner.clone(
            source_prim_path=source_env_path,
            prim_paths=env_paths,
            replicate_physics=True,
            base_env_path="/World/envs",
            root_path="/World/envs/env_",
            enable_env_ids=True,
            clone_in_fabric=True,
        )
        cloner.filter_collisions(
            physicsscene_path="/World/physicsScene",
            collision_root_path="/World/collisions",
            prim_paths=env_paths,
            global_paths=list(self.floor_prim_paths),
        )

        robot_expression = (
            "/World/envs/env_.*/" + self._normalized_robot_prim_path().lstrip("/")
            + articulation_root_relative_path
        )
        self._articulation = self._world.scene.add(
            Articulation(
                prim_paths_expr=robot_expression,
                name="isaac_sim_robots",
                reset_xform_properties=False,
            )
        )
        self._world.reset()

        body_relative_paths = self._rigid_body_relative_paths(
            stage,
            source_robot_path,
            Usd,
            UsdPhysics,
        )
        if not body_relative_paths:
            raise RuntimeError(
                f"No rigid bodies were found below {source_robot_path!r}."
            )
        self._body_prim_paths = body_relative_paths
        self._body_names = tuple(
            path.rsplit("/", 1)[-1]
            for path in body_relative_paths
        )
        body_paths = [
            f"/World/envs/env_{env_id}{self._normalized_robot_prim_path()}{relative}"
            for env_id in range(self.num_envs)
            for relative in body_relative_paths
        ]
        self._body_view = self._world.scene.add(
            RigidPrim(
                prim_paths_expr=body_paths,
                name="isaac_sim_robot_bodies",
                reset_xform_properties=False,
                track_contact_forces=True,
                contact_filter_prim_paths_expr=list(self.floor_prim_paths),
            )
        )
        self._world.reset()

        if self.render_mode == "rgb_array":
            from isaacsim.sensors.camera import Camera  # pyright: ignore[reportMissingImports]

            self.camera_prim_path = (
                self._requested_camera_prim_path or "/World/Camera"
            )
            self._camera = Camera(
                prim_path=self.camera_prim_path,
                position=np.asarray((2.5, 2.5, 1.8)),
                frequency=1.0 / (self.sim_dt * self.frame_skip),
                resolution=self.camera_resolution,
            )
            self._camera.initialize()


    def _normalized_robot_prim_path(self) -> str:
        return "/" + self.robot_prim_path.strip("/")


    @staticmethod
    def _select_physx_variant(stage: Any, robot_path: str) -> None:
        """Select an imported asset's PhysX payload before inspecting it."""

        robot_prim = stage.GetPrimAtPath(robot_path)
        variant_sets = robot_prim.GetVariantSets()
        if not variant_sets.HasVariantSet("Physics"):
            return

        physics_variant = variant_sets.GetVariantSet("Physics")
        variant_names = tuple(physics_variant.GetVariantNames())
        physx_variant = next(
            (
                name
                for name in variant_names
                if name.casefold() == "physx"
            ),
            None,
        )
        if physx_variant is None:
            raise RuntimeError(
                f"USD asset at {robot_path!r} defines a 'Physics' variant "
                f"set without a PhysX variant; available variants: "
                f"{list(variant_names)}."
            )
        if not physics_variant.SetVariantSelection(physx_variant):
            raise RuntimeError(
                f"Failed to select Physics={physx_variant!r} for "
                f"{robot_path!r}."
            )


    @staticmethod
    def _find_articulation_root_relative_path(
        stage: Any,
        robot_path: str,
        usd: Any,
        usd_physics: Any,
    ) -> str:
        robot_prim = stage.GetPrimAtPath(robot_path)
        articulation_roots = [
            str(prim.GetPath())
            for prim in usd.PrimRange(robot_prim)
            if prim.HasAPI(usd_physics.ArticulationRootAPI)
        ]
        if not articulation_roots:
            raise RuntimeError(
                "No articulation root was found at or below "
                f"{robot_path!r}."
            )
        if len(articulation_roots) > 1:
            raise RuntimeError(
                "Expected one articulation root at or below "
                f"{robot_path!r}, found {articulation_roots}."
            )
        return articulation_roots[0][len(robot_path):]


    @staticmethod
    def _rigid_body_relative_paths(
        stage: Any,
        robot_path: str,
        usd: Any,
        usd_physics: Any,
    ) -> tuple[str, ...]:
        
        root = stage.GetPrimAtPath(robot_path)
        paths: list[str] = []
        for prim in usd.PrimRange(root):
            if prim.HasAPI(usd_physics.RigidBodyAPI):
                paths.append(str(prim.GetPath())[len(robot_path):])
        return tuple(paths)


    def _build_metadata(self) -> None:

        dof_names = tuple(self._articulation.dof_names)
        joint_default_pos = self._tensor(
            self._articulation.get_joint_positions(clone=True)[0]
        )
        joint_pos_limits = self._tensor(
            self._articulation.get_dof_limits(clone=True)[0]
        )
        self.base_body_prim_path = (
            self._requested_base_body_prim_path
            or self._body_prim_paths[0]
        )
        self.foot_body_prim_paths = self._requested_foot_body_prim_paths or tuple(
            path
            for path in self._body_prim_paths
            if path.rsplit("/", 1)[-1].lower().endswith("foot")
        )
        if not self.foot_body_prim_paths:
            raise ValueError(
                "No foot bodies were inferred; configure "
                "'foot_body_prim_paths'."
            )
        unknown_paths = {
            self.base_body_prim_path,
            *self.foot_body_prim_paths,
        } - set(self._body_prim_paths)
        if unknown_paths:
            raise ValueError(
                "Unknown rigid body Prim path(s): "
                f"{sorted(unknown_paths)}. Available paths: "
                f"{list(self._body_prim_paths)}."
            )
        self.metadata = IsaacSimModelMetadata(
            body_names=self._body_names,
            dof_names=dof_names,
            joint_default_pos=joint_default_pos,
            joint_pos_limits=joint_pos_limits,
            actuator_ctrl_range=joint_pos_limits.clone(),
            actuator_default_ctrl=joint_default_pos.clone(),
            gravity=torch.tensor(
                (0.0, 0.0, -9.81),
                dtype=self.context.dtype,
                device=self.context.device,
            ),
            body_prim_paths=self._body_prim_paths,
            base_body_prim_path=self.base_body_prim_path,
            foot_body_prim_paths=self.foot_body_prim_paths,
            floor_prim_paths=self.floor_prim_paths,
        )


    def _capture_default_state(self) -> None:

        root_positions, root_orientations = self._articulation.get_world_poses(
            clone=True
        )
        self._default_root_positions = self._tensor(root_positions)
        self._default_root_orientations = self._tensor(root_orientations)
        self._default_joint_positions = self._tensor(
            self._articulation.get_joint_positions(clone=True)
        )
        if self.reset_state.base_position is not None:
            base_position = torch.tensor(
                self.reset_state.base_position,
                dtype=self.context.dtype,
                device=self.context.device,
            )
            # Preserve the grid offsets in x/y while applying the configured
            # local spawn pose to every environment.
            self._default_root_positions[:, :2] += base_position[:2]
            self._default_root_positions[:, 2] = base_position[2]
        if self.reset_state.base_orientation is not None:
            orientation = torch.tensor(
                self.reset_state.base_orientation,
                dtype=self.context.dtype,
                device=self.context.device,
            )
            orientation /= orientation.norm().clamp_min(
                torch.finfo(orientation.dtype).eps
            )
            self._default_root_orientations[:] = orientation
        joint_name_to_id = {
            name: index
            for index, name in enumerate(self.metadata.dof_names)
        }
        unknown_names = (
            self.reset_state.joint_positions.keys() - joint_name_to_id.keys()
        )
        if unknown_names:
            raise ValueError(
                "Unknown default joint position name(s): "
                f"{sorted(unknown_names)}."
            )
        for name, position in self.reset_state.joint_positions.items():
            self._default_joint_positions[:, joint_name_to_id[name]] = position
        self.metadata.joint_default_pos.copy_(self._default_joint_positions[0])
        self.metadata.actuator_default_ctrl.copy_(
            self._default_joint_positions[0]
        )


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        
        indices = self._indices(env_ids)
        self._articulation.set_world_poses(
            positions=self._default_root_positions[indices],
            orientations=self._default_root_orientations[indices],
            indices=indices,
        )
        self._articulation.set_velocities(
            torch.zeros(
                (indices.numel(), 6),
                dtype=self.context.dtype,
                device=self.context.device,
            ),
            indices=indices,
        )
        self._articulation.set_joint_positions(
            self._default_joint_positions[indices],
            indices=indices,
        )
        self._articulation.set_joint_velocities(
            torch.zeros_like(self._default_joint_positions[indices]),
            indices=indices,
        )
        self._articulation.set_joint_position_targets(
            self._default_joint_positions[indices],
            indices=indices,
        )
        self._ctrl[indices] = self._default_joint_positions[indices]
        self._previous_qvel = None


    def step(
        self,
        action: torch.Tensor,
        frame_skip: int
    ) -> None:
        
        targets = action.to(device=self.context.device, dtype=self.context.dtype)
        self._articulation.set_joint_position_targets(targets)
        self._ctrl.copy_(targets)
        self._last_frame_skip = frame_skip
        for _ in range(frame_skip):
            self._world.step(render=self.render_mode == "human")


    def get_state(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        
        state = self._get_full_state()
        if env_ids is None:
            return state
        indices = env_ids.to(device=self.context.device, dtype=torch.long)
        return {name: value[indices] for name, value in state.items()}


    def _get_full_state(self) -> dict[str, torch.Tensor]:

        root_pos, root_quat = self._articulation.get_world_poses(clone=True)
        root_velocity = self._articulation.get_velocities(clone=True)
        joint_pos = self._articulation.get_joint_positions(clone=True)
        joint_vel = self._articulation.get_joint_velocities(clone=True)
        root_pos = self._tensor(root_pos)
        root_quat = self._tensor(root_quat)
        root_velocity = self._tensor(root_velocity)
        joint_pos = self._tensor(joint_pos)
        joint_vel = self._tensor(joint_vel)
        qpos = torch.cat((root_pos, root_quat, joint_pos), dim=-1)
        qvel = torch.cat((root_velocity[..., :6], joint_vel), dim=-1)
        if self._previous_qvel is None:
            qacc = torch.zeros_like(qvel)
        else:
            qacc = (
                qvel - self._previous_qvel
            ) / (self.sim_dt * self._last_frame_skip)
        self._previous_qvel = qvel.clone()

        body_pos, _ = self._body_view.get_world_poses(clone=True)
        body_velocity = self._tensor(self._body_view.get_velocities(clone=True))
        body_pos = self._tensor(body_pos).reshape(self.num_envs, -1, 3)
        body_velocity = body_velocity.reshape(self.num_envs, -1, 6)
        # Isaac uses [linear, angular], while the framework follows MuJoCo's
        # mj_objectVelocity layout [angular, linear].
        geom_xvel = torch.cat(
            (body_velocity[..., 3:6], body_velocity[..., 0:3]),
            dim=-1,
        )
        geom_xpos = torch.cat(
            (
                body_pos,
                torch.zeros(
                    (self.num_envs, len(self.floor_prim_paths), 3),
                    dtype=self.context.dtype,
                    device=self.context.device,
                ),
            ),
            dim=1,
        )
        geom_xvel = torch.cat(
            (
                geom_xvel,
                torch.zeros(
                    (self.num_envs, len(self.floor_prim_paths), 6),
                    dtype=self.context.dtype,
                    device=self.context.device,
                ),
            ),
            dim=1,
        )
        contact_ids, contact_forces, foot_contact = self._contact_state()
        measured_effort = self._articulation.get_measured_joint_efforts(
            clone=True
        )
        actuator_force = self._tensor(measured_effort)[
            ..., -len(self.metadata.dof_names):
        ]

        rotation_world_to_body = self._quaternion_inverse_rotate_matrix(root_quat)
        base_lin_vel_body = torch.bmm(
            rotation_world_to_body,
            root_velocity[..., :3].unsqueeze(-1),
        ).squeeze(-1)
        base_ang_vel_body = torch.bmm(
            rotation_world_to_body,
            root_velocity[..., 3:6].unsqueeze(-1),
        ).squeeze(-1)
        return {
            "qpos": qpos,
            "qvel": qvel,
            "qacc": qacc,
            "ctrl": self._ctrl,
            "geom_xpos": geom_xpos,
            "geom_xvel": geom_xvel,
            "actuator_force": actuator_force,
            "base_lin_vel_body": base_lin_vel_body,
            "base_ang_vel_body": base_ang_vel_body,
            "contact_geom_ids": contact_ids,
            "contact_forces": contact_forces,
            "foot_ground_contact": foot_contact,
        }


    def _contact_state(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        matrix = self._body_view.get_contact_force_matrix(
            dt=self.sim_dt,
            clone=True,
        )
        force_vectors = self._tensor(matrix).reshape(
            self.num_envs,
            len(self._body_names),
            len(self.floor_prim_paths),
            3,
        )
        normal_force = force_vectors.norm(dim=-1)
        active = normal_force > 0.0
        body_ids = torch.arange(
            len(self._body_names),
            dtype=torch.long,
            device=self.context.device,
        ).view(1, -1, 1).expand(
            self.num_envs,
            -1,
            len(self.floor_prim_paths),
        )
        floor_ids = (
            len(self._body_names)
            + torch.arange(
                len(self.floor_prim_paths),
                dtype=torch.long,
                device=self.context.device,
            )
        ).view(1, 1, -1).expand_as(body_ids)
        contact_ids = torch.stack(
            (body_ids, floor_ids),
            dim=-1,
        ).reshape(self.num_envs, -1, 2)
        contact_ids = torch.where(
            active.reshape(self.num_envs, -1, 1),
            contact_ids,
            torch.full_like(contact_ids, -1),
        )
        contact_forces = torch.zeros(
            (
                self.num_envs,
                len(self._body_names) * len(self.floor_prim_paths),
                6,
            ),
            dtype=self.context.dtype,
            device=self.context.device,
        )
        contact_forces[..., 0] = normal_force.reshape(self.num_envs, -1)
        foot_ids = torch.tensor(
            [
                self._body_prim_paths.index(path)
                for path in self.metadata.foot_body_prim_paths
            ],
            dtype=torch.long,
            device=self.context.device,
        )
        foot_contact = (
            body_ids.reshape(self.num_envs, -1, 1)
            == foot_ids.view(1, 1, -1)
        ) & (
            normal_force.reshape(self.num_envs, -1)
            >= self.foot_contact_force_threshold
        ).unsqueeze(-1)
        return contact_ids, contact_forces, foot_contact


    @staticmethod
    def _quaternion_inverse_rotate_matrix(quaternion: torch.Tensor) -> torch.Tensor:

        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(
            torch.finfo(quaternion.dtype).eps
        )
        w, x, y, z = quaternion.unbind(dim=-1)
        return torch.stack(
            (
                1 - 2 * (y * y + z * z),
                2 * (x * y + w * z),
                2 * (x * z - w * y),
                2 * (x * y - w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z + w * x),
                2 * (x * z + w * y),
                2 * (y * z - w * x),
                1 - 2 * (x * x + y * y),
            ),
            dim=-1,
        ).reshape(-1, 3, 3)


    def render(self) -> np.ndarray | None:

        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            self._app.update()
            return None
        if self._camera is None:
            raise RuntimeError("The Isaac Sim RGB camera was not initialized.")
        self._world.render()
        frame = self._camera.get_rgba()
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        return np.asarray(frame)[..., :3]


    def close(self) -> None:

        if self._closed:
            return
        if self._world is not None:
            self._world.stop()
            self._world.clear()
        if self._app is not None:
            self._app.close()
        self._closed = True


    def _indices(
        self,
        env_ids: torch.Tensor | None
    ) -> torch.Tensor:
        
        if env_ids is None:
            return torch.arange(
                self.num_envs,
                dtype=torch.long,
                device=self.context.device,
            )
        return env_ids.to(device=self.context.device, dtype=torch.long)


    def _tensor(
        self,
        value: Any
    ) -> torch.Tensor:
        
        if isinstance(value, torch.Tensor):
            return value.to(
                device=self.context.device,
                dtype=self.context.dtype,
            )
        return torch.as_tensor(
            value,
            dtype=self.context.dtype,
            device=self.context.device,
        )
