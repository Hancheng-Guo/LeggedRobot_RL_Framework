from __future__ import annotations

import gc
import logging
import math
import os
import sys
import threading

import torch
import numpy as np
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .isaac_sim_backend import DEFAULT_TERRAIN_LAYOUT, IsaacSimModelMetadata, IsaacSimResetState
from .isaac_sim_friction import IsaacSimFriction
from .isaac_sim_model import IsaacSimModelConverter
from .isaac_sim_terrain import IsaacSimTerrain, measure_spawn_geometry
from .terrain import TerrainGeometry, TerrainPatch, TerrainRegionKey, TerrainSpawnGeometry
from utils import RuntimeContext, get_logger


LOGGER = get_logger("isaac_sim")


def _requires_windows_shutdown_workaround() -> bool:

    if sys.platform != "win32":
        return False

    try:
        return version("isaacsim").startswith("6.1.")
    except PackageNotFoundError:
        return False


def _keep_windows_kit_libraries_loaded() -> None:

    if sys.platform != "win32":
        return

    import ctypes
    from ctypes import wintypes

    # Kit shuts down these plugins before some SDK Python bindings release
    # their native references. Keep the DLLs mapped through Python teardown.
    pin_module = ctypes.WinDLL("kernel32", use_last_error=True).GetModuleHandleExW
    pin_module.argtypes = [
        wintypes.DWORD, wintypes.LPCWSTR, ctypes.POINTER(wintypes.HMODULE)
    ]
    pin_module.restype = wintypes.BOOL
    pin_until_process_exit = 0x1  # GET_MODULE_HANDLE_EX_FLAG_PIN
    for name in (
        "omni.appwindow.plugin.dll",
        "omni.timeline.plugin.dll",
        "carb.events.plugin.dll",
        "carb.variant.plugin.dll",
        "omni.stageupdate.plugin.dll",
    ):
        handle = wintypes.HMODULE()
        if not pin_module(pin_until_process_exit, name, ctypes.byref(handle)):
            error = ctypes.get_last_error()
            if error != 126:  # ERROR_MOD_NOT_FOUND
                raise ctypes.WinError(error)


@dataclass
class _IsaacResetSnapshot:

    terrain: IsaacSimTerrain | None
    indices: torch.Tensor
    regions: torch.Tensor | None
    positions: torch.Tensor
    orientations: torch.Tensor
    linear_velocities: torch.Tensor
    angular_velocities: torch.Tensor
    joint_positions: torch.Tensor
    joint_velocities: torch.Tensor
    targets: torch.Tensor
    previous_qvel: torch.Tensor | None
    contact_stale: torch.Tensor


@dataclass
class _IsaacFrictionResetSnapshot:
    materials: object
    articulation: _IsaacResetSnapshot


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
        self._robot_rays = None
        self._body_view: Any = None
        self._camera: Any = None
        self.render_mode: str | None = None
        self._camera_env_index: int | None = None
        self._playback_restore: tuple[str | None, bool, bool, bool] | None = None
        self._simulation_manager: Any = None
        self._kit_logging: Any = None
        self._kit_logger_handle: Any = None
        self._kit_log_callback: Any = None
        self._kit_log_forwarding = threading.local()
        self._kit_log_levels: dict[int, int] = {}
        self._closed = False
        self._terrain: IsaacSimTerrain | None = None
        self._friction: IsaacSimFriction | None = None
        self._friction_requested = False
        self._windows_shutdown_workaround = False


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
        terrain_geometry: TerrainGeometry | None = None,
        terrain_layout: tuple[int, tuple[int, int], float] = DEFAULT_TERRAIN_LAYOUT,
        friction_enabled: bool = False,
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
        self._friction_requested = friction_enabled
        if terrain_geometry is not None and floor_prim_paths:
            raise ValueError("Isaac shared terrain requires empty floor_prim_paths.")

        self.base_body_prim_path: str
        self.foot_body_prim_paths: tuple[str, ...]
        self.floor_prim_paths: tuple[str, ...]
        self.camera_prim_path: str | None = None

        self._start_application()
        if terrain_geometry is not None:
            self._terrain = IsaacSimTerrain(
                terrain_geometry,
                self.num_envs,
                self.context.seed,
                terrain_layout,
                self.context.device,
            )
        converter = IsaacSimModelConverter(
            ros_package_paths=ros_package_paths,
            merge_fixed_joints=merge_fixed_joints,
            allow_self_collision=allow_self_collision,
            joint_stiffness=joint_stiffness,
            joint_damping=joint_damping,
        )
        self.model_path = converter.convert_if_needed(self.model_path)
        self._build_scene()
        self._build_metadata()
        self._prepare_state_buffers()
        self._capture_default_state()
        self._ctrl = self.metadata.actuator_default_ctrl.repeat(
            self.num_envs,
            1,
        )
        self._previous_qvel: torch.Tensor | None = None
        self._last_frame_skip = 1
        if self._terrain is not None:
            self.reset(self._all_env_indices[:1])
            self._terrain_spawn_geometry = self._measure_spawn_geometry()


    def _start_application(self) -> None:

        try:
            from isaacsim.simulation_app import SimulationApp  # pyright: ignore[reportMissingImports]
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Isaac Sim 6.1 is required by the isaac_sim backend."
            ) from error

        headless = self.render_mode != "human"
        LOGGER.info("Starting Isaac Sim runtime.")
        extra_args = [
            "--/app/enableStdoutOutput=false",
            "--/app/python/logSysStdOutput=false",
            "--/log/enableStandardStreamOutput=false",
        ]
        self._windows_shutdown_workaround = _requires_windows_shutdown_workaround()
        launcher_kwargs = {}
        if self._windows_shutdown_workaround:
            # The default 6.1 experience loads extensions which retain PhysX
            # Python callbacks beyond interpreter shutdown on Windows.
            launcher_kwargs["experience"] = str(Path(__file__).with_name("isaac_sim_windows.kit"))
            sdk_path = Path(os.environ["ISAAC_PATH"])
            extra_args.extend([
                "--/app/python/interceptSysStdOutput=false",
                "--ext-folder", str(sdk_path / "extscache"),
                "--ext-folder", str(sdk_path / "extsDeprecated"),
            ])
            if self.render_mode == "human":
                for extension in (
                    "omni.kit.window.stage",
                    "omni.kit.window.property",
                    "omni.kit.viewport.menubar.camera",
                    "omni.kit.viewport.menubar.display",
                    "omni.kit.viewport.menubar.lighting",
                    "omni.kit.viewport.menubar.render",
                    "omni.kit.viewport.menubar.settings",
                ):
                    extra_args.extend(("--enable", extension))
        if self.render_mode == "rgb_array":
            # A standalone camera does not benefit from deferred texture
            # residency, and the streaming client can stall Kit teardown after
            # its render product has been destroyed.
            extra_args.append(
                "--/rtx-transient/resourcemanager/"
                "texturestreaming/enabled=false"
            )
        self._start_log_bridge()
        try:
            with open(os.devnull, "w", encoding="utf-8") as output_sink:
                with redirect_stdout(output_sink):
                    self._app = SimulationApp(
                        {
                            "headless": headless,
                            # Kit's quick-shutdown path can finalize Python-backed
                            # extensions after their native state is already gone.
                            # Use orderly extension teardown for this embedded app.
                            "fast_shutdown": False,
                            "extra_args": extra_args,
                        },
                        **launcher_kwargs,
                    )
        except BaseException:
            self._stop_log_bridge()
            raise
        LOGGER.info("Isaac Sim runtime initialized.")
        import carb.settings  # pyright: ignore[reportMissingImports]

        fabric_output_enabled = self.render_mode is not None
        settings = carb.settings.get_settings()
        settings.set_bool("/physics/fabricUpdateTransformations", fabric_output_enabled)
        settings.set_bool("/physics/fabricUpdateVelocities", fabric_output_enabled)
        LOGGER.info(
            "Fabric transform and velocity output %s for render mode %s.",
            "enabled" if fabric_output_enabled else "disabled",
            self.render_mode,
        )


    def _start_log_bridge(self) -> None:
        """Forward runtime Kit warnings and errors through project logging."""

        import carb.logging  # pyright: ignore[reportMissingImports]

        self._kit_log_levels = {
            carb.logging.LEVEL_WARN: logging.WARNING,
            carb.logging.LEVEL_ERROR: logging.ERROR,
            carb.logging.LEVEL_FATAL: logging.CRITICAL,
        }
        self._kit_logging = carb.logging.acquire_logging()
        self._kit_log_callback = self._forward_kit_log
        self._kit_logger_handle = self._kit_logging.add_logger(
            self._kit_log_callback
        )


    def _forward_kit_log(
        self,
        source: str,
        level: int,
        filename: str,
        line_number: int,
        message: str,
    ) -> None:
        """Send selected Kit records to the configured project handlers."""

        project_level = self._kit_log_levels.get(level)
        if project_level is None:
            return
        if getattr(self._kit_log_forwarding, "active", False):
            return
        self._kit_log_forwarding.active = True
        try:
            LOGGER.log(
                project_level,
                "[Isaac Sim: %s] %s",
                source,
                message.rstrip(),
            )
        finally:
            self._kit_log_forwarding.active = False


    def _stop_log_bridge(self) -> None:
        """Stop forwarding Kit records before Kit shuts down."""

        if self._kit_logging is None or self._kit_logger_handle is None:
            return
        self._kit_logging.remove_logger(self._kit_logger_handle)
        self._kit_logger_handle = None
        self._kit_log_callback = None
        self._kit_logging = None


    def _build_scene(self) -> None:
        # Isaac/Omniverse modules must be imported after SimulationApp starts.
        from isaacsim.core.api import World  # pyright: ignore[reportMissingImports]
        from isaacsim.core.cloner import GridCloner  # pyright: ignore[reportMissingImports]
        from isaacsim.core.experimental.prims import Articulation, RigidPrim  # pyright: ignore[reportMissingImports]
        from isaacsim.core.simulation_manager import SimulationManager  # pyright: ignore[reportMissingImports]
        from isaacsim.core.utils.stage import add_reference_to_stage  # pyright: ignore[reportMissingImports]
        from pxr import Usd, UsdGeom, UsdLux, UsdPhysics

        self._simulation_manager = SimulationManager
        self._world = World(
            physics_dt=self.sim_dt,
            rendering_dt=self.sim_dt,
            backend="torch",
            device=str(self.context.device),
            physics_prim_path="/World/physicsScene",
        )
        physics_context = self._world.get_physics_context()
        found_lost_capacity = max(1024, self.num_envs * 64)
        if self._windows_shutdown_workaround and self._terrain is not None:
            # Shared terrain can bring robots together. PhysX discovers those
            # overlaps before the USD collision groups reject cross-env pairs.
            env_pairs = self.num_envs * (self.num_envs - 1) // 2
            total_capacity = found_lost_capacity + env_pairs
            if (
                physics_context.get_gpu_total_aggregate_pairs_capacity()
                < total_capacity
            ):
                physics_context.set_gpu_total_aggregate_pairs_capacity(total_capacity)
            # Batched resets move overlapping articulations in one step. The
            # GPU broadphase can report several found/lost records per env pair.
            found_lost_capacity += 8 * env_pairs
        if (
            physics_context.get_gpu_found_lost_aggregate_pairs_capacity()
            < found_lost_capacity
        ):
            physics_context.set_gpu_found_lost_aggregate_pairs_capacity(found_lost_capacity)
        stage = self._world.stage
        source_env_path = "/World/envs/env_0"
        source_robot_path = source_env_path + self._normalized_robot_prim_path()
        self._source_robot_path = source_robot_path
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

        if self._terrain is not None:
            self._terrain.build(stage)
            self.floor_prim_paths = (self._terrain.prim_path,)
            light: Any = UsdLux.DomeLight.Define(stage, "/World/TerrainSkyLight")
            light.CreateIntensityAttr(500.)
        elif self._requested_floor_prim_paths:
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
        # On Windows 6.1 the native replicator retains Python callbacks until
        # DLL detach. USD clones still use the collision groups below.
        replicate_physics = not self._windows_shutdown_workaround
        cloner.clone(
            source_prim_path=source_env_path,
            prim_paths=env_paths,
            replicate_physics=replicate_physics,
            base_env_path="/World/envs",
            root_path="/World/envs/env_",
            enable_env_ids=replicate_physics,
            # The legacy Articulation/RigidPrim views discover their default
            # states from USD. Fabric-only clones leave those defaults at one
            # row while the PhysX view contains every environment.
            clone_in_fabric=False,
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
        # The legacy isaacsim.core.prims.Articulation crashes during native
        # interpreter teardown on Isaac Sim 6.1. The experimental wrapper is
        # the current implementation and owns its lifecycle subscriptions via
        # weak references, so it can be released cleanly before Kit shuts down.
        self._articulation = Articulation(robot_expression)

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
        self._resolve_foot_body_prim_paths()
        if self._friction_requested:
            self._bind_friction()
        body_paths = [
            f"/World/envs/env_{env_id}{self._normalized_robot_prim_path()}{relative}"
            for env_id in range(self.num_envs)
            for relative in body_relative_paths
        ]
        contact_filters = [
            self._find_floor_collision_prim_path(stage, path, Usd, UsdPhysics)
            for path in self.floor_prim_paths
        ]
        self._body_view = RigidPrim(body_paths, contact_filter_paths=contact_filters)
        # Contact filters alone do not apply PhysxContactReportAPI in the
        # experimental wrapper. Enable it before reset creates tensor views.
        self._body_view.set_enabled_contact_tracking([True])

        if self.render_mode == "rgb_array":
            from isaacsim.sensors.camera import Camera  # pyright: ignore[reportMissingImports]

            self._build_camera(Camera)

        # Initialize every physics-backed view together through the simulation
        # manager callbacks triggered by reset.
        self._world.reset()
        if self._camera is not None:
            self._camera.initialize()


    def _build_camera(self, camera_type: Any) -> None:
        """Create a camera rendered explicitly once per control step."""

        self.camera_prim_path = (
            self._requested_camera_prim_path or "/World/Camera"
        )
        self._camera = camera_type(
            prim_path=self.camera_prim_path,
            position=np.asarray((2.5, 2.5, 1.8)),
            resolution=self.camera_resolution,
        )


    def activate_playback(self, render_mode: str) -> None:
        """Enable RGB playback in the existing headless SimulationApp."""

        if self._playback_restore is not None:
            raise RuntimeError(
                "Previous Isaac playback cleanup is incomplete; "
                "call terminate_playback() before starting playback again."
            )
        if render_mode == "human":
            raise RuntimeError(
                "Cannot use render_mode='human' in the running headless Isaac Sim "
                "application. Start a separate non-headless process for human playback."
            )
        if render_mode != "rgb_array":
            raise ValueError(f"Unsupported Isaac playback mode: {render_mode!r}.")
        if self.render_mode == "rgb_array":
            return

        import carb.settings  # pyright: ignore[reportMissingImports]
        from isaacsim.sensors.camera import Camera  # pyright: ignore[reportMissingImports]

        settings = carb.settings.get_settings()
        self._playback_restore = (
            self.render_mode,
            bool(settings.get("/physics/fabricUpdateTransformations")),
            bool(settings.get("/physics/fabricUpdateVelocities")),
            self._world.stage.GetPrimAtPath(
                self._requested_camera_prim_path or "/World/Camera"
            ).IsValid(),
        )
        try:
            settings.set_bool("/physics/fabricUpdateTransformations", True)
            settings.set_bool("/physics/fabricUpdateVelocities", True)
            self._build_camera(Camera)
            self._camera.initialize()
            self._camera_env_index = int(
                self._default_root_positions[:, :2].square().sum(dim=-1).argmin().item()
            )
            self.render_mode = "rgb_array"
        except BaseException:
            self.terminate_playback()
            raise
        LOGGER.info("Enabled Fabric output and RGB camera for Isaac playback.")


    def terminate_playback(self) -> None:

        restore = self._playback_restore
        if restore is None:
            return

        import carb.settings  # pyright: ignore[reportMissingImports]

        previous_mode, transformations, velocities, camera_prim_existed = restore
        try:
            if self._camera is not None:
                self._camera.destroy()
                self._camera = None
                self._app.update()
                gc.collect()
            if self.camera_prim_path and not camera_prim_existed:
                self._world.stage.RemovePrim(self.camera_prim_path)
            self.camera_prim_path = None
            self._camera_env_index = None
        finally:
            settings = carb.settings.get_settings()
            settings.set_bool("/physics/fabricUpdateTransformations", transformations)
            settings.set_bool("/physics/fabricUpdateVelocities", velocities)
            self.render_mode = previous_mode
        self._playback_restore = None
        LOGGER.info("Restored Isaac training camera and Fabric settings after playback.")


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


    @staticmethod
    def _find_floor_collision_prim_path(
        stage: Any,
        floor_path: str,
        usd: Any,
        usd_physics: Any,
    ) -> str:
        floor_prim = stage.GetPrimAtPath(floor_path)
        collision_paths = [
            str(prim.GetPath())
            for prim in usd.PrimRange(floor_prim)
            if prim.HasAPI(usd_physics.CollisionAPI)
        ]
        if not collision_paths:
            raise ValueError(
                f"Floor Prim {floor_path!r} contains no collision Prim."
            )
        if len(collision_paths) > 1:
            raise ValueError(
                f"Floor Prim {floor_path!r} must contain exactly one "
                f"collision Prim, found {list(collision_paths)}."
            )
        return collision_paths[0]


    def prepare_friction(self) -> None:
        if self._friction is not None:
            return
        self._world.stop()
        try:
            self._bind_friction()
        finally:
            self._world.reset()


    def disable_friction(self) -> None:
        if self._friction is None:
            return
        self._world.stop()
        try:
            self._friction.unbind()
            self._friction = None
        finally:
            self._world.reset()


    def _bind_friction(self) -> None:

        from pxr import Usd, UsdPhysics

        stage = self._world.stage
        floor_paths = [
            self._find_floor_collision_prim_path(stage, path, Usd, UsdPhysics)
            for path in self.floor_prim_paths
        ]
        robot_path = self._normalized_robot_prim_path()
        foot_paths_by_env = [
            [
                f"/World/envs/env_{env_id}{robot_path}{path}"
                for path in self.foot_body_prim_paths
            ]
            for env_id in range(self.num_envs)
        ]
        self._friction = IsaacSimFriction(stage, floor_paths, foot_paths_by_env)


    def capture_friction(
        self,
        env_ids: Sequence[int],
    ) -> object:
        indices = torch.tensor(env_ids, dtype=torch.long, device=self.context.device)
        return _IsaacFrictionResetSnapshot(
            materials=self._require_friction().capture(env_ids),
            articulation=self._capture_articulation_state(indices),
        )


    def apply_friction(
        self,
        env_ids: Sequence[int],
        factors: Sequence[float],
    ) -> None:
        self._require_friction().apply(env_ids, factors)


    def restore_friction(
        self,
        snapshot: object,
    ) -> None:
        if not isinstance(snapshot, _IsaacFrictionResetSnapshot):
            raise TypeError("Invalid Isaac friction reset snapshot.")
        self._require_friction().restore(snapshot.materials)
        self._restore_articulation_state(snapshot.articulation)


    def _require_friction(self) -> IsaacSimFriction:
        if self._friction is None:
            raise RuntimeError("Prepare Isaac friction before updating it.")
        return self._friction


    def _resolve_foot_body_prim_paths(self) -> None:
        self.foot_body_prim_paths = self._requested_foot_body_prim_paths or tuple(
            path for path in self._body_prim_paths
            if path.rsplit("/", 1)[-1].lower().endswith("foot")
        )
        if not self.foot_body_prim_paths:
            raise ValueError(
                "No foot bodies were inferred; configure 'foot_body_prim_paths'."
            )


    def _build_metadata(self) -> None:

        dof_names = tuple(self._articulation.dof_names)
        joint_default_pos = self._tensor(
            self._articulation.get_dof_positions()[0]
        )
        lower_limits, upper_limits = self._articulation.get_dof_limits()
        joint_pos_limits = torch.stack(
            (self._tensor(lower_limits)[0], self._tensor(upper_limits)[0]),
            dim=-1,
        )
        self.base_body_prim_path = (
            self._requested_base_body_prim_path
            or self._body_prim_paths[0]
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

        root_positions, root_orientations = self._articulation.get_world_poses()
        # Experimental getters may expose live PhysX/Warp storage. Reset poses
        # must own their memory so stepping cannot overwrite the defaults.
        self._default_root_positions = self._tensor(root_positions).clone()
        self._default_root_orientations = self._tensor(root_orientations).clone()
        if self._camera is not None:
            self._camera_env_index = int(
                self._default_root_positions[:, :2]
                .square()
                .sum(dim=-1)
                .argmin()
                .item()
            )
        self._default_joint_positions = self._tensor(
            self._articulation.get_dof_positions()
        ).clone()
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


    def _prepare_state_buffers(self) -> None:
        """Precompute immutable tensor layouts used by every state query."""

        self._body_count = len(self._body_names)
        self._floor_count = len(self.floor_prim_paths)
        self._all_env_indices = torch.arange(
            self.num_envs,
            dtype=torch.long,
            device=self.context.device,
        )
        self._floor_geom_xpos = torch.zeros(
            (self.num_envs, self._floor_count, 3),
            dtype=self.context.dtype,
            device=self.context.device,
        )
        self._floor_geom_xvel = torch.zeros(
            (self.num_envs, self._floor_count, 6),
            dtype=self.context.dtype,
            device=self.context.device,
        )
        self._contact_stale = torch.zeros_like(
            self._all_env_indices, dtype=torch.bool
        )

        body_ids = torch.arange(
            self._body_count,
            dtype=torch.long,
            device=self.context.device,
        ).view(1, -1, 1).expand(
            self.num_envs,
            -1,
            self._floor_count,
        )
        floor_ids = (
            self._body_count
            + torch.arange(
                self._floor_count,
                dtype=torch.long,
                device=self.context.device,
            )
        ).view(1, 1, -1).expand_as(body_ids)
        self._contact_geom_pairs = torch.stack(
            (body_ids, floor_ids),
            dim=-1,
        ).reshape(self.num_envs, -1, 2)

        foot_ids = torch.tensor(
            [
                self._body_prim_paths.index(path)
                for path in self.foot_body_prim_paths
            ],
            dtype=torch.long,
            device=self.context.device,
        )
        self._foot_contact_mask = (
            body_ids.reshape(self.num_envs, -1, 1)
            == foot_ids.view(1, 1, -1)
        )
        self._foot_body_ids = foot_ids


    def _require_terrain(self) -> IsaacSimTerrain:
        if self._terrain is None:
            raise ValueError(
                "Isaac terrain collision geometry is not configured."
            )
        return self._terrain


    def resolve_scan_mounts(
        self,
        names: tuple[str, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        from pxr import Gf, UsdGeom

        stage = self._world.stage
        base_path = self._source_robot_path + self.base_body_prim_path
        base = stage.GetPrimAtPath(base_path)
        cache: Any = UsdGeom.XformCache()
        positions, axes = [], []
        for name in names:
            path = (
                self._source_robot_path + name if name.startswith("/")
                else base_path + "/" + name
            )
            mount = stage.GetPrimAtPath(path)
            if not mount or mount.GetParent() != base or not mount.IsA(UsdGeom.Xform):
                raise ValueError(f"Scan mount {name!r} must be an Xform attached to the base.")
            transform, _ = cache.ComputeRelativeTransform(mount, base)
            positions.append(tuple(transform.ExtractTranslation()))
            axes.append(tuple(transform.TransformDir(Gf.Vec3d(0, 0, 1)).GetNormalized()))
        return self._tensor(np.asarray(positions)), self._tensor(np.asarray(axes))


    def query_robot_rays(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        max_distance: float,
        env_ids: torch.Tensor | None,
    ) -> torch.Tensor:

        from .isaac_sim_robot_rays import IsaacRobotRayQuery

        if self._robot_rays is None:
            self._robot_rays = IsaacRobotRayQuery(
                stage=self._world.stage,
                root=self._source_robot_path,
                paths=self._body_prim_paths,
                device=origins.device,
            )
        positions, orientations = self._body_view.get_world_poses()
        positions = self._tensor(positions).reshape(self.num_envs, -1, 3)
        orientations = self._tensor(orientations).reshape(self.num_envs, -1, 4)
        if env_ids is not None:
            positions, orientations = positions[env_ids], orientations[env_ids]
        return self._robot_rays(
            origins=origins,
            directions=directions,
            limit=max_distance,
            positions=positions,
            orientations=orientations,
        )


    def _measure_spawn_geometry(self) -> TerrainSpawnGeometry:
        positions, orientations = self._body_view.get_world_poses()
        return measure_spawn_geometry(
            self._world.stage,
            self._source_robot_path,
            self._body_prim_paths,
            self._tensor(positions)[:self._body_count].cpu().numpy(),
            self._tensor(orientations)[:self._body_count].cpu().numpy(),
            np.zeros(3),
            self._default_root_positions[0].cpu().numpy(),
        )


    def get_terrain_spawn_geometry(self) -> TerrainSpawnGeometry:
        self._require_terrain()
        return self._terrain_spawn_geometry


    def _terrain_indices(
        self,
        env_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            env_ids is not None
            and (env_ids.ndim != 1 or env_ids.dtype != torch.long)
        ):
            raise ValueError(
                "Environment IDs must be a one-dimensional long tensor."
            )
        indices = self._indices(env_ids)
        if env_ids is not None and (
            indices.unique().numel() != indices.numel()
            or torch.any((indices < 0) | (indices >= self.num_envs))
        ):
            raise ValueError("Environment IDs must be unique and in range.")
        return indices


    def get_ground_bounds(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        return (
            None if self._terrain is None
            else self._terrain.bounds(self._terrain_indices(env_ids))
        )


    def get_ground_heights(
        self,
        xy: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        indices = self._terrain_indices(env_ids)
        if (
            xy.ndim != 3
            or xy.shape[0] != indices.numel()
            or xy.shape[-1] != 2
        ):
            raise ValueError(
                "Ground query rows must match environments with shape [env, sample, 2]."
            )
        if not xy.is_floating_point() or not torch.isfinite(xy).all():
            raise ValueError(
                "Ground queries must contain finite floating-point coordinates."
            )
        return self._require_terrain().heights(xy.to(self.context.device)).to(xy)


    def get_terrain_layout(self) -> tuple[int, tuple[int, int], float]:
        return self._require_terrain().layout


    def get_terrain_patch(
        self,
        key: TerrainRegionKey,
    ) -> TerrainPatch:
        return self._require_terrain().get_patch(key)


    def select_terrain(
        self,
        env_ids_tensor: torch.Tensor,
        keys: Sequence[TerrainRegionKey],
    ) -> None:
        indices = self._terrain_indices(env_ids_tensor)
        self._require_terrain().select(indices, keys)


    def _update_articulation_kinematics(self) -> None:
        self._simulation_manager.get_physics_simulation_view().update_articulations_kinematic()


    def set_root_positions(
        self,
        env_ids_tensor: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        indices = self._terrain_indices(env_ids_tensor)
        if (
            positions.shape != (indices.numel(), 3)
            or not positions.is_floating_point()
            or not torch.isfinite(positions).all()
        ):
            raise ValueError(
                "Root positions must be finite floating-point data with shape [env, 3]."
            )
        self._articulation.set_world_poses(
            positions=self._warp(self._tensor(positions)),
            indices=self._warp(indices),
        )
        self._contact_stale[indices] = True
        self._update_articulation_kinematics()


    def capture_reset_state(
        self,
        env_ids_tensor: torch.Tensor,
    ) -> object:
        indices = self._terrain_indices(env_ids_tensor)
        return self._capture_articulation_state(indices, self._require_terrain())


    def _capture_articulation_state(
        self,
        indices: torch.Tensor,
        terrain: IsaacSimTerrain | None = None,
    ) -> _IsaacResetSnapshot:
        positions, orientations = self._articulation.get_world_poses()
        linear, angular = self._articulation.get_velocities()
        return _IsaacResetSnapshot(
            terrain=terrain, indices=indices.clone(),
            regions=None if terrain is None else terrain.region_ids[indices],
            positions=self._tensor(positions)[indices],
            orientations=self._tensor(orientations)[indices],
            linear_velocities=self._tensor(linear)[indices],
            angular_velocities=self._tensor(angular)[indices],
            joint_positions=self._tensor(self._articulation.get_dof_positions())[indices],
            joint_velocities=self._tensor(self._articulation.get_dof_velocities())[indices],
            targets=self._ctrl[indices],
            previous_qvel=None if self._previous_qvel is None else self._previous_qvel.clone(),
            contact_stale=self._contact_stale[indices],
        )


    def restore_reset_state(
        self,
        state: object,
    ) -> None:
        terrain = self._require_terrain()
        if (
            not isinstance(state, _IsaacResetSnapshot)
            or state.terrain is not terrain
            or state.regions is None
        ):
            raise ValueError("Reset snapshot belongs to a different Isaac terrain scene.")
        terrain.region_ids[state.indices] = state.regions
        self._restore_articulation_state(state)


    def _restore_articulation_state(
        self,
        state: _IsaacResetSnapshot,
    ) -> None:
        indices = self._warp(state.indices)
        self._articulation.set_world_poses(
            positions=self._warp(state.positions),
            orientations=self._warp(state.orientations), indices=indices,
        )
        self._articulation.set_velocities(
            linear_velocities=self._warp(state.linear_velocities),
            angular_velocities=self._warp(state.angular_velocities), indices=indices,
        )
        self._articulation.set_dof_positions(self._warp(state.joint_positions), indices=indices)
        self._articulation.set_dof_velocities(self._warp(state.joint_velocities), indices=indices)
        self._articulation.set_dof_position_targets(self._warp(state.targets), indices=indices)
        self._ctrl[state.indices] = state.targets
        self._previous_qvel = state.previous_qvel
        self._contact_stale[state.indices] = state.contact_stale
        self._update_articulation_kinematics()


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        
        indices = self._indices(env_ids)
        self._articulation.set_world_poses(
            positions=self._warp(self._default_root_positions[indices]),
            orientations=self._warp(self._default_root_orientations[indices]),
            indices=self._warp(indices),
        )
        self._articulation.set_velocities(
            linear_velocities=self._warp(torch.zeros(
                (indices.numel(), 3),
                dtype=self.context.dtype,
                device=self.context.device,
            )),
            angular_velocities=self._warp(torch.zeros(
                (indices.numel(), 3),
                dtype=self.context.dtype,
                device=self.context.device,
            )),
            indices=self._warp(indices),
        )
        self._articulation.set_dof_positions(
            self._warp(self._default_joint_positions[indices]),
            indices=self._warp(indices),
        )
        self._articulation.set_dof_velocities(
            self._warp(torch.zeros_like(self._default_joint_positions[indices])),
            indices=self._warp(indices),
        )
        self._articulation.set_dof_position_targets(
            self._warp(self._default_joint_positions[indices]),
            indices=self._warp(indices),
        )
        self._ctrl[indices] = self._default_joint_positions[indices]
        if self._terrain is not None:
            self._contact_stale[indices] = True
        # Reset readers may query link poses before the next physics step.
        self._update_articulation_kinematics()
        if env_ids is None:
            self._previous_qvel = None
        elif self._previous_qvel is not None:
            self._previous_qvel[indices] = 0.


    def step(
        self,
        action: torch.Tensor,
        frame_skip: int
    ) -> None:
        
        targets = action.to(device=self.context.device, dtype=self.context.dtype)
        self._articulation.set_dof_position_targets(self._warp(targets))
        self._ctrl.copy_(targets)
        self._last_frame_skip = frame_skip
        for _ in range(frame_skip):
            self._world.step(render=self.render_mode == "human")
        self._contact_stale.zero_()


    def get_state(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        
        state = self._get_full_state(env_ids)
        if env_ids is None:
            return state
        indices = env_ids.to(device=self.context.device, dtype=torch.long)
        return {name: value[indices] for name, value in state.items()}


    def _get_full_state(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:

        root_pos, root_quat = self._articulation.get_world_poses()
        root_linear_velocity, root_angular_velocity = (
            self._articulation.get_velocities()
        )
        joint_pos = self._articulation.get_dof_positions()
        joint_vel = self._articulation.get_dof_velocities()
        root_pos = self._tensor(root_pos)
        root_quat = self._tensor(root_quat)
        root_velocity = torch.cat(
            (
                self._tensor(root_linear_velocity),
                self._tensor(root_angular_velocity),
            ),
            dim=-1,
        )
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
        if env_ids is None or self._previous_qvel is None:
            self._previous_qvel = qvel.clone()
        else:
            indices = self._indices(env_ids)
            self._previous_qvel[indices] = qvel[indices]

        body_pos, _ = self._body_view.get_world_poses()
        body_linear_velocity, body_angular_velocity = (
            self._body_view.get_velocities()
        )
        body_velocity = torch.cat(
            (
                self._tensor(body_linear_velocity),
                self._tensor(body_angular_velocity),
            ),
            dim=-1,
        )
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
                self._floor_geom_xpos,
            ),
            dim=1,
        )
        geom_xvel = torch.cat(
            (
                geom_xvel,
                self._floor_geom_xvel,
            ),
            dim=1,
        )
        contact_ids, contact_forces, foot_contact, foot_contact_force = (
            self._contact_state()
        )
        actuator_force = self._tensor(
            self._articulation.get_dof_projected_joint_forces()
        )

        rotation_world_to_body = self._quaternion_inverse_rotate_matrix(root_quat)
        base_lin_vel_body = torch.bmm(
            rotation_world_to_body,
            root_velocity[..., :3].unsqueeze(-1),
        ).squeeze(-1)
        base_ang_vel_body = torch.bmm(
            rotation_world_to_body,
            root_velocity[..., 3:6].unsqueeze(-1),
        ).squeeze(-1)
        state = {
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
            "foot_contact_normal_force": foot_contact_force,
        }
        if self._terrain is not None:
            terrain_xy = torch.cat(
                (root_pos[:, None, :2], body_pos[:, self._foot_body_ids, :2]),
                dim=1
            )
            state.update(self._terrain.ground_state(terrain_xy, self._all_env_indices))
        return state


    def _contact_state(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        matrix = self._body_view.get_contact_force_matrix(
            dt=self.sim_dt,
        )
        force_vectors = self._tensor(matrix).reshape(
            self.num_envs,
            self._body_count,
            self._floor_count,
            3,
        )
        normal_force = force_vectors.norm(dim=-1)
        if self._terrain is not None:
            # PhysX reports the preceding simulation step until the next step.
            normal_force[self._contact_stale] = 0.
        contact_ids = torch.where(
            (normal_force > 0.0).reshape(self.num_envs, -1, 1),
            self._contact_geom_pairs,
            -1,
        )
        contact_forces = torch.zeros(
            (
                self.num_envs,
                self._body_count * self._floor_count,
                6,
            ),
            dtype=self.context.dtype,
            device=self.context.device,
        )
        contact_forces[..., 0] = normal_force.reshape(self.num_envs, -1)
        valid_foot_contact = (
            self._foot_contact_mask
            & (
                (normal_force.reshape(self.num_envs, -1) > 0.0)
                & (
                    normal_force.reshape(self.num_envs, -1)
                    >= self.foot_contact_force_threshold
                )
            ).unsqueeze(-1)
        )
        foot_contact = valid_foot_contact.any(dim=1)
        foot_contact_force = (
            normal_force.reshape(self.num_envs, -1, 1) * valid_foot_contact
        ).sum(dim=1)
        return contact_ids, contact_forces, foot_contact, foot_contact_force


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
        self._update_camera_pose()
        self._world.render()
        frame = self._camera.get_rgba()
        if frame is None:
            return None
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        frame_array = np.asarray(frame)
        if (
            frame_array.ndim != 3
            or frame_array.shape[-1] < 3
            or frame_array.size == 0
        ):
            return None
        if frame_array.shape[-1] == 4 and not frame_array[..., 3].any():
            return None
        return frame_array[..., :3]


    @property
    def playback_env_index(self) -> int:
        return self._camera_env_index or 0


    def _update_camera_pose(self) -> None:
        """Follow the playback environment from an elevated front-right viewpoint."""

        if getattr(self, "_camera_env_index", None) is None:
            return
        root_positions, _ = self._articulation.get_world_poses()
        target = self._tensor(root_positions)[self.playback_env_index]
        target_array = target.detach().cpu().numpy()

        forward_offset = 2.5
        right_offset = -2.5
        follow_height = 1.8
        horizontal_distance = math.hypot(forward_offset, right_offset)
        yaw = math.atan2(-right_offset, -forward_offset)
        pitch = math.atan2(follow_height, horizontal_distance)
        camera_position = target_array + np.asarray(
            (forward_offset, right_offset, follow_height),
            dtype=target_array.dtype,
        )
        half_yaw = yaw / 2.0
        half_pitch = pitch / 2.0
        camera_orientation = np.asarray(
            (
                math.cos(half_yaw) * math.cos(half_pitch),
                -math.sin(half_yaw) * math.sin(half_pitch),
                math.cos(half_yaw) * math.sin(half_pitch),
                math.sin(half_yaw) * math.cos(half_pitch),
            ),
            dtype=target_array.dtype,
        )
        self._camera.set_world_pose(
            position=camera_position,
            orientation=camera_orientation,
            camera_axes="world",
        )


    def close(self) -> None:

        if self._closed:
            return
        self._robot_rays = None
        failures: list[Exception] = []

        def attempt(operation: Any) -> None:
            try:
                operation()
            except Exception as error:
                failures.append(error)

        if self._camera is not None:
            attempt(self._camera.destroy)
            self._camera = None
            # Render-product and annotator destruction is applied through Kit's
            # update loop. Flush it while Replicator extensions are still alive,
            # then finalize Python wrappers before extension teardown begins.
            if self._app is not None:
                attempt(self._app.update)
            gc.collect()
        if self._world is not None:
            attempt(self._world.stop)
            self._body_view = None
            self._articulation = None
            self._terrain = None
            self._friction = None
            simulation_manager = getattr(self, "_simulation_manager", None)
            if simulation_manager is not None:
                attempt(simulation_manager.invalidate_physics)
            attempt(self._world.clear)
            self._world = None
            self._simulation_manager = None
            # Finalize the experimental views and their weak lifecycle
            # subscriptions while Kit and PhysX modules are still loaded.
            gc.collect()
        if self._app is not None:
            attempt(self._stop_log_bridge)
            if self._windows_shutdown_workaround:
                if self.render_mode == "human":
                    attempt(self._shutdown_app_window)
                attempt(_keep_windows_kit_libraries_loaded)
            # Camera render products have already been released, so there is no
            # Replicator work to drain. Avoid its update loop while retaining the
            # orderly Kit teardown selected by fast_shutdown=False.
            attempt(lambda: self._app.close(wait_for_replicator=False))
            self._app = None
        self._closed = True
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ExceptionGroup(
                "Multiple errors occurred while closing Isaac Sim.",
                failures,
            )


    @staticmethod
    def _shutdown_app_window() -> None:

        from omni.appwindow import get_default_app_window  # pyright: ignore[reportMissingImports]

        window = get_default_app_window()
        if window is not None:
            # Release input and windowing resources while their interfaces are
            # alive, even if external Python code still holds the window.
            window.shutdown()


    def _indices(
        self,
        env_ids: torch.Tensor | None
    ) -> torch.Tensor:
        
        if env_ids is None:
            return self._all_env_indices
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
        if type(value).__module__.startswith("warp."):
            import warp as wp

            return wp.to_torch(value).to(
                device=self.context.device,
                dtype=self.context.dtype,
            )
        return torch.as_tensor(
            value,
            dtype=self.context.dtype,
            device=self.context.device,
        )


    @staticmethod
    def _warp(value: torch.Tensor) -> Any:
        """Expose a contiguous Torch tensor to Isaac's Warp-based API."""

        import warp as wp

        return wp.from_torch(value.contiguous())
