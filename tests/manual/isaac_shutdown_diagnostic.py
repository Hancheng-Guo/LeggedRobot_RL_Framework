from __future__ import annotations

import argparse
import gc
import logging
import sys
import threading
from pathlib import Path
from typing import Any

from isaacsim.simulation_app import SimulationApp


LOGGER = logging.getLogger("isaac_shutdown_diagnostic")
SCENE_MODES = ("model", "cloner", "articulation", "contacts")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class KitLogBridge:
    """Small replica of the project's Kit-to-Python logging bridge."""

    def __init__(self) -> None:
        self._logging: Any = None
        self._handle: Any = None
        self._forwarding = threading.local()
        self._levels: dict[int, int] = {}

    def start(self) -> None:
        import carb.logging

        self._levels = {
            carb.logging.LEVEL_WARN: logging.WARNING,
            carb.logging.LEVEL_ERROR: logging.ERROR,
            carb.logging.LEVEL_FATAL: logging.CRITICAL,
        }
        self._logging = carb.logging.acquire_logging()
        self._handle = self._logging.add_logger(self._forward)
        LOGGER.info("Kit logging bridge registered.")

    def stop(self) -> None:
        if self._logging is None or self._handle is None:
            return
        self._logging.remove_logger(self._handle)
        self._handle = None
        self._logging = None
        LOGGER.info("Kit logging bridge removed.")

    def _forward(
        self,
        source: str,
        level: int,
        filename: str,
        line_number: int,
        message: str,
    ) -> None:
        project_level = self._levels.get(level)
        if project_level is None:
            return
        if getattr(self._forwarding, "active", False):
            return
        self._forwarding.active = True
        try:
            LOGGER.log(
                project_level,
                "[Isaac Sim: %s] %s",
                source,
                message.rstrip(),
            )
        finally:
            self._forwarding.active = False


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolate Isaac Sim shutdown lifecycle failures."
    )
    parser.add_argument(
        "--mode",
        choices=(
            "baseline",
            "bridge-before",
            "bridge-after",
            "world",
            *SCENE_MODES,
        ),
        required=True,
    )
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=(
            PROJECT_ROOT
            / "assets"
            / "unitree_go1"
            / "USD"
            / "go1"
            / "go1.usda"
        ),
    )
    return parser.parse_args()


def build_project_scene(
    mode: str,
    *,
    model_path: Path,
    num_envs: int,
) -> tuple[Any, Any, Any]:
    """Build progressively more of IsaacSimRuntime._build_scene()."""

    from isaacsim.core.api import World
    from isaacsim.core.cloner import GridCloner
    from isaacsim.core.prims import Articulation, RigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Usd, UsdGeom, UsdPhysics

    from envs.simulators.isaac_sim_runtime import IsaacSimRuntime

    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Isaac Sim model does not exist: {model_path}")

    world = World(
        physics_dt=0.002,
        rendering_dt=0.002,
        backend="torch",
        device="cuda:0",
        physics_prim_path="/World/physicsScene",
    )
    stage = world.stage
    source_env_path = "/World/envs/env_0"
    source_robot_path = source_env_path + "/Robot"
    UsdGeom.Xform.Define(stage, source_env_path)
    add_reference_to_stage(
        usd_path=str(model_path),
        prim_path=source_robot_path,
    )
    IsaacSimRuntime._select_physx_variant(stage, source_robot_path)
    articulation_relative_path = (
        IsaacSimRuntime._find_articulation_root_relative_path(
            stage,
            source_robot_path,
            Usd,
            UsdPhysics,
        )
    )
    world.scene.add_default_ground_plane(
        prim_path="/World/GroundPlane",
        name="GroundPlane",
    )
    floor_collision_path = IsaacSimRuntime._find_floor_collision_prim_path(
        stage,
        "/World/GroundPlane",
        Usd,
        UsdPhysics,
    )
    LOGGER.info("Go1 USD and default ground loaded.")

    articulation: Any = None
    body_view: Any = None
    if mode == "model":
        world.reset()
        return world, articulation, body_view

    cloner = GridCloner(spacing=2.0, stage=stage)
    cloner.define_base_env("/World/envs")
    env_paths = cloner.generate_paths("/World/envs/env", num_envs)
    cloner.clone(
        source_prim_path=source_env_path,
        prim_paths=env_paths,
        replicate_physics=True,
        base_env_path="/World/envs",
        root_path="/World/envs/env_",
        enable_env_ids=True,
        clone_in_fabric=False,
    )
    cloner.filter_collisions(
        physicsscene_path="/World/physicsScene",
        collision_root_path="/World/collisions",
        prim_paths=env_paths,
        global_paths=["/World/GroundPlane"],
    )
    LOGGER.info("Cloned %d environments.", num_envs)
    if mode == "cloner":
        world.reset()
        return world, articulation, body_view

    articulation = world.scene.add(
        Articulation(
            prim_paths_expr=(
                "/World/envs/env_.*/Robot" + articulation_relative_path
            ),
            name="diagnostic_robots",
            reset_xform_properties=False,
        )
    )
    LOGGER.info("Articulation view created.")
    if mode == "articulation":
        world.reset()
        return world, articulation, body_view

    body_relative_paths = IsaacSimRuntime._rigid_body_relative_paths(
        stage,
        source_robot_path,
        Usd,
        UsdPhysics,
    )
    body_paths = [
        f"/World/envs/env_{env_id}/Robot{relative_path}"
        for env_id in range(num_envs)
        for relative_path in body_relative_paths
    ]
    body_view = RigidPrim(
        prim_paths_expr=body_paths,
        name="diagnostic_robot_bodies",
        reset_xform_properties=False,
        track_contact_forces=True,
        contact_filter_prim_paths_expr=[
            [floor_collision_path]
            for _ in body_paths
        ],
    )
    world.reset()
    body_view.initialize()
    LOGGER.info(
        "Contact view initialized for %d rigid bodies.",
        len(body_paths),
    )
    return world, articulation, body_view


def main() -> None:
    arguments = parse_arguments()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    LOGGER.info("Starting diagnostic mode: %s", arguments.mode)

    bridge = KitLogBridge()
    if arguments.mode == "bridge-before":
        bridge.start()

    app = SimulationApp(
        {
            "headless": True,
            "fast_shutdown": False,
        }
    )
    LOGGER.info("SimulationApp initialized.")

    if arguments.mode in {"bridge-after", "world"}:
        bridge.start()

    world: Any = None
    articulation: Any = None
    body_view: Any = None
    try:
        if arguments.mode == "world":
            from isaacsim.core.api import World

            LOGGER.info("Creating World.")
            world = World(
                physics_dt=1.0 / 200.0,
                rendering_dt=1.0 / 60.0,
                backend="torch",
                device="cuda:0",
            )
            world.reset()
            LOGGER.info("World initialized.")

        if arguments.mode in SCENE_MODES:
            world, articulation, body_view = build_project_scene(
                arguments.mode,
                model_path=arguments.model_path,
                num_envs=arguments.num_envs,
            )

        if world is not None:
            for _ in range(arguments.steps):
                world.step(render=False)
            LOGGER.info("Completed %d simulation steps.", arguments.steps)
    finally:
        if world is not None:
            body_view = None
            articulation = None
            world.stop()
            world.clear()
            world = None
            LOGGER.info("World cleared.")
            collected = gc.collect()
            LOGGER.info(
                "Collected %d Python objects before Kit shutdown.",
                collected,
            )
        bridge.stop()
        LOGGER.info("Closing SimulationApp.")
        app.close(wait_for_replicator=False)

    print("DIAGNOSTIC COMPLETED SUCCESSFULLY", flush=True)


if __name__ == "__main__":
    main()
