from __future__ import annotations

import logging
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from envs.simulators.isaac_sim_runtime import IsaacSimRuntime
from envs.simulators.isaac_sim_model import IsaacSimModelConverter
from envs.simulators import isaac_sim_runtime


pytestmark = pytest.mark.isaacsim


class FakePrim:
    def __init__(
        self,
        path: str,
        *,
        articulation_root: bool = False,
        rigid_body: bool = False,
        collision: bool = False,
        descendants: tuple[FakePrim, ...] = (),
    ) -> None:
        self.path = path
        self.articulation_root = articulation_root
        self.rigid_body = rigid_body
        self.collision = collision
        self.descendants = descendants
        self.variant_sets = FakeVariantSets()

    def GetPath(self) -> str:
        return self.path

    def HasAPI(self, api: Any) -> bool:
        return (
            api is FakeUsdPhysics.ArticulationRootAPI
            and self.articulation_root
        ) or (
            api is FakeUsdPhysics.RigidBodyAPI
            and self.rigid_body
        ) or (
            api is FakeUsdPhysics.CollisionAPI
            and self.collision
        )

    def GetVariantSets(self) -> FakeVariantSets:
        return self.variant_sets


class FakeVariantSet:
    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names
        self.selection = ""

    def GetVariantNames(self) -> tuple[str, ...]:
        return self.names

    def SetVariantSelection(self, selection: str) -> bool:
        if selection not in self.names:
            return False
        self.selection = selection
        return True


class FakeVariantSets:
    def __init__(self) -> None:
        self.sets: dict[str, FakeVariantSet] = {}

    def HasVariantSet(self, name: str) -> bool:
        return name in self.sets

    def GetVariantSet(self, name: str) -> FakeVariantSet:
        return self.sets[name]


class FakeStage:
    def __init__(self, robot_prim: FakePrim) -> None:
        self.robot_prim = robot_prim

    def GetPrimAtPath(self, path: str) -> FakePrim:
        assert path == self.robot_prim.path
        return self.robot_prim


class FakeUsdPhysics:
    class ArticulationRootAPI:
        pass

    class RigidBodyAPI:
        pass

    class CollisionAPI:
        pass


class FakeUsd:
    @staticmethod
    def PrimRange(root: FakePrim) -> tuple[FakePrim, ...]:
        return (root, *root.descendants)


def _model_converter(
    package_path: str = "assets/robot_description",
) -> IsaacSimModelConverter:
    return IsaacSimModelConverter(
        ros_package_paths=({"robot_description": package_path},),
        merge_fixed_joints=False,
        allow_self_collision=False,
        joint_stiffness=100.0,
        joint_damping=2.0,
    )


def test_kit_log_bridge_forwards_selected_levels_without_recursion(
    monkeypatch,
) -> None:
    runtime = IsaacSimRuntime.__new__(IsaacSimRuntime)
    runtime._kit_log_levels = {2: logging.WARNING}
    runtime._kit_log_forwarding = threading.local()
    records: list[tuple[int, tuple[Any, ...]]] = []

    def capture(level: int, message: str, *args: Any) -> None:
        records.append((level, args))
        runtime._forward_kit_log("nested", 2, "", 0, "ignored")

    monkeypatch.setattr(isaac_sim_runtime.LOGGER, "log", capture)

    runtime._forward_kit_log("omni.physx", 1, "", 0, "info")
    runtime._forward_kit_log("omni.physx", 2, "", 0, "warning\n")

    assert records == [
        (logging.WARNING, ("omni.physx", "warning")),
    ]


def test_start_application_disables_native_kit_console(
    monkeypatch,
    capsys,
) -> None:
    launch_configs: list[dict[str, Any]] = []
    lifecycle: list[str] = []

    class FakeSimulationApp:
        def __init__(self, config: dict[str, Any]) -> None:
            lifecycle.append("application")
            print("native startup info")
            launch_configs.append(config)

    simulation_app = ModuleType("isaacsim.simulation_app")
    simulation_app.SimulationApp = FakeSimulationApp  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaacsim.simulation_app", simulation_app)
    monkeypatch.setattr(isaac_sim_runtime.LOGGER, "info", lambda message: None)

    runtime = IsaacSimRuntime.__new__(IsaacSimRuntime)
    runtime.render_mode = None
    runtime._start_log_bridge = lambda: lifecycle.append("bridge")

    runtime._start_application()

    assert launch_configs == [
        {
            "headless": True,
            "extra_args": [
                "--/app/enableStdoutOutput=false",
                "--/app/python/logSysStdOutput=false",
                "--/log/enableStandardStreamOutput=false",
            ],
        }
    ]
    assert lifecycle == ["bridge", "application"]
    assert "native startup info" not in capsys.readouterr().out


def test_start_application_stops_log_bridge_after_startup_failure(
    monkeypatch,
) -> None:
    lifecycle: list[str] = []

    class FailingSimulationApp:
        def __init__(self, config: dict[str, Any]) -> None:
            lifecycle.append("application")
            raise RuntimeError("startup failed")

    simulation_app = ModuleType("isaacsim.simulation_app")
    simulation_app.SimulationApp = FailingSimulationApp  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaacsim.simulation_app", simulation_app)
    monkeypatch.setattr(isaac_sim_runtime.LOGGER, "info", lambda message: None)

    runtime = IsaacSimRuntime.__new__(IsaacSimRuntime)
    runtime.render_mode = None
    runtime._start_log_bridge = lambda: lifecycle.append("bridge")
    runtime._stop_log_bridge = lambda: lifecycle.append("stop")

    with pytest.raises(RuntimeError, match="startup failed"):
        runtime._start_application()

    assert lifecycle == ["bridge", "application", "stop"]


def test_rgb_camera_does_not_use_kit_run_loop_frequency() -> None:
    camera_arguments: list[dict[str, Any]] = []

    class FakeCamera:
        def __init__(self, **kwargs: Any) -> None:
            camera_arguments.append(kwargs)

    runtime = IsaacSimRuntime.__new__(IsaacSimRuntime)
    runtime._requested_camera_prim_path = None
    runtime.camera_resolution = (640, 480)

    runtime._build_camera(FakeCamera)

    assert runtime.camera_prim_path == "/World/Camera"
    assert camera_arguments == [
        {
            "prim_path": "/World/Camera",
            "position": pytest.approx((2.5, 2.5, 1.8)),
            "resolution": (640, 480),
        }
    ]
    assert "frequency" not in camera_arguments[0]


def test_close_releases_camera_before_world_and_application() -> None:
    lifecycle: list[str] = []

    runtime = IsaacSimRuntime.__new__(IsaacSimRuntime)
    runtime._closed = False
    runtime._camera = SimpleNamespace(
        destroy=lambda: lifecycle.append("camera.destroy")
    )
    runtime._world = SimpleNamespace(
        stop=lambda: lifecycle.append("world.stop"),
        clear=lambda: lifecycle.append("world.clear"),
    )
    runtime._body_view = object()
    runtime._articulation = object()
    runtime._stop_log_bridge = lambda: lifecycle.append("bridge.stop")
    runtime._app = SimpleNamespace(
        close=lambda: lifecycle.append("application.close")
    )

    runtime.close()

    assert lifecycle == [
        "camera.destroy",
        "world.stop",
        "world.clear",
        "bridge.stop",
        "application.close",
    ]
    assert runtime._camera is None
    assert runtime._body_view is None
    assert runtime._articulation is None
    assert runtime._world is None
    assert runtime._app is None
    assert runtime._closed


def test_close_continues_after_resource_cleanup_failure() -> None:
    lifecycle: list[str] = []

    def fail_camera_destroy() -> None:
        lifecycle.append("camera.destroy")
        raise RuntimeError("camera cleanup failed")

    runtime = IsaacSimRuntime.__new__(IsaacSimRuntime)
    runtime._closed = False
    runtime._camera = SimpleNamespace(destroy=fail_camera_destroy)
    runtime._world = SimpleNamespace(
        stop=lambda: lifecycle.append("world.stop"),
        clear=lambda: lifecycle.append("world.clear"),
    )
    runtime._body_view = object()
    runtime._articulation = object()
    runtime._stop_log_bridge = lambda: lifecycle.append("bridge.stop")
    runtime._app = SimpleNamespace(
        close=lambda: lifecycle.append("application.close")
    )

    with pytest.raises(RuntimeError, match="camera cleanup failed"):
        runtime.close()

    assert lifecycle == [
        "camera.destroy",
        "world.stop",
        "world.clear",
        "bridge.stop",
        "application.close",
    ]
    assert runtime._closed


def test_model_conversion_uses_model_usd_directory(tmp_path) -> None:
    model_path = tmp_path / "assets" / "robots" / "go1.urdf"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("<robot name='go1'/>", encoding="utf-8")

    output_directory = _model_converter()._asset_output_directory(model_path)

    assert output_directory == tmp_path / "assets" / "robots" / "USD"


def test_conversion_cache_is_portable_across_package_locations(
    tmp_path,
) -> None:
    model_path = tmp_path / "assets" / "robots" / "go1.urdf"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("<robot name='go1'/>", encoding="utf-8")
    first = _model_converter("C:/first/robot_description")
    second = _model_converter("D:/second/robot_description")

    first_fingerprint = first._conversion_fingerprint(model_path, "urdf")
    second_fingerprint = second._conversion_fingerprint(model_path, "urdf")
    output_directory = first._asset_output_directory(model_path)
    output_path = output_directory / "go1" / "go1.usda"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("#usda 1.0", encoding="utf-8")
    first._write_conversion_manifest(
        output_directory,
        output_path,
        first_fingerprint,
    )

    assert first_fingerprint == second_fingerprint
    assert second._cached_import_path(
        output_directory,
        second_fingerprint,
    ) == output_path


def test_selects_physx_variant_before_inspecting_robot() -> None:
    root = FakePrim("/World/envs/env_0/Robot")
    physics = FakeVariantSet(("mujoco", "none", "physics", "physx"))
    root.variant_sets.sets["Physics"] = physics

    IsaacSimRuntime._select_physx_variant(FakeStage(root), root.path)

    assert physics.selection == "physx"


def test_allows_assets_without_physics_variant_set() -> None:
    root = FakePrim("/World/envs/env_0/Robot")

    IsaacSimRuntime._select_physx_variant(FakeStage(root), root.path)


def test_rejects_physics_variant_set_without_physx() -> None:
    root = FakePrim("/World/envs/env_0/Robot")
    root.variant_sets.sets["Physics"] = FakeVariantSet(("none", "physics"))

    with pytest.raises(RuntimeError, match="without a PhysX variant"):
        IsaacSimRuntime._select_physx_variant(FakeStage(root), root.path)


def test_finds_nested_articulation_root_relative_to_model_prim() -> None:
    root = FakePrim(
        "/World/envs/env_0/Robot",
        descendants=(
            FakePrim(
                "/World/envs/env_0/Robot/trunk",
                articulation_root=True,
            ),
        ),
    )

    relative_path = IsaacSimRuntime._find_articulation_root_relative_path(
        FakeStage(root),
        root.path,
        FakeUsd,
        FakeUsdPhysics,
    )

    assert relative_path == "/trunk"


def test_accepts_model_prim_as_articulation_root() -> None:
    root = FakePrim(
        "/World/envs/env_0/Robot",
        articulation_root=True,
    )

    relative_path = IsaacSimRuntime._find_articulation_root_relative_path(
        FakeStage(root),
        root.path,
        FakeUsd,
        FakeUsdPhysics,
    )

    assert relative_path == ""


def test_finds_rigid_bodies_with_usd_prim_range() -> None:
    root = FakePrim(
        "/World/envs/env_0/Robot",
        descendants=(
            FakePrim("/World/envs/env_0/Robot/trunk", rigid_body=True),
            FakePrim("/World/envs/env_0/Robot/visual"),
            FakePrim("/World/envs/env_0/Robot/FR_foot", rigid_body=True),
        ),
    )

    paths = IsaacSimRuntime._rigid_body_relative_paths(
        FakeStage(root),
        root.path,
        FakeUsd,
        FakeUsdPhysics,
    )

    assert paths == ("/trunk", "/FR_foot")


def test_resolves_floor_container_to_collision_prim() -> None:
    floor = FakePrim(
        "/World/GroundPlane",
        descendants=(
            FakePrim(
                "/World/GroundPlane/CollisionPlane",
                collision=True,
            ),
        ),
    )

    collision_path = IsaacSimRuntime._find_floor_collision_prim_path(
        FakeStage(floor),
        floor.path,
        FakeUsd,
        FakeUsdPhysics,
    )

    assert collision_path == "/World/GroundPlane/CollisionPlane"


class FakeBodyView:
    def __init__(self, contact_forces: torch.Tensor) -> None:
        self.contact_forces = contact_forces

    def get_contact_force_matrix(
        self,
        *,
        dt: float,
        clone: bool,
    ) -> torch.Tensor:
        assert dt == 0.002
        assert clone
        return self.contact_forces.clone()


def test_precomputed_contact_layout_is_reused_for_state_queries(
    runtime_context,
) -> None:
    runtime = object.__new__(IsaacSimRuntime)
    runtime.context = runtime_context
    runtime.num_envs = 2
    runtime.sim_dt = 0.002
    runtime.foot_contact_force_threshold = 15.0
    runtime._body_names = ("trunk", "foot")
    runtime._body_prim_paths = ("/trunk", "/foot")
    runtime.foot_body_prim_paths = ("/foot",)
    runtime.floor_prim_paths = ("/World/GroundPlane",)
    runtime._body_view = FakeBodyView(torch.tensor([
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0, 20.0]],
        [[0.0, 0.0, 5.0]],
        [[0.0, 0.0, 10.0]],
    ]))

    runtime._prepare_state_buffers()
    contact_ids, contact_forces, foot_contact = runtime._contact_state()

    assert runtime._indices(None) is runtime._all_env_indices
    assert contact_ids.tolist() == [
        [[-1, -1], [1, 2]],
        [[0, 2], [1, 2]],
    ]
    assert contact_forces[..., 0].tolist() == [
        [0.0, 20.0],
        [5.0, 10.0],
    ]
    assert foot_contact.squeeze(-1).tolist() == [
        [False, True],
        [False, False],
    ]


@pytest.mark.parametrize(
    ("descendants", "message"),
    [
        ((), "No articulation root"),
        (
            (
                FakePrim("/World/envs/env_0/Robot/first", articulation_root=True),
                FakePrim("/World/envs/env_0/Robot/second", articulation_root=True),
            ),
            "Expected one articulation root",
        ),
    ],
)
def test_rejects_ambiguous_articulation_roots(
    descendants: tuple[FakePrim, ...],
    message: str,
) -> None:
    root = FakePrim(
        "/World/envs/env_0/Robot",
        descendants=descendants,
    )

    with pytest.raises(RuntimeError, match=message):
        IsaacSimRuntime._find_articulation_root_relative_path(
            FakeStage(root),
            root.path,
            FakeUsd,
            FakeUsdPhysics,
        )
