from __future__ import annotations

from typing import Any

import pytest
import torch

from envs.simulators.isaac_sim_runtime import IsaacSimRuntime


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
