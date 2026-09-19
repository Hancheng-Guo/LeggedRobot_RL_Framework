from __future__ import annotations

from typing import Any

import pytest

from envs.simulators.isaac_sim_runtime import IsaacSimRuntime


pytestmark = pytest.mark.isaacsim


class FakePrim:
    def __init__(
        self,
        path: str,
        *,
        articulation_root: bool = False,
        descendants: tuple[FakePrim, ...] = (),
    ) -> None:
        self.path = path
        self.articulation_root = articulation_root
        self.descendants = descendants
        self.variant_sets = FakeVariantSets()

    def GetPath(self) -> str:
        return self.path

    def GetDescendants(self) -> tuple[FakePrim, ...]:
        return self.descendants

    def HasAPI(self, api: Any) -> bool:
        return api is FakeUsdPhysics.ArticulationRootAPI and self.articulation_root

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
        FakeUsdPhysics,
    )

    assert relative_path == ""


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
            FakeUsdPhysics,
        )
