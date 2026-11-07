import pytest
import torch
import json
import ast
from pathlib import Path
import random
from copy import deepcopy

from envs.simulators.terrain import TerrainGeometry
from envs.tasks.managers.randomization.terms.terrain import (
    TerrainConfig, TerrainAssignments, generate_terrain_patch, terrain_patch_seed,
)
from envs.simulators.isaac_sim import IsaacSimSimulator
from utils import Component


def test_default_terrain_is_plane_and_uses_runtime_seed():
    config = TerrainConfig.from_mapping(None)
    assert config.is_plane
    assert config.effective_seed(17) == 17


def test_patch_seed_is_stable_and_independent_of_patch_order():
    seeds = [terrain_patch_seed(17, patch_id) for patch_id in (0, 1, 2)]
    assert seeds[2] == 7592564718643782160
    assert seeds == [terrain_patch_seed(17, patch_id) for patch_id in (0, 1, 2)]
    assert len(set(seeds)) == 3
    assert terrain_patch_seed(17, 2) == seeds[2]
    assert terrain_patch_seed(18, 2) != seeds[2]


def test_random_terrain_configuration_is_backend_independent():
    config = TerrainConfig.from_mapping({
        "types": ["plane", "slope", "rough"],
        "weights": [1, 2, 3],
        "patch_size": [6, 4],
        "resolution": 0.2,
        "max_height": 0.15,
        "max_slope": 0.25,
        "spawn_margin": 0.6,
        "seed": 42,
    })
    assert config.types == ("plane", "slope", "rough")
    assert config.weights == (1.0, 2.0, 3.0)
    assert config.effective_seed(17) == 42


@pytest.mark.parametrize("configuration", [
    {"types": ["slope"]},
    {"types": ["rough"], "max_height": 0.0},
    {"types": ["plane", "plane"]},
    {"weights": [0]},
    {"patch_size": [1.0, 1.0], "spawn_margin": 0.5},
    {"resolution": 0.0},
    {"seed": -1},
    {"unexpected": 1},
])
def test_invalid_terrain_configuration_fails_early(configuration):
    with pytest.raises((TypeError, ValueError)):
        TerrainConfig.from_mapping(configuration)


def test_flat_height_query_contract(runtime_context):
    simulator = IsaacSimSimulator(runtime_context)
    simulator.num_envs = 3
    xy = torch.tensor([[[0.0, 0.0], [1.0, 2.0]]])
    heights = simulator.get_ground_heights(xy, torch.tensor([2]))
    torch.testing.assert_close(heights, torch.zeros(1, 2))
    with pytest.raises(ValueError, match="rows"):
        simulator.get_ground_heights(xy)


def test_isaac_rejects_unimplemented_geometry_before_backend_start(runtime_context):
    simulator = IsaacSimSimulator(runtime_context)
    with pytest.raises(NotImplementedError, match="not implemented"):
        simulator.config_update(
            component=Component(None, None, None, None, None, None),
            terrain_geometry=TerrainGeometry((4, 4), (41, 41), .1),
        )
    assert simulator._backend is None


@pytest.mark.parametrize("kind", ["plane", "slope", "rough"])
def test_reference_geometry_bounds_spawn_and_reproducibility(kind):
    config = TerrainConfig.from_mapping(dict(types=[kind], max_height=0.15,
                                             max_slope=0.3, difficulty_levels=[0, 0.5, 1]))
    patch = generate_terrain_patch(config, 42, 3, 1)
    assert patch == generate_terrain_patch(config, 42, 3, 1)
    assert len(patch.geometry.heights) == len(patch.geometry.y)
    assert all(len(row) == len(patch.geometry.x) for row in patch.geometry.heights)
    assert abs(patch.slope) <= 0.15
    if kind == "rough":
        assert all(abs(h) <= 0.075 + 1e-12 for row in patch.geometry.heights for h in row)
    for j, y in enumerate(patch.geometry.y):
        for i, x in enumerate(patch.geometry.x):
            if abs(x) <= config.spawn_size[0] / 2 or kind == "plane":
                if abs(y) <= config.spawn_size[1] / 2 or kind != "rough":
                    assert patch.geometry.heights[j][i] == 0
    flat = generate_terrain_patch(config, 42, 3, 0)
    assert all(h == 0 for row in flat.geometry.heights for h in row)


def test_allocation_partial_reset_order_and_checkpoint_continuation():
    config = TerrainConfig.from_mapping(dict(types=["plane", "slope", "rough"],
        max_height=0.1, max_slope=0.2, difficulty_levels=[0.25, 1]))
    allocations = TerrainAssignments(config, 17, 4)
    untouched = allocations.patch(1)
    allocations.reset([3, 0], [1, 1])
    assert allocations.patch(1) == untouched
    other = TerrainAssignments(config, 17, 4)
    other.reset([0, 3], [1, 1])
    assert allocations.state_dict() == other.state_dict()
    restored = TerrainAssignments(config, 17, 4)
    restored.load_state_dict(json.loads(json.dumps(allocations.state_dict())))
    for instance in (allocations, restored):
        instance.reset([0, 2], [0, 1])
    assert [allocations.patch(i) for i in range(4)] == [restored.patch(i) for i in range(4)]
    saved = restored.state_dict()
    invalid = deepcopy(saved)
    invalid["generations"][0] = -1
    with pytest.raises(ValueError):
        restored.load_state_dict(invalid)
    assert restored.state_dict() == saved
    with pytest.raises(ValueError):
        restored.reset([1, 1])
    assert restored.state_dict() == saved


def test_terrain_does_not_consume_global_rngs():
    python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
    allocations = TerrainAssignments(TerrainConfig(), 7, 3)
    allocations.reset([1])
    allocations.patch(1)
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_reference_rough_golden_vertex_and_seed_override():
    config = TerrainConfig.from_mapping(dict(types=["rough"], max_height=0.1, seed=17))
    patch = generate_terrain_patch(config, 99, 2)
    assert patch.seed == 7592564718643782160
    assert patch.geometry.heights[0][0] == 0.07404575023805203
    assert patch == generate_terrain_patch(config, 0, 2)


def test_invalid_allocation_checkpoint_schema_is_atomic():
    assignments = TerrainAssignments(TerrainConfig(), 0, 2)
    state = assignments.state_dict()
    with pytest.raises(ValueError):
        assignments.load_state_dict({})
    assert assignments.state_dict() == state


def test_terrain_dependencies_point_from_task_terms_to_backend_contracts():
    root = Path(__file__).resolve().parents[1]
    violations = []
    groups = [
        (root / "envs/simulators", ("envs.tasks",)),
        (root / "envs/tasks/managers/randomization", (
            "mujoco", "isaacsim", "omni", "pxr",
            "envs.simulators.mujoco", "envs.simulators.isaac_sim",
        )),
    ]
    for folder, forbidden in groups:
        for path in folder.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                if any(module == name or module.startswith(name + ".")
                       for module in modules for name in forbidden):
                    violations.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not violations, violations


@pytest.mark.parametrize("configuration", [
    {"difficulty_levels": []}, {"difficulty_levels": [1.1]},
    {"difficulty_levels": [0, 0]}, {"spawn_size": [4, 1]},
    {"spawn_size": [0, 1]}, {"spawn_size": [1]},
])
def test_invalid_difficulty_and_safe_spawn(configuration):
    with pytest.raises(ValueError):
        TerrainConfig.from_mapping(configuration)


@pytest.mark.parametrize("env_id", [-1, 2, True, 0.5])
def test_allocation_public_patch_rejects_invalid_id(env_id):
    allocation = TerrainAssignments(TerrainConfig(), 0, 2)
    before = allocation.state_dict()
    with pytest.raises(ValueError, match="environment IDs"):
        allocation.patch(env_id)
    assert allocation.state_dict() == before


@pytest.mark.parametrize("ids", [[-1], [2], [True], [0.5], [0, 0]])
def test_allocation_public_reset_rejects_invalid_ids_atomically(ids):
    allocation = TerrainAssignments(TerrainConfig(), 0, 2)
    before = allocation.state_dict()
    with pytest.raises(ValueError, match="environment IDs"):
        allocation.reset(ids)
    assert allocation.state_dict() == before
