import pytest
import torch
import json
import ast
from pathlib import Path
import random
from copy import deepcopy
from dataclasses import asdict, fields

from envs.simulators.terrain import TerrainGeometry
from envs.tasks.managers.randomization.terms.terrain import (
    TerrainConfig, TerrainAssignments, generate_terrain_patch, terrain_patch_seed,
    PlaneTerrainConfig, SlopeTerrainConfig, RoughTerrainConfig,
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
        "types": {"plane": {"weight": 1}, "slope": {"weight": 2, "max_slope": .25},
                  "rough": {"weight": 3, "max_height": .15}},
        "patch_size": [6, 4],
        "resolution": 0.2,
        "spawn_margin": 0.6,
        "seed": 42,
    })
    assert tuple(config.types) == ("plane", "slope", "rough")
    assert config.types == {'plane':PlaneTerrainConfig(weight=1),
        'slope':SlopeTerrainConfig(max_slope=.25,weight=2),
        'rough':RoughTerrainConfig(max_height=.15,weight=3)}
    assert config.effective_seed(17) == 42


@pytest.mark.parametrize("configuration", [
    {"types": ["slope"]},
    {"types": ["rough"], "max_height": 0.0},
    {"types": ["plane", "plane"]},
    {"weights": [0]},
    {"types": {}},
    {"types": {"unknown":{}}},
    {"types": {"plane":None}},
    {"types": {"plane":{"weight":0}}},
    {"types": {"plane":{"weight":True}}},
    {"types": {"plane":{"weight":float('inf')}}},
    {"types": {"plane":{"max_slope":.1}}},
    {"types": {"slope":{"max_height":.1}}},
    {"types": {"rough":{"max_slope":.1}}},
    {"types": {"slope":{}}},
    {"types": {"rough":{"max_height":0}}},
    {"types": {"slope":{"max_slope":float('nan')}}},
    {"max_height":.1},
    {"max_slope":.1},
    {"patch_size": [1.0, 1.0], "spawn_margin": 0.5},
    {"resolution": 0.0},
    {"seed": -1},
    {"unexpected": 1},
])
def test_invalid_terrain_configuration_fails_early(configuration):
    with pytest.raises((TypeError, ValueError)):
        TerrainConfig.from_mapping(configuration)


def test_nested_type_parameters_round_trip_and_remain_bound_to_names():
    config = TerrainConfig.from_mapping({'types':{
        'rough':{'weight':3,'max_height':.04},
        'plane':{},
        'slope':{'weight':2,'max_slope':.15},
    }})
    assert tuple(config.types) == ('rough','plane','slope')
    assert config.types['rough'] == RoughTerrainConfig(weight=3,max_height=.04)
    assert config.types['plane'] == PlaneTerrainConfig()
    assert config.types['slope'] == SlopeTerrainConfig(weight=2,max_slope=.15)
    exported = asdict(config)
    assert exported['types'] == {
        'rough':{'weight':3.,'max_height':.04},
        'plane':{'weight':1.},
        'slope':{'weight':2.,'max_slope':.15},
    }
    assert not {'weights','max_height','max_slope'} & exported.keys()
    assert TerrainConfig.from_mapping(json.loads(json.dumps(exported))) == config


def test_type_specific_fields_and_independent_defaults():
    assert not {'weights','max_height','max_slope'} & {item.name for item in fields(TerrainConfig)}
    assert {item.name for item in fields(PlaneTerrainConfig)} == {'weight'}
    assert {item.name for item in fields(SlopeTerrainConfig)} == {'weight','max_slope'}
    assert {item.name for item in fields(RoughTerrainConfig)} == {'weight','max_height'}
    first, second = TerrainConfig(), TerrainConfig()
    assert first.types is not second.types
    exported = asdict(first)
    exported['types']['plane']['weight'] = 9
    assert first.types['plane'].weight == 1 and second.types['plane'].weight == 1


def test_type_order_remains_part_of_config_and_restore_compatibility():
    types = {'plane':{},'slope':{'max_slope':.2},'rough':{'max_height':.1}}
    original = TerrainConfig.from_mapping({'types':types})
    reordered = TerrainConfig.from_mapping({'types':dict(reversed(list(types.items())))})
    assert original != reordered
    state = TerrainAssignments(original,42,2).state_dict()
    restored = TerrainAssignments(reordered,42,2)
    before = restored.state_dict()
    with pytest.raises(ValueError,match='configuration differs'):
        restored.load_state_dict(state)
    assert restored.state_dict() == before


def test_difficulty_levels_are_shared_and_independent_of_type_selection():
    config = TerrainConfig.from_mapping({'types':{'slope':{'max_slope':.3},
        'rough':{'max_height':.1}}, 'difficulty_levels':[.25,1.]})
    seen = set()
    for patch_id in range(16):
        low = generate_terrain_patch(config,42,patch_id,0)
        high = generate_terrain_patch(config,42,patch_id,1)
        assert low.terrain_type == high.terrain_type and low.seed == high.seed
        seen.add(low.terrain_type)
        torch.testing.assert_close(torch.tensor(low.geometry.heights),
                                   torch.tensor(high.geometry.heights)*.25)
    assert seen == {'slope','rough'}


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
    params = {'max_slope':.3} if kind == 'slope' else {'max_height':.15} if kind == 'rough' else {}
    config = TerrainConfig.from_mapping(dict(types={kind:params}, difficulty_levels=[0, 0.5, 1]))
    patch = generate_terrain_patch(config, 42, 3, 1)
    assert patch == generate_terrain_patch(config, 42, 3, 1)
    assert len(patch.geometry.heights) == len(patch.geometry.y)
    assert all(len(row) == len(patch.geometry.x) for row in patch.geometry.heights)
    heights = torch.tensor(patch.geometry.heights)
    assert not heights[[0, -1]].any() and not heights[:, [0, -1]].any()
    if kind == 'slope':
        slope_config = config.types['slope']
        assert isinstance(slope_config,SlopeTerrainConfig)
        assert (heights.diff(dim=1).abs() <= slope_config.max_slope * config.resolution + 1e-7).all()
        assert (heights.diff(dim=0).abs() <= slope_config.max_slope * config.resolution + 1e-7).all()
    if kind == "rough":
        assert all(abs(h) <= 0.075 + 1e-12 for row in patch.geometry.heights for h in row)
    for j, y in enumerate(patch.geometry.y):
        for i, x in enumerate(patch.geometry.x):
            if kind == "plane" or (abs(x) <= config.spawn_size[0] / 2
                                   and abs(y) <= config.spawn_size[1] / 2):
                assert patch.geometry.heights[j][i] == 0
    flat = generate_terrain_patch(config, 42, 3, 0)
    assert all(h == 0 for row in flat.geometry.heights for h in row)


def test_allocation_partial_reset_order_and_checkpoint_continuation():
    config = TerrainConfig.from_mapping(dict(types={"plane":{}, "slope":{"max_slope":.2},
        "rough":{"max_height":.1}}, difficulty_levels=[0.25, 1]))
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
    config = TerrainConfig.from_mapping(dict(types={"rough": {"max_height": 0.1}}, seed=17))
    patch = generate_terrain_patch(config, 99, 2)
    assert patch.seed == 7592564718643782160
    assert patch.geometry.heights[0][0] == 0.0
    assert patch.geometry.heights[1][1] == -0.003739335163572144
    assert patch == generate_terrain_patch(config, 0, 2)


@pytest.mark.parametrize('version', [1,2,3,4])
def test_terrain_checkpoint_rejects_previous_geometry_rules(version):
    allocations = TerrainAssignments(TerrainConfig(), 0, 2)
    old = allocations.state_dict()
    old['version'] = version
    with pytest.raises(ValueError, match='Incompatible'):
        allocations.load_state_dict(old)


@pytest.mark.parametrize('patch_size', [(4.,4.),(3.7,2.9)])
@pytest.mark.parametrize('difficulty', [.25,1.])
def test_slope_coefficient_and_collision_triangle_gradients(patch_size, difficulty):
    import math
    config = TerrainConfig.from_mapping(dict(types={"slope": {"max_slope": .3}},
        patch_size=patch_size, difficulty_levels=[difficulty]))
    # max_slope bounds the coefficient; combined triangle gradients may exceed it.
    patch = generate_terrain_patch(config,42,12)
    slope_config = config.types['slope']
    assert isinstance(slope_config,SlopeTerrainConfig)
    heights, xs, ys = patch.geometry.heights, patch.geometry.x, patch.geometry.y
    centre_row = heights[len(ys)//2]
    axis_slope = max(abs((b-a)/(right-left))
                     for a,b,left,right in zip(centre_row,centre_row[1:],xs,xs[1:]))
    assert axis_slope == pytest.approx(.25949638722166 * difficulty)
    assert axis_slope <= slope_config.max_slope * difficulty + 1e-12
    for j in range(len(ys)-1):
        for i in range(len(xs)-1):
            a,b = heights[j][i:i+2]
            d,c = heights[j+1][i:i+2]
            dx,dy = xs[i+1]-xs[i], ys[j+1]-ys[j]
            for gradient in (math.hypot((b-a)/dx,(c-b)/dy),
                             math.hypot((c-d)/dx,(d-a)/dy)):
                assert gradient <= axis_slope*math.sqrt(2) + 1e-12


@pytest.mark.parametrize('patch_id,sign', [(3,-1), (12,1)])
def test_slope_has_one_sign_and_rises_or_falls_in_all_four_directions(patch_id, sign):
    config = TerrainConfig.from_mapping(dict(types={"slope": {"max_slope": .3}}, spawn_size=[1.,1.]))
    patch = generate_terrain_patch(config,42,patch_id)
    heights = torch.tensor(patch.geometry.heights,dtype=torch.float64)
    assert (heights * sign >= 0).all()
    torch.testing.assert_close(heights,heights.flip(0),rtol=0,atol=1e-12)
    torch.testing.assert_close(heights,heights.flip(1),rtol=0,atol=1e-12)
    torch.testing.assert_close(heights,heights.T,rtol=0,atol=1e-12)
    centre = len(patch.geometry.x)//2
    for j,i in [(centre,7),(centre,33),(7,centre),(33,centre)]:
        assert heights[j,i] * sign > 0
    assert not heights[15:26,15:26].any()
    assert not heights[[0,-1]].any() and not heights[:,[0,-1]].any()


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
