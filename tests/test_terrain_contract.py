import pytest
import torch

from envs.simulators.terrain import TerrainConfig, terrain_patch_seed
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


def test_isaac_rejects_unimplemented_geometry_without_mutating_config(runtime_context):
    simulator = IsaacSimSimulator(runtime_context)
    with pytest.raises(NotImplementedError, match="not implemented"):
        simulator.config_update(
            component=Component(None, None, None, None, None, None),
            terrain={"types": ["rough"], "max_height": 0.1},
        )
    assert simulator.terrain_config.is_plane
