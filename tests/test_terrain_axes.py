"""Terrain axes own candidates and sampling; terrain owns physical composition."""
from copy import deepcopy
from typing import Any

from envs.tasks.managers.curriculum import DiscreteDimensionConfig

import pytest
import torch

from envs.tasks.managers.randomization import RandomizationManager
from envs.tasks.managers.randomization.terms.terrain import TerrainConfig, TerrainRandomizationTerm
from envs.tasks.managers.randomization.terms.terrain_category import TerrainCategoryConfig, TerrainCategoryTerm
from envs.tasks.managers.randomization.terms.terrain_difficulty import TerrainDifficultyConfig, TerrainDifficultyTerm
from test_terrain_curriculum import TerrainBackend


def category_config(strategy="Weighted"):
    values = {"plane": {}, "rough": {"max_height": .1}}
    if strategy == "Weighted":
        values["plane"]["weight"] = 1
        values["rough"]["weight"] = 3
    elif strategy == "Fixed":
        values = {"rough": values["rough"]}
    return {"type": strategy, "params": {"values": values}}



def test_category_curriculum_without_difficulty_uses_fixed_one(runtime_context):
    category = category_config("Curriculum")
    category["params"]["curriculum"] = "course"
    config = {"category": category}
    spaces = RandomizationManager.get_curriculum_space_config({"terms": {"terrain": config}})
    assert set(spaces["course"].dimensions) == {"terrain_category"}

    class Sampler:
        def get_bounds(self, dimension, env_ids=None):
            assert dimension == "terrain_category"
            value = torch.ones(1, 1)
            return value, value

    class Provider:
        def get_term(self, name):
            assert name == "course"
            return Sampler()

    term = TerrainRandomizationTerm("terrain", 2, runtime_context, curriculum_manager=Provider(), **config)
    backend = TerrainBackend(runtime_context, 2)
    term.bind(backend)
    term.before_simulator_reset()
    backend.reset()
    term.after_simulator_reset()
    term.commit_simulator_reset()
    assert all(patch is not None and patch.terrain_type == "rough" and patch.difficulty == 1.
               for patch in term.patches)
    assert term.difficulty.config.type == "Fixed"
    assert not term.difficulty._curriculum_samplers
    snapshot = term.checkpoint_state_dict()
    assert snapshot["version"] == 4
    assert snapshot["allocations"]["version"] == 8


def test_both_axes_describe_discrete_spaces_with_the_same_interface():
    category = category_config("Curriculum")
    category["params"]["curriculum"] = "course"
    difficulty = {"type": "Curriculum", "params": {"curriculum": "course", "values": [.25, .6, 1.]}}
    assert TerrainCategoryTerm.get_curriculum_references(category) == {"course"}
    assert TerrainDifficultyTerm.get_curriculum_references(difficulty) == {"course"}
    category_space = TerrainCategoryTerm.get_curriculum_space_config("terrain_category", category)["course"]
    difficulty_space = TerrainDifficultyTerm.get_curriculum_space_config("terrain_difficulty", difficulty)["course"]
    assert category_space.dimensions["terrain_category"] == DiscreteDimensionConfig((0., 1.))
    assert difficulty_space.dimensions["terrain_difficulty"] == DiscreteDimensionConfig((0., 1., 2.))


def test_weighted_category_sampling_is_deterministic_and_does_not_consume_rng(runtime_context):
    config = TerrainCategoryConfig.from_mapping(category_config())
    first = TerrainCategoryTerm("terrain_category", config, runtime_context)
    second = TerrainCategoryTerm("terrain_category", config, runtime_context)
    before = torch.get_rng_state().clone()
    choices = []
    repeated = []
    for _ in range(2000):
        choices.append(first.sample(0, 17))
        repeated.append(second.sample(0, 17))
        first.commit_sample(0)
        second.commit_sample(0)
    assert choices == repeated
    assert .70 < choices.count("rough") / len(choices) < .80
    assert torch.equal(before, torch.get_rng_state())


def test_fixed_difficulty_uses_its_only_candidate(runtime_context):
    config = TerrainDifficultyConfig.from_mapping({
        "type": "Fixed", "params": {"values": [.6]},
    })
    term = TerrainDifficultyTerm("terrain_difficulty", config, runtime_context)
    assert term.sample(0, 17) == .6
    assert not hasattr(config, "value")
    assert config.to_mapping() == {"type": "Fixed", "params": {"values": [.6]}}
    assert TerrainDifficultyConfig.from_mapping(None).values == (1.,)
    with pytest.raises(ValueError, match="Unsupported parameters"):
        TerrainDifficultyConfig.from_mapping({"type": "Fixed", "params": {"value": .6}})


@pytest.mark.parametrize("value,dtype", [(.6, torch.float32), (.6, torch.float64), (1, torch.int64)])
def test_requested_difficulty_resolves_actual_numeric_value(runtime_context, value, dtype):
    term = TerrainRandomizationTerm("terrain", 2, runtime_context, category=category_config("Fixed"),
        difficulty={"type": "Uniform", "params": {"values": [.25, .6, 1.]}})
    term.set_difficulty(torch.tensor([1]), torch.tensor([value], dtype=dtype))
    assert term.difficulty._requests == {1: float(value)}
    before = deepcopy(term.difficulty._requests)
    with pytest.raises(ValueError, match="configured value"):
        term.set_difficulty(torch.tensor([1]), torch.tensor([0]))
    assert term.difficulty._requests == before


@pytest.mark.parametrize("configuration", [
    {"types": {"plane": {}}}, {"difficulty_levels": [1.]}, {"sampling": {}},
    {"category": {"type": 'Weighted', "params": {"values": {}}}},
    {"difficulty": {"type": "Fixed", "params": {"value": True}}},
    {"difficulty": {"type": "Uniform", "params": {"values": [.5, .5]}}},
])
def test_new_configuration_rejects_old_fields_and_invalid_candidates(configuration):
    with pytest.raises((TypeError, ValueError)):
        TerrainConfig.from_mapping(configuration)


def test_axis_order_participates_in_checkpoint_compatibility(runtime_context):
    config: dict[str, Any] = {"category": category_config(), "difficulty": {"type": "Uniform", "params": {"values": [.25, 1.]}}}
    term = TerrainRandomizationTerm("terrain", 2, runtime_context, **config)
    saved = term.checkpoint_state_dict()
    config["difficulty"]["params"]["values"] = [1., .25]
    other = TerrainRandomizationTerm("terrain", 2, runtime_context, **config)
    with pytest.raises(ValueError, match="Incompatible terrain"):
        other.load_checkpoint_state_dict(saved)


def test_failed_reset_preserves_child_sampling_state_and_requests(runtime_context, model_context, monkeypatch):
    from test_terrain_curriculum import make_task, reset, terrain, assert_equal

    axes = {"category": {"type": "Fixed", "params": {"values": {"rough": {"max_height": .1}}}},
            "difficulty": {"type": "Uniform"}}
    source, backend = make_task(runtime_context, model_context, sampling="Uniform", axes=axes, resample_patch=True)
    control, physical = make_task(runtime_context, model_context, sampling="Uniform", axes=axes, resample_patch=True)
    for task, simulator in ((source, backend), (control, physical)):
        reset(task, simulator)
        terrain(task).set_difficulty(torch.tensor([1]), torch.tensor([1.]))
    term = terrain(source)
    before = term.checkpoint_state_dict()
    apply = backend.apply_terrain

    def fail(*args):
        raise RuntimeError("injected terrain apply failure")

    monkeypatch.setattr(backend, "apply_terrain", fail)
    with pytest.raises(RuntimeError, match="injected terrain apply failure"):
        reset(source, backend)
    assert_equal(before, term.checkpoint_state_dict())
    assert term.difficulty._requests == {1: 1.}
    monkeypatch.setattr(backend, "apply_terrain", apply)
    reset(source, backend)
    reset(control, physical)
    assert_equal(term.checkpoint_state_dict(), terrain(control).checkpoint_state_dict())
    assert term.patches == terrain(control).patches
    assert term.category._sample_counts == [0, 0, 0]
    assert term.difficulty._sample_counts == [2, 2, 2]
    assert not term.difficulty._requests


@pytest.mark.parametrize("axis", ["category", "difficulty"])
def test_corrupt_child_checkpoint_is_rejected_before_mutation(runtime_context, model_context, axis):
    from test_terrain_curriculum import make_task, reset, terrain, assert_equal

    task, backend = make_task(runtime_context, model_context, sampling="Uniform")
    reset(task, backend)
    term = terrain(task)
    before = term.checkpoint_state_dict()
    saved = deepcopy(before)
    saved["axes"][axis]["sample_counts"][0] = -1
    with pytest.raises(ValueError, match="sampling counts"):
        term.load_checkpoint_state_dict(saved)
    assert_equal(before, term.checkpoint_state_dict())


def test_child_playback_projects_pending_difficulty_request(runtime_context, model_context):
    from test_terrain_curriculum import make_task, reset, terrain

    task, backend = make_task(runtime_context, model_context, sampling="Uniform")
    reset(task, backend)
    term = terrain(task)
    term.set_difficulty(torch.tensor([2]), torch.tensor([.25]))
    child = term.difficulty.playback_state_dict(2)
    assert child["num_envs"] == 1
    assert child["sample_counts"] == [1]
    assert child["requests"] == {0: .25}
    assert term.difficulty._requests == {2: .25}

@pytest.mark.parametrize("strategy", ["Fixed", "Curriculum"])
@pytest.mark.parametrize("weight", [1, 0, None, float("inf")])
def test_only_weighted_category_accepts_explicit_weight(strategy, weight):
    params: dict[str, Any] = {"values": {"slope": {"max_slope": .2, "weight": weight}}}
    if strategy == "Curriculum":
        params["curriculum"] = "course"
    with pytest.raises(ValueError, match="does not accept weight"):
        TerrainCategoryConfig.from_mapping({"type": strategy, "params": params})


def test_fixed_category_rejects_multiple_candidates():
    with pytest.raises(ValueError, match="exactly one candidate"):
        TerrainCategoryConfig.from_mapping({"type": "Fixed", "params": {
            "values": {"plane": {}, "slope": {"max_slope": .2}},
        }})


def test_fixed_category_selects_its_only_candidate_without_sampling(runtime_context, monkeypatch):
    config = TerrainCategoryConfig.from_mapping({"type": "Fixed", "params": {
        "values": {"slope": {"max_slope": .2}},
    }})
    term = TerrainCategoryTerm("terrain_category", config, runtime_context, num_envs=2)

    def unexpected(*args, **kwargs):
        raise AssertionError("Fixed selection must not read curriculum bounds")

    monkeypatch.setattr(term, "_curriculum_index", unexpected)
    for env_id in range(2):
        assert term.sample(env_id, 17) == "slope"
        term.commit_sample(env_id)
        assert term.sample(env_id, 99, "slope") == "slope"
    assert term._sample_counts == [0, 0]


@pytest.mark.parametrize("strategy", ["Fixed", "Weighted", "Curriculum"])
def test_category_exports_only_strategy_supported_weights(strategy):
    params: dict[str, Any] = {"values": {"slope": {"max_slope": .2}}}
    if strategy == "Weighted":
        params["values"]["slope"]["weight"] = 3
    elif strategy == "Curriculum":
        params["curriculum"] = "course"
    config = TerrainCategoryConfig.from_mapping({"type": strategy, "params": params})
    exported = config.to_mapping()
    values = exported["params"]["values"]
    assert ("weight" in values["slope"]) == (strategy == "Weighted")
    if strategy == "Weighted":
        assert values["slope"]["weight"] == 3
    assert TerrainCategoryConfig.from_mapping(exported) == config
    complete = TerrainConfig.from_mapping({"category": exported})
    assert TerrainConfig.from_mapping(complete.to_mapping()) == complete


def test_failed_physical_rollback_keeps_snapshot_for_retry(runtime_context, monkeypatch):
    from test_terrain_manager import make_manager, reset_from_environment
    from test_terrain_curriculum import assert_equal

    term, backend = make_manager(runtime_context)
    reset_from_environment(term, backend)
    before = term.checkpoint_state_dict()
    positions = backend.positions.clone()
    term.before_simulator_reset(torch.tensor([0]))
    backend.positions[0] = 99
    restore = backend.restore_reset_state

    def fail_restore(state):
        raise RuntimeError("injected restore failure")

    monkeypatch.setattr(backend, "restore_reset_state", fail_restore)
    with pytest.raises(RuntimeError, match="injected restore failure"):
        term.abort_simulator_reset()
    assert term._pending is not None
    with pytest.raises(RuntimeError, match="unfinished terrain reset"):
        term.checkpoint_state_dict()
    monkeypatch.setattr(backend, "restore_reset_state", restore)
    term.abort_simulator_reset()
    term.abort_simulator_reset()
    assert term._pending is None
    torch.testing.assert_close(backend.positions, positions)
    assert_equal(before, term.checkpoint_state_dict())


@pytest.mark.parametrize("operation", ["difficulty", "spawn", "reallocation", "configuration", "bind", "empty_reset"])
def test_pending_reset_rejects_new_requests_and_reconfiguration(runtime_context, operation):
    from test_terrain_manager import make_manager

    term, backend = make_manager(runtime_context)
    term.before_simulator_reset(torch.tensor([0]))
    with pytest.raises(RuntimeError, match="pending terrain reset"):
        if operation == "difficulty":
            term.set_difficulty(torch.tensor([0]), torch.tensor([1.]))
        elif operation == "spawn":
            term.request_spawn(torch.tensor([0]))
        elif operation == "reallocation":
            term.request_reallocation(torch.tensor([0]))
        elif operation == "configuration":
            term.config_update(None)
        elif operation == "bind":
            term.bind(backend)
        else:
            term.before_simulator_reset(torch.empty(0, dtype=torch.long))
    assert not term.difficulty._requests
    assert not term._requested_spawns and not term._requested_reallocations
    term.abort_simulator_reset()


def test_plane_sampling_validates_reset_pair_before_skipping_physics(runtime_context):
    term = TerrainRandomizationTerm("terrain", 2, runtime_context,
        category={"type": 'Weighted', "params": {"values": {"plane": {}}}})
    backend = TerrainBackend(runtime_context, 2)
    term.bind(backend)
    assert term.geometry is None
    term.before_simulator_reset(torch.tensor([0]))
    with pytest.raises(RuntimeError, match="must match"):
        term.after_simulator_reset(torch.tensor([1]))
    term.after_simulator_reset(torch.tensor([0]))
    term.commit_simulator_reset(torch.tensor([0]))
    assert term.category._sample_counts == [1, 0]


def test_curriculum_difficulty_rejects_checkpoint_request_injection(runtime_context, model_context):
    from test_terrain_curriculum import make_task, terrain, assert_equal

    task, _ = make_task(runtime_context, model_context)
    child = terrain(task).difficulty
    before = child.checkpoint_state_dict()
    saved = deepcopy(before)
    saved["requests"] = {0: .25}
    with pytest.raises(ValueError, match="pending terrain difficulty requests"):
        child.load_checkpoint_state_dict(saved)
    assert_equal(before, child.checkpoint_state_dict())


def test_failed_provider_update_restores_both_axis_bindings(runtime_context):
    class Sampler:
        def __init__(self, index):
            self.index = index

        def get_bounds(self, dimension, env_ids=None):
            values = torch.full((1, 1), float(self.index))
            return values, values

    class Provider:
        def __init__(self, terms):
            self.terms = terms

        def get_term(self, name):
            return self.terms[name]

    old = Provider({"types": Sampler(0), "levels": Sampler(0)})
    term = TerrainRandomizationTerm("terrain", 1, runtime_context, curriculum_manager=old,
        category={"type": "Curriculum", "params": {"curriculum": "types", "values": {
            "plane": {}, "rough": {"max_height": .1}}}},
        difficulty={"type": "Curriculum", "params": {"curriculum": "levels", "values": [.25, 1.]}})
    replacement = Provider({"types": Sampler(1)})
    with pytest.raises(KeyError, match="levels"):
        term.config_update(replacement)
    assert term.category.sample(0, 17) == "plane"
    assert term.difficulty.sample(0, 17) == .25
    assert term.category._curriculum_samplers["types"] is old.terms["types"]
    assert term.difficulty._curriculum_samplers["levels"] is old.terms["levels"]


@pytest.mark.parametrize("field,value", [
    ("generations", None), ("allocation_ids", [[0], [1], [2]]),
])
def test_malformed_allocation_lists_are_rejected_without_mutation(runtime_context, field, value):
    from test_terrain_manager import make_manager
    from test_terrain_curriculum import assert_equal

    term, _ = make_manager(runtime_context)
    assert term.assignments is not None
    before = term.assignments.state_dict()
    saved = deepcopy(before)
    saved[field] = value
    with pytest.raises(ValueError):
        term.assignments.load_state_dict(saved)
    assert_equal(before, term.assignments.state_dict())

@pytest.mark.parametrize("axis,strategy", [("category", "Uniform"), ("difficulty", "Weighted")])
def test_category_and_difficulty_use_distribution_specific_strategy_names(axis, strategy):
    parser = TerrainCategoryConfig if axis == "category" else TerrainDifficultyConfig
    with pytest.raises(ValueError, match="Terrain axis type"):
        parser.from_mapping({"type": strategy})



def test_fixed_category_has_only_candidates_without_a_separate_value():
    config = TerrainCategoryConfig.from_mapping({"type": "Fixed", "params": {
        "values": {"slope": {"max_slope": .2}},
    }})
    assert config.initial_value == "slope"
    assert not hasattr(config, "value")
    assert config.to_mapping()["params"] == {"values": {"slope": {"max_slope": .2}}}
    with pytest.raises(ValueError, match="Unsupported parameters"):
        TerrainCategoryConfig.from_mapping({"type": "Fixed", "params": {
            "values": {"slope": {"max_slope": .2}}, "value": "slope",
        }})


def test_fixed_difficulty_rejects_multiple_candidates():
    with pytest.raises(ValueError, match="exactly one candidate"):
        TerrainDifficultyConfig.from_mapping({"type": "Fixed", "params": {"values": [.25, .6]}})


@pytest.mark.parametrize("strategy", ["Uniform", "Curriculum"])
def test_nonfixed_difficulty_keeps_multiple_candidates(strategy):
    params: dict[str, Any] = {"values": [.25, .6, 1.]}
    if strategy == "Curriculum":
        params["curriculum"] = "course"
    config = TerrainDifficultyConfig.from_mapping({"type": strategy, "params": params})
    assert config.values == (.25, .6, 1.)
    assert TerrainDifficultyConfig.from_mapping(config.to_mapping()) == config


def test_fixed_difficulty_requests_cannot_select_another_candidate(runtime_context):
    config = TerrainDifficultyConfig.from_mapping({"type": "Fixed", "params": {"values": [.6]}})
    term = TerrainDifficultyTerm("terrain_difficulty", config, runtime_context)
    with pytest.raises(ValueError, match="configured value"):
        term.request([0], torch.tensor([.25]))
    assert not term._requests
    term.request([0], torch.tensor([.6]))
    assert term.sample(0, 17) == .6
    term.commit_sample(0)
    assert term.sample(0, 99) == .6
    assert term._sample_counts == [0]
