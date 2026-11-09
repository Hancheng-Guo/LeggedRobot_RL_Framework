"""Exercise terrain decisions through a backend-neutral fake physical interface."""
from dataclasses import asdict, replace
from collections.abc import Sequence
from typing import Any, cast

import pytest
import torch

from envs.simulators.terrain import TerrainPatch, TerrainSpawnGeometry
from envs.tasks.managers.randomization.terms.terrain import TerrainConfig, TerrainRandomizationTerm
from envs.tasks import BaseTaskLogic
from envs.tasks.managers.randomization import RandomizationManager
from utils import Component


class PhysicalBackend:
    def __init__(self, context, num_envs=3):
        self.context = context
        self.num_envs = num_envs
        self.events = []
        self.positions = torch.zeros(num_envs, 3, dtype=torch.float64)
        self.measured = TerrainSpawnGeometry((4., -3., .2), (0., 0., .3),
                                             (-.2, -.1, -.31), (.3, .1, .2))

    def get_terrain_spawn_geometry(self) -> TerrainSpawnGeometry:
        self.events.append("measure")
        return self.measured

    def apply_terrain(self, env_ids_tensor: torch.Tensor, patches: Sequence[TerrainPatch]) -> None:
        self.events.append(("apply", env_ids_tensor.tolist(), list(patches)))

    def capture_reset_state(self, env_ids_tensor):
        return self.positions.clone()

    def restore_reset_state(self, state):
        self.positions.copy_(state)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.events.append(("reset", None if env_ids is None else env_ids.tolist()))

    def get_ground_heights(self, xy: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        self.events.append("heights")
        return torch.full(xy.shape[:2], .2, dtype=xy.dtype, device=xy.device)

    def set_root_positions(self, env_ids_tensor: torch.Tensor, positions: torch.Tensor) -> None:
        self.events.append(("positions", env_ids_tensor.tolist()))
        self.positions[env_ids_tensor] = positions


def make_manager(context, num_envs=3):
    config = TerrainConfig.from_mapping(dict(types={"rough": {"max_height": .05}},
        spawn_size=[1, 1], difficulty_levels=[.25, 1], seed=17))
    manager = TerrainRandomizationTerm(num_envs, context, **asdict(config))
    backend = PhysicalBackend(context, num_envs)
    manager.bind(backend)
    backend.events.clear()
    return manager, backend


def reset_from_environment(term, backend, env_ids=None, difficulty_indices=None):
    try:
        term.before_simulator_reset(env_ids, difficulty_indices)
        backend.reset(env_ids)
        term.after_simulator_reset(env_ids)
        term.commit_simulator_reset(env_ids)
    except Exception:
        term.abort_simulator_reset(env_ids)
        raise


def test_manager_coordinates_geometry_reset_and_safe_spawn(runtime_context):
    manager, backend = make_manager(runtime_context)
    original = manager.patches
    reset_from_environment(manager, backend, torch.tensor([2]), torch.tensor([1]))
    assert manager.patches[2] is not None
    assert backend.events == [("apply", [2], [manager.patches[2].geometry]), ("reset", [2]), "heights", ("positions", [2])]
    assert manager.patches[:2] == original[:2]
    assert manager.patches[2] is not None
    assert manager.patches[2].difficulty == 1
    assert 3.7 <= backend.positions[2, 0] <= 4.2
    assert -3.4 <= backend.positions[2, 1] <= -2.6
    assert backend.positions[2, 2] == pytest.approx(.512)


def test_manager_sampling_is_reproducible_and_local(runtime_context):
    first, a = make_manager(runtime_context)
    second, b = make_manager(replace(runtime_context, seed=999))
    reset_from_environment(first, a, torch.tensor([2, 0]))
    reset_from_environment(second, b, torch.tensor([0, 2]))
    assert first.patches == second.patches
    torch.testing.assert_close(a.positions, b.positions, rtol=0, atol=0)


def test_manager_rejects_invalid_choices_before_backend_mutation(runtime_context):
    manager, backend = make_manager(runtime_context)
    assert manager.assignments is not None
    before = manager.assignments.state_dict()
    with pytest.raises(ValueError):
        manager.before_simulator_reset(torch.tensor([1]), torch.tensor([2]))
    assert manager.assignments.state_dict() == before
    assert backend.events == []


def test_manager_validates_spawn_space_without_engine_objects(runtime_context):
    config = TerrainConfig.from_mapping(dict(types={"rough": {"max_height": .05}}))
    manager = TerrainRandomizationTerm(3, runtime_context, **asdict(config))
    backend = PhysicalBackend(runtime_context)
    backend.measured = replace(backend.measured, collision_upper=(.4, .1, .2))
    with pytest.raises(ValueError, match="collision footprint"):
        manager.bind(backend)
    assert backend.events == ["measure"]


def test_legacy_plane_does_not_measure_allocate_or_move_spawn(runtime_context):
    manager = TerrainRandomizationTerm(3, runtime_context)
    backend = PhysicalBackend(runtime_context)
    manager.bind(backend)
    manager.before_simulator_reset()
    manager.after_simulator_reset()
    assert manager.geometry is None and manager.assignments is None and manager.patches == ()
    assert backend.events == []


def test_task_randomization_owns_policy_and_preserves_unchanged_configuration(runtime_context, model_context):
    task = BaseTaskLogic(runtime_context)
    config: dict[str, Any] = {"terms": {"terrain": dict(types={"rough": {"max_height": .05}}, spawn_size=[1, 1])}}
    task.prepare({"randomization_manager_config": config})
    assert not hasattr(task, "randomization_manager")
    assert task.get_simulator_requirements()["terrain_geometry"] is not None

    def build_task():
        task.config_update(Component(None, None, None, None, None, None),
            num_envs=3, model_context=model_context, constants={},
            randomization_manager_config=config,
            action_manager_config={"terms": {}}, command_manager_config={"terms": {}},
            observation_manager_config={"terms": {"base_height": {}}}, reward_manager_config={"terms": {}},
            termination_manager_config={"terms": {}})

    build_task()
    manager = task.randomization_manager
    terrain = cast(TerrainRandomizationTerm, manager.terms["terrain"])
    assert terrain.geometry == task.get_simulator_requirements()["terrain_geometry"]
    backend = PhysicalBackend(runtime_context)
    task.bind(backend)
    task.before_simulator_reset(torch.tensor([1]))
    assert not any(isinstance(event, tuple) and event[0] == "reset" for event in backend.events)
    backend.reset(torch.tensor([1]))
    task.after_simulator_reset(torch.tensor([1]))
    patches = terrain.patches
    # Explicit default weights and omitted weights describe the same policy.
    config["terms"]["terrain"]["types"]["rough"]["weight"] = 1
    task.prepare({"randomization_manager_config": config})
    build_task()
    assert task.randomization_manager is manager
    assert terrain.patches == patches
    assert [event[0] for event in backend.events if isinstance(event, tuple)].count("reset") == 1


def test_manager_configuration_matching_uses_effective_term_values(runtime_context):
    terms = {"terrain": dict(types={"rough": {"max_height": .05}}, spawn_size=[1, 1])}
    manager = RandomizationManager(3, runtime_context, terms=terms)
    assert manager.matches_config(3, terms=terms)
    changed = {"terrain": dict(terms["terrain"], types={"rough":{"max_height":.1}})}
    assert not manager.matches_config(3, terms=changed)
    assert not manager.matches_config(4, terms=terms)
    assert not manager.matches_config(3)


def test_layout_declaration_does_not_construct_randomization_terms(monkeypatch):
    def unexpected_construction(*args, **kwargs):
        raise AssertionError("Resource declaration must not construct a term")

    monkeypatch.setattr(TerrainRandomizationTerm, "__init__", unexpected_construction)
    monkeypatch.setattr(RandomizationManager, "__init__", unexpected_construction)
    config = {"terrain": dict(types={"rough": {"max_height": .05}}, spawn_size=[1, 1])}
    requirements = RandomizationManager.get_simulator_requirements(terms=config)
    assert requirements["terrain_geometry"] is not None


@pytest.mark.parametrize("terms", [{"friction_typo": {}}, {"terrain": {"types": {"unknown":{}}}}])
def test_task_randomization_rejects_invalid_terms(runtime_context, terms):
    with pytest.raises(ValueError):
        RandomizationManager(3, runtime_context, terms=terms)


def test_empty_task_randomization_leaves_physics_reset_to_environment(runtime_context):
    manager = RandomizationManager(3, runtime_context)
    backend = PhysicalBackend(runtime_context)
    manager.bind(backend)
    manager.before_simulator_reset()
    manager.after_simulator_reset()
    assert manager.terms == {}
    assert backend.events == []


def test_bind_does_not_generate_or_apply_terrain(runtime_context, monkeypatch):
    def unexpected(*args):
        raise AssertionError("bind must not sample a patch")
    monkeypatch.setattr("envs.tasks.managers.randomization.terms.terrain.generate_terrain_patch", unexpected)
    term, backend = make_manager(runtime_context)
    assert term.patches == (None, None, None)
    assert term.assignments is not None
    assert term.assignments.generations == [0, 0, 0]
    assert backend.events == []


def test_registry_dispatch_and_rollback_across_terms(runtime_context, monkeypatch):
    from envs.tasks.managers.randomization.terms import BaseRandomizationTerm, RANDOMIZATION_CLASS_MAP
    events = []

    class Probe(BaseRandomizationTerm):
        @classmethod
        def normalize_config(cls, params):
            return dict(params)

        @classmethod
        def get_simulator_requirements(cls, params):
            return {"probe_resource": params["value"]}

        def bind(self, backend):
            events.append("bind")

        def before_simulator_reset(self, env_ids=None):
            events.append("before")

        def after_simulator_reset(self, env_ids=None):
            events.append("after")
            if self.config.get("fail"):
                raise RuntimeError("probe failed")

        def commit_simulator_reset(self, env_ids=None):
            events.append("commit")

        def abort_simulator_reset(self, env_ids=None):
            events.append("abort")

    monkeypatch.setitem(RANDOMIZATION_CLASS_MAP, "probe", Probe)
    terrain = dict(types={"rough": {"max_height": .05}}, spawn_size=[1, 1])
    terms = {"terrain": terrain, "probe": {"value": 7, "fail": True}}
    assert RandomizationManager.get_simulator_requirements(terms)["probe_resource"] == 7
    manager = RandomizationManager(3, runtime_context, terms)
    backend = PhysicalBackend(runtime_context)
    manager.bind(backend)
    term = cast(TerrainRandomizationTerm, manager.terms["terrain"])
    assert term.assignments is not None
    old = term.assignments.state_dict()
    manager.before_simulator_reset(torch.tensor([1]))
    backend.reset(torch.tensor([1]))
    with pytest.raises(RuntimeError, match="probe failed"):
        manager.after_simulator_reset(torch.tensor([1]))
    assert events == ["bind", "before", "after", "abort"]
    assert term.assignments.state_dict() == old and term.patches == (None, None, None)
    assert torch.count_nonzero(backend.positions) == 0
    manager.terms["probe"].config["fail"] = False
    manager.before_simulator_reset(torch.tensor([1]))
    backend.reset(torch.tensor([1]))
    manager.after_simulator_reset(torch.tensor([1]))
    assert events[-3:] == ["before", "after", "commit"]
    assert term.assignments.generations == [0, 1, 0]


def test_requirements_reject_conflicting_resources(runtime_context, monkeypatch):
    from envs.tasks.managers.randomization.terms import BaseRandomizationTerm, RANDOMIZATION_CLASS_MAP

    class Resource(BaseRandomizationTerm):
        @classmethod
        def normalize_config(cls, params):
            return dict(params)

        @classmethod
        def get_simulator_requirements(cls, params):
            return {"shared_resource": params["value"]}

    monkeypatch.setitem(RANDOMIZATION_CLASS_MAP, "first", Resource)
    monkeypatch.setitem(RANDOMIZATION_CLASS_MAP, "second", Resource)
    with pytest.raises(ValueError, match="Conflicting simulator requirement"):
        RandomizationManager.get_simulator_requirements({"first": {"value": 1}, "second": {"value": 2}})


def test_physical_patch_contains_only_geometry():
    from dataclasses import fields
    assert [f.name for f in fields(TerrainPatch)] == ["x", "y", "heights"]


@pytest.mark.parametrize("ids", [None, torch.tensor([2, 0])], ids=["full", "partial"])
def test_episode_reset_reuses_existing_terrain(runtime_context, monkeypatch, ids):
    term, backend = make_manager(runtime_context)
    reset_from_environment(term, backend)
    assert term.assignments is not None
    patches, state = term.patches, term.assignments.state_dict()

    def unexpected_generation(*args, **kwargs):
        raise AssertionError("Ordinary reset must not generate terrain")

    monkeypatch.setattr("envs.tasks.managers.randomization.terms.terrain.generate_terrain_patch", unexpected_generation)
    backend.events.clear()
    reset_from_environment(term, backend, ids)
    assert term.assignments.state_dict() == state
    assert all(a is b for a, b in zip(term.patches, patches))
    assert not any(isinstance(event, tuple) and event[0] == "apply" for event in backend.events)
    assert any(isinstance(event, tuple) and event[0] == "reset" for event in backend.events)
    assert any(isinstance(event, tuple) and event[0] == "positions" for event in backend.events)


def test_only_changed_difficulty_regenerates_selected_terrain(runtime_context):
    term, backend = make_manager(runtime_context)
    reset_from_environment(term, backend)
    assert term.assignments is not None
    patches = term.patches
    backend.events.clear()
    term.set_difficulty(torch.tensor([2, 0]), torch.tensor([1, 0]))
    reset_from_environment(term, backend, torch.tensor([2, 0]))
    assert term.assignments.generations == [1, 1, 2]
    assert term.patches[0] is patches[0] and term.patches[1] is patches[1]
    assert term.patches[2] is not patches[2]
    applies = [event for event in backend.events if isinstance(event, tuple) and event[0] == "apply"]
    assert len(applies) == 1 and applies[0][1] == [2]
    backend.events.clear()
    term.set_difficulty(torch.tensor([2]), torch.tensor([1]))
    reset_from_environment(term, backend, torch.tensor([2]))
    assert term.assignments.generations == [1, 1, 2]
    assert not any(isinstance(event, tuple) and event[0] == "apply" for event in backend.events)


def test_rebind_installs_cached_geometry_without_regeneration(runtime_context, monkeypatch):
    term, backend = make_manager(runtime_context)
    reset_from_environment(term, backend)
    assert term.assignments is not None
    patches, state = term.patches, term.assignments.state_dict()
    replacement = PhysicalBackend(runtime_context)
    term.bind(replacement)

    def unexpected_generation(*args, **kwargs):
        raise AssertionError("Rebinding must preserve the assigned terrain")

    monkeypatch.setattr("envs.tasks.managers.randomization.terms.terrain.generate_terrain_patch", unexpected_generation)
    reset_from_environment(term, replacement, torch.tensor([1]))
    reset_from_environment(term, replacement, torch.tensor([2, 0]))
    assert term.assignments.state_dict() == state
    assert all(a is b for a, b in zip(term.patches, patches))
    applies = [event for event in replacement.events if isinstance(event, tuple) and event[0] == "apply"]
    assert [event[1] for event in applies] == [[1], [2, 0]]
    replacement.events.clear()
    reset_from_environment(term, replacement)
    assert not any(isinstance(event, tuple) and event[0] == "apply" for event in replacement.events)


@pytest.mark.parametrize("ids", [torch.tensor([-1]), torch.tensor([3]),
    torch.tensor([1, 1]), torch.tensor([0.]), torch.tensor([[0]])])
def test_term_rejects_invalid_ids_before_any_physical_operation(runtime_context, ids):
    term, backend = make_manager(runtime_context)
    assert term.assignments is not None
    before = term.assignments.state_dict()
    with pytest.raises(ValueError, match="Environment IDs"):
        term.before_simulator_reset(ids)
    assert term.assignments.state_dict() == before
    assert term.patches == (None, None, None)
    assert backend.events == []
