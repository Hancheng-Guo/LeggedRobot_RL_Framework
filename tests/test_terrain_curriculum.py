"""Stage-four sampling, reset transactions and backend-neutral persistence."""
from copy import deepcopy
from dataclasses import replace
from typing import cast

import pytest
import torch

from envs.tasks.locomotion import LocomotionTaskLogic
from envs.tasks.managers.curriculum.terms.reward import BaseLpacReward
from envs.tasks.managers.randomization.terms.terrain import TerrainRandomizationTerm
from envs.simulators.terrain import TerrainSpawnGeometry
from utils import Component


class TerrainBackend:
    def __init__(self, context, num_envs):
        self.context = context
        self.num_envs = num_envs
        self.positions = torch.zeros(num_envs, 3, dtype=torch.float64)
        self.grids = [None] * num_envs

    def get_terrain_spawn_geometry(self):
        return TerrainSpawnGeometry((0., 0., 0.), (0., 0., .3), (-.2, -.1, -.31), (.3, .1, .2))

    def capture_reset_state(self, env_ids_tensor):
        return env_ids_tensor.clone(), self.positions[env_ids_tensor].clone(), [self.grids[i] for i in env_ids_tensor.tolist()]

    def restore_reset_state(self, state):
        env_ids_tensor, positions, grids = state
        self.positions[env_ids_tensor] = positions
        for env_id, grid in zip(env_ids_tensor.tolist(), grids):
            self.grids[env_id] = grid

    def apply_terrain(self, env_ids_tensor, patches):
        for env_id, patch in zip(env_ids_tensor.tolist(), patches):
            self.grids[env_id] = patch

    def reset(self, env_ids=None):
        if env_ids is None:
            self.positions.zero_()
        else:
            self.positions[env_ids] = 0

    def get_ground_heights(self, xy, env_ids=None):
        return xy.new_zeros(xy.shape[:2])

    def get_ground_bounds(self, env_ids=None):
        count = self.num_envs if env_ids is None else env_ids.numel()
        extent = self.positions.new_full((count, 2), 2.)
        return -extent, extent

    def set_root_positions(self, env_ids_tensor, positions):
        self.positions[env_ids_tensor] = positions


def make_task(context, model_context, *, sampling="Curriculum", coupled=True, separate=False, num_envs=3, axes=None, **sampling_options):
    policy = {"type": sampling, **sampling_options}
    courses = {}
    command = {"type": "Uniform", "params": {"min_value": -1., "max_value": 1.}}
    if sampling == "Curriculum":
        policy["curriculum"] = "motion"
        courses["motion"] = {"type": "fast_lpac_reward", "params": {
            "reward_sources": ["reward"], "min_samples_per_cell": 1, "min_coverage": .1,
        }}
        if coupled:
            command = {"type": "Curriculum", "params": {
                "curriculum": "motion", "min_value": -1., "max_value": 1., "num_cell": 2,
            }}
            if separate:
                courses["command_course"] = deepcopy(courses["motion"])
                command["params"]["curriculum"] = "command_course"
    axis_params = {"curriculum": "motion"} if sampling == "Curriculum" else {}
    policy = {
        "category": {"type": "Weighted" if sampling == "Uniform" else sampling, "params": dict(axis_params)},
        "difficulty": {"type": sampling, "params": dict(axis_params)},
    }
    if axes is not None:
        policy.update(axes)
        for axis in axes.values():
            reference = axis.get("params", {}).get("curriculum")
            if reference is not None and reference not in courses:
                courses[reference] = {"type": "fast_lpac_reward", "params": {
                    "reward_sources": ["reward"], "min_samples_per_cell": 1, "min_coverage": .1}}
    category_params = policy["category"].setdefault("params", {})
    category_params.setdefault("values", {
        "plane": {}, "slope": {"max_slope": .2}, "rough": {"max_height": .1},
    })
    if policy["category"]["type"] == "Fixed" and len(category_params["values"]) > 1:
        chosen = sampling_options.get("category", "rough")
        category_params["values"] = {chosen: category_params["values"][chosen]}
    difficulty_params = policy["difficulty"].setdefault("params", {})
    difficulty_params.setdefault("values", [.25] if policy["difficulty"]["type"] == "Fixed" else [.25, 1.])
    config = {
        "constants": {}, "task_state_manager_config": {},
        "action_manager_config": {"terms": {}},
        "command_manager_config": {"terms": {"speed": command}},
        "observation_manager_config": {"terms": {"base_height": {}}},
        "reward_manager_config": {"terms": {}}, "termination_manager_config": {"terms": {}},
        "curriculum_manager_config": {"terms": courses},
        "randomization_manager_config": {"terms": {"terrain": {
            "resolution": .5, "spawn_size": [1., 1.], "seed": 17,
            **policy,
            **{key: value for key, value in sampling_options.items() if key.startswith("resample_")},
        }}},
    }
    task = LocomotionTaskLogic(context)
    task.prepare(config=config)
    task.config_update(Component(None, None, None, None, None, None),
                       num_envs=num_envs, model_context=model_context, **config)
    backend = TerrainBackend(context, num_envs)
    task.bind(simulator=backend)
    return task, backend


def terrain(task):
    return cast(TerrainRandomizationTerm, task.randomization_manager.terms["terrain"])


def reset(task, backend, env_ids=None):
    try:
        task.before_simulator_reset(env_ids)
        backend.reset(env_ids)
        task.after_simulator_reset(env_ids)
        task.reset(env_ids)
        task.commit_simulator_reset(env_ids)
    except Exception:
        task.abort_simulator_reset(env_ids)
        raise


def curriculum_runtime_state(owner):
    terms = owner.terms if hasattr(owner, "terms") else {"term": owner}
    return {name: {
        "checkpoint": term.checkpoint_state_dict(),
        "assigned": term.buffer.assigned_cell_ids.clone(),
        "episode_reward": term.episode_reward_sum.clone(),
        "episode_steps": term.episode_step_count.clone(),
        "restored": term._restored_assignments.clone(),
    } for name, term in terms.items()}


def assert_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, equal_nan=True, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_equal(a, b)
    else:
        assert first == second


@pytest.mark.parametrize("sampling", ["Uniform", "Curriculum"])
def test_sampling_is_reproducible_for_the_same_selected_environment_order(runtime_context, model_context, sampling):
    first, a = make_task(runtime_context, model_context, sampling=sampling)
    second, b = make_task(runtime_context, model_context, sampling=sampling)
    for _ in range(8):
        torch.manual_seed(37)
        reset(first, a, torch.tensor([2, 0]))
        torch.manual_seed(37)
        reset(second, b, torch.tensor([2, 0]))
        assert_equal(first.checkpoint_state_dict(), second.checkpoint_state_dict())
        assert_equal(first.command_manager.command, second.command_manager.command)
        assert terrain(first).patches == terrain(second).patches
        assert terrain(first).patches[1] is None


@pytest.mark.parametrize("coupled", [False, True])
def test_terrain_and_command_consume_the_same_discrete_assignment(runtime_context, model_context, coupled):
    task, backend = make_task(runtime_context, model_context, coupled=coupled)
    reset(task, backend)
    term = terrain(task)
    assert term.assignments is not None
    course = cast(BaseLpacReward, task.curriculum_manager.get_term("motion"))
    expected = 12 if coupled else 6
    assert course.buffer.cell_starts.shape[0] == expected
    for name, config, choices in (("terrain_category", term.config.category, term.assignments.categories),
                                   ("terrain_difficulty", term.config.difficulty, term.assignments.difficulties)):
        lower, upper = course.get_bounds(name)
        assert [config.resolve_index(int(index)) for index in lower[:, 0].tolist()] == choices
        torch.testing.assert_close(lower, upper)
    if coupled:
        lower, upper = course.get_bounds("speed")
        value = task.command_manager.command["speed"]
        assert torch.all((value >= lower) & (value <= upper))


def test_explicit_patch_and_spawn_requests_do_not_change_other_environments(runtime_context, model_context):
    task, backend = make_task(runtime_context, model_context, sampling="Fixed", category="rough")
    reset(task, backend)
    term = terrain(task)
    original = term.patches
    positions = backend.positions.clone()
    reset(task, backend)
    assert term.patches == original
    term.request_spawn(torch.tensor([2]))
    reset(task, backend, torch.tensor([2]))
    assert term.patches == original
    assert not torch.equal(backend.positions[2], positions[2])
    torch.testing.assert_close(backend.positions[:2], positions[:2])
    term.request_reallocation(torch.tensor([0]))
    reset(task, backend, torch.tensor([0]))
    assert term.patches[0] != original[0]
    assert term.patches[1:] == original[1:]


@pytest.mark.parametrize("failure", ["apply_terrain", "reset", "set_root_positions", "command"])
def test_failed_reset_restores_course_physics_requests_and_rng(runtime_context, model_context, monkeypatch, failure):
    task, backend = make_task(runtime_context, model_context)
    control, control_backend = make_task(runtime_context, model_context)
    for instance, physical in ((task, backend), (control, control_backend)):
        torch.manual_seed(19)
        reset(instance, physical)
        instance.curriculum_manager.update(raw_term_rewards={}, term_rewards={}, total_reward=torch.tensor([1., 2., 3.]))
        terrain(instance).request_reallocation(torch.tensor([2]))
    state = deepcopy(task.checkpoint_state_dict())
    course_state = curriculum_runtime_state(task.curriculum_manager)
    positions = backend.positions.clone()
    grids = backend.grids.copy()
    target = task.command_manager if failure == "command" else backend
    method_name = "reset" if failure == "command" else failure
    original = getattr(target, method_name)
    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected reset failure")
    monkeypatch.setattr(target, method_name, fail)
    torch.manual_seed(91)
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="injected reset failure"):
        reset(task, backend, torch.tensor([2, 0]))
    assert_equal(state, task.checkpoint_state_dict())
    assert_equal(course_state, curriculum_runtime_state(task.curriculum_manager))
    assert_equal(positions, backend.positions)
    assert backend.grids == grids
    assert torch.equal(torch.get_rng_state(), rng)
    monkeypatch.setattr(target, method_name, original)
    reset(task, backend, torch.tensor([2, 0]))
    torch.manual_seed(91)
    reset(control, control_backend, torch.tensor([2, 0]))
    assert_equal(task.checkpoint_state_dict(), control.checkpoint_state_dict())
    assert terrain(task).patches == terrain(control).patches


@pytest.mark.parametrize("sampling", ["Fixed", "Uniform", "Curriculum"])
def test_checkpoint_rebuild_preserves_current_and_next_assignments(runtime_context, model_context, tmp_path, sampling):
    source, a = make_task(runtime_context, model_context, sampling=sampling, resample_spawn=True)
    reset(source, a)
    reset(source, a, torch.tensor([1]))
    terrain(source).request_reallocation(torch.tensor([2]))
    path = tmp_path / "terrain.pt"
    torch.save(source.checkpoint_state_dict(), path)
    saved = torch.load(path, weights_only=True)
    restored, b = make_task(runtime_context, model_context, sampling=sampling, resample_spawn=True)
    restored.load_checkpoint_state_dict(saved)
    reset(restored, b)
    assert_equal(saved, restored.checkpoint_state_dict())
    assert terrain(source).patches == terrain(restored).patches
    assert_equal(a.positions, b.positions)
    for instance, physical in ((source, a), (restored, b)):
        torch.manual_seed(123)
        reset(instance, physical)
    assert_equal(source.checkpoint_state_dict(), restored.checkpoint_state_dict())
    assert terrain(source).patches == terrain(restored).patches


def test_evaluation_freezes_joint_assignment_and_pending_requests(runtime_context, model_context):
    task, backend = make_task(runtime_context, model_context, resample_patch=True, resample_spawn=True)
    reset(task, backend)
    terrain(task).request_reallocation()
    before = deepcopy(task.checkpoint_state_dict())
    positions = backend.positions.clone()
    task.set_training_mode(False)
    for _ in range(3):
        reset(task, backend)
        assert_equal(before, task.checkpoint_state_dict())
        assert_equal(positions, backend.positions)


def test_playback_projection_keeps_original_patch_identity(runtime_context, model_context):
    task, backend = make_task(runtime_context, model_context)
    reset(task, backend)
    terrain(task).request_reallocation(torch.tensor([2]))
    reset(task, backend, torch.tensor([2]))
    playback, physical = make_task(runtime_context, model_context, num_envs=1)
    playback.load_checkpoint_state_dict(task.playback_state_dict(2))
    playback.set_training_mode(False)
    reset(playback, physical)
    assert terrain(playback).patches[0] == terrain(task).patches[2]
    assert_equal(physical.positions[0], backend.positions[2])


@pytest.mark.parametrize("field", ["seed", "num_envs", "version", "config", "allocations"])
def test_invalid_terrain_restore_is_rejected_before_any_course_mutation(runtime_context, model_context, field):
    task, backend = make_task(runtime_context, model_context)
    reset(task, backend)
    state = deepcopy(task.checkpoint_state_dict())
    state["randomization"]["terms"]["terrain"][field] = None
    before = task.checkpoint_state_dict()
    with pytest.raises((ValueError, TypeError)):
        task.load_checkpoint_state_dict(state)
    assert_equal(before, task.checkpoint_state_dict())


def test_separate_instances_keep_distinct_spaces_and_sampling_rng(runtime_context, model_context):
    task, backend = make_task(runtime_context, model_context, separate=True)
    reset(task, backend)
    course = cast(BaseLpacReward, task.curriculum_manager.get_term("motion"))
    command = cast(BaseLpacReward, task.curriculum_manager.get_term("command_course"))
    assert set(course.buffer.term_slices) == {"terrain_category", "terrain_difficulty"}
    assert set(command.buffer.term_slices) == {"speed"}
    before = curriculum_runtime_state(course)
    command.finalize_episodes()
    command.resample()
    assert_equal(before, curriculum_runtime_state(course))


def test_uniform_covers_conditions_without_consuming_training_rng(runtime_context, model_context):
    task, backend = make_task(runtime_context, model_context, sampling="Uniform")
    term = terrain(task)
    seen = set()
    torch.manual_seed(278)
    rng = torch.get_rng_state().clone()
    for _ in range(50):
        term.before_simulator_reset()
        backend.reset()
        term.after_simulator_reset()
        term.commit_simulator_reset()
        seen.update((patch.terrain_type, patch.difficulty) for patch in term.patches if patch is not None)
    assert len(seen) == 6
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("options", [
    {"type": "invalid"}, {"type": "Curriculum"},
    {"type": "Weighted", "curriculum": "course"},
    {"type": "Weighted", "difficulty_index": 0},
    {"resample_patch": 1}, {"resample_spawn": "yes"},
    {"difficulty_index": -1}, {"unknown": True},
])
def test_sampling_configuration_rejects_invalid_options(options):
    from envs.tasks.managers.randomization.terms.terrain_category import TerrainCategoryConfig
    with pytest.raises((TypeError, ValueError)):
        TerrainCategoryConfig.from_mapping(options)


@pytest.mark.parametrize("field", ["config", "terms", "sampling_rng"])
def test_malformed_course_restore_does_not_mutate_terrain(runtime_context, model_context, field):
    task, backend = make_task(runtime_context, model_context)
    reset(task, backend)
    state = deepcopy(task.checkpoint_state_dict())
    if field == "terms":
        state["curriculum"]["terms"] = None
    else:
        state["curriculum"]["terms"]["motion"][field] = None
    before = task.checkpoint_state_dict()
    with pytest.raises((TypeError, ValueError)):
        task.load_checkpoint_state_dict(state)
    assert_equal(before, task.checkpoint_state_dict())


@pytest.mark.parametrize("sampling", ["Uniform", "Curriculum"])
def test_plane_only_sampling_needs_no_nonflat_backend(runtime_context, model_context, sampling):
    from envs.tasks.managers.randomization import RandomizationManager
    config = {"terms": {"terrain": {'category': {'type': 'Weighted' if sampling == 'Uniform' else sampling, 'params': {'values': {'plane': {}}}}, 'difficulty': {'type': sampling, 'params': {'values': [0.5, 1.0]}}}}}
    if sampling == "Curriculum":

        for axis in config['terms']['terrain'].values():
            axis['params']['curriculum'] = 'flat'
    provider = None
    if sampling == "Curriculum":
        class Sampler:
            def get_bounds(self, dimension, env_ids=None):
                values = torch.zeros(1, 1)
                return values, values
        class Provider:
            def get_term(self, name):
                return Sampler()
        provider = Provider()
    manager = RandomizationManager(3, runtime_context, curriculum_manager=provider, **config)
    term = cast(TerrainRandomizationTerm, manager.terms["terrain"])
    class FlatBackend:
        num_envs = 3
        context = runtime_context
    term.bind(FlatBackend())
    term.before_simulator_reset()
    term.after_simulator_reset()
    term.commit_simulator_reset()
    assert len(term.patches) == 3
    assert all(patch is not None and patch.terrain_type == 'plane' for patch in term.patches)
    assert term.geometry is None


def test_curriculum_difficulty_cannot_be_overridden_during_reset(runtime_context, model_context):
    task, backend = make_task(runtime_context, model_context)
    with pytest.raises(ValueError, match="owns terrain difficulty"):
        terrain(task).set_difficulty(torch.tensor([0]), torch.tensor([1.]))


def test_reset_snapshot_restores_replaced_tensors_and_nested_containers():
    from envs.tasks.utils.reset import capture_task_reset_state, ResetStateOwner, ResetStateManager
    class Term(ResetStateOwner):
        history: list[list[int]]
        tensor: torch.Tensor
        buffers: dict[str, torch.Tensor]
        RESET_STATE_FIELDS = ("history", "tensor", "buffers")
    class Manager(ResetStateManager):
        pass
    tensor = torch.tensor([1., 2.])
    nested = [1, 2]
    term = Term()
    term.history, term.tensor, term.buffers = [nested], tensor, {'value': tensor}
    manager = Manager()
    manager.terms = {'term': term}
    saved = capture_task_reset_state([manager])
    term.tensor = torch.zeros(2)
    tensor.zero_()
    nested.clear()
    term.history = []
    term.buffers.clear()
    saved.restore()
    assert term.tensor is tensor and term.buffers['value'] is tensor
    torch.testing.assert_close(tensor, torch.tensor([1., 2.]))
    assert term.history == [[1, 2]] and term.history[0] is nested


def test_standalone_task_reset_does_not_resample_physical_curriculum(runtime_context, model_context):
    task, backend = make_task(runtime_context, model_context)
    reset(task, backend)
    before = task.checkpoint_state_dict()
    task.reset()
    assert_equal(before, task.checkpoint_state_dict())


def test_reset_snapshots_capture_only_explicit_fields():
    from envs.tasks.utils.reset import ResetStateOwner
    class Owner(ResetStateOwner):
        RESET_STATE_FIELDS = ('buffer',)
        def __init__(self):
            self.buffer = torch.ones(2)
            self.configuration = torch.ones(2)
    owner = Owner()
    state = owner.capture_reset_state()
    owner.buffer.zero_()
    owner.configuration.zero_()
    state.restore()
    assert torch.equal(owner.buffer, torch.ones(2))
    assert torch.equal(owner.configuration, torch.zeros(2))


def test_persistence_is_declared_in_terrain_space(runtime_context, model_context):
    task, backend = make_task(runtime_context, model_context)
    assert task.curriculum_manager.sampling_spaces['motion'].persistent_assignments
    course = task.curriculum_manager.get_term('motion')
    assert course.persistent_assignments


def test_task_checkpoint_composes_and_restores_base_state(runtime_context, model_context, monkeypatch):
    from envs.tasks.base import BaseTaskLogic
    task, backend = make_task(runtime_context, model_context)
    reset(task, backend)
    monkeypatch.setattr(BaseTaskLogic, 'checkpoint_state_dict', lambda self: {'base_statistics': 23})
    received = []
    monkeypatch.setattr(BaseTaskLogic, 'load_checkpoint_state_dict', lambda self, state: received.append(state))
    state = task.checkpoint_state_dict()
    assert state['base_statistics'] == 23
    task.load_checkpoint_state_dict(state)
    assert received == [{'base_statistics': 23}]


def test_task_playback_composes_base_projection(runtime_context, model_context, monkeypatch):
    from envs.tasks.base import BaseTaskLogic
    task, backend = make_task(runtime_context, model_context)
    reset(task, backend)
    monkeypatch.setattr(BaseTaskLogic, 'playback_state_dict',
                        lambda self, env_index: {'base_assignment': env_index})
    state = task.playback_state_dict(2)
    assert state['base_assignment'] == 2
    assert 'curriculum' in state and 'randomization' in state


@pytest.mark.parametrize('reverse', [False, True])
def test_space_merge_preserves_assignment_policy_in_either_order(runtime_context, reverse):
    from envs.tasks.managers.curriculum.spaces import build_curriculum_spaces
    from envs.tasks.managers.curriculum.types import CurriculumSpaceConfig, DiscreteDimensionConfig
    contributions = [
        {'joint': CurriculumSpaceConfig(dimensions={'command': DiscreteDimensionConfig((0., 1.))})},
        {'joint': CurriculumSpaceConfig(dimensions={'terrain': DiscreteDimensionConfig((0., 1.))},
                                      persistent_assignments=True)},
    ]
    if reverse:
        contributions.reverse()
    spaces = build_curriculum_spaces(contributions, ['joint'], runtime_context)
    assert spaces['joint'].persistent_assignments
    assert spaces['joint'].cell_starts.shape[0] == 4


def test_independent_command_space_keeps_default_assignment_policy(runtime_context, model_context):
    task, backend = make_task(runtime_context, model_context, separate=True)
    assert not task.curriculum_manager.sampling_spaces['command_course'].persistent_assignments
    assert not task.curriculum_manager.get_term('command_course').persistent_assignments
    assert task.curriculum_manager.get_term('motion').persistent_assignments


def test_randomization_dependency_refresh_preserves_allocations_and_backend(runtime_context, model_context):
    task, backend = make_task(runtime_context, model_context)
    replacement, other_backend = make_task(runtime_context, model_context)
    reset(task, backend)
    reset(replacement, other_backend)
    manager = task.randomization_manager
    term = terrain(task)
    before = manager.checkpoint_state_dict()
    patches = term.patches
    manager.config_update(curriculum_manager=replacement.curriculum_manager)
    assert manager.terms['terrain'] is term
    assert term.simulator is backend
    assert term.difficulty._curriculum_samplers['motion'] is replacement.curriculum_manager.get_term('motion')
    assert term.patches == patches
    assert_equal(before, manager.checkpoint_state_dict())


def test_randomization_constructor_requires_provider_only_for_course_axes(runtime_context):
    from envs.tasks.managers.randomization import RandomizationManager
    terms = {'terrain': {'difficulty': {'type': 'Curriculum', 'params': {'curriculum': 'levels'}}}}
    with pytest.raises(ValueError, match='requires a curriculum provider'):
        RandomizationManager(3, runtime_context, terms=terms)


def test_dependency_resolution_does_not_build_space_descriptions(runtime_context, model_context, monkeypatch):
    from envs.tasks.managers.randomization import RandomizationManager
    task, backend = make_task(runtime_context, model_context)
    def unexpected_space_description(cls, name, params):
        raise AssertionError('Dependency resolution must not construct a space description')
    monkeypatch.setattr(TerrainRandomizationTerm, 'get_curriculum_space_config',
                        classmethod(unexpected_space_description))
    manager = RandomizationManager(3, runtime_context, terms={'terrain': {'difficulty': {'type': 'Curriculum', 'params': {'curriculum': 'motion'}}}}, curriculum_manager=task.curriculum_manager)
    manager.config_update(curriculum_manager=task.curriculum_manager)
    term = cast(TerrainRandomizationTerm, manager.terms['terrain'])
    assert term.get_curriculum_references({'difficulty': {'type': 'Curriculum', 'params': {'curriculum': 'motion'}}}) == {'motion'}
    assert term.difficulty._curriculum_samplers['motion'] is task.curriculum_manager.get_term('motion')


def test_randomization_manager_only_forwards_provider(runtime_context, monkeypatch):
    from envs.tasks.managers.randomization import RandomizationManager
    from envs.tasks.managers.randomization.terms.base import BaseRandomizationTerm
    received = []
    class Provider:
        def get_term(self, name):
            raise AssertionError('The manager must not resolve the provider')
    class Consumer(BaseRandomizationTerm):
        def matches_config(self, params):
            return not params

        def __init__(self, *args, curriculum_manager=None, **kwargs):
            received.append(curriculum_manager)
            super().__init__(*args, **kwargs)
        def config_update(self, *, curriculum_manager, **kwargs):
            received.append(curriculum_manager)
    monkeypatch.setattr('envs.tasks.managers.randomization.base.get_randomization_class', lambda name: Consumer)
    provider = Provider()
    manager = RandomizationManager(3, runtime_context, terms={'consumer': {}}, curriculum_manager=provider)
    manager.config_update(curriculum_manager=provider)
    assert received == [provider, provider]


def test_task_delegates_playback_schema_to_managers(runtime_context, model_context, monkeypatch):
    task, backend = make_task(runtime_context, model_context)
    reset(task, backend)
    # Opaque manager payloads must not be inspected or rewritten by the task.
    course_state = {"custom_course_state": "opaque"}
    randomization_state = {"custom_randomization_state": "opaque"}
    monkeypatch.setattr(task.curriculum_manager, 'playback_state_dict', lambda index: course_state)
    monkeypatch.setattr(task.randomization_manager, 'playback_state_dict', lambda index: randomization_state)
    state = task.playback_state_dict(2)
    assert state['curriculum'] is course_state
    assert state['randomization'] is randomization_state


def test_task_restores_buffers_even_when_course_rollback_fails(runtime_context, model_context):
    task, backend = make_task(runtime_context, model_context)
    reset(task, backend)
    torch.manual_seed(786)
    rng = torch.get_rng_state().clone()
    before = deepcopy(task.command_manager.command)
    task.before_simulator_reset()
    backend.reset()
    task.after_simulator_reset()
    task.reset()
    # Break one captured tensor to prove later owners and RNGs still restore.
    course = cast(BaseLpacReward, task.curriculum_manager.get_term('motion'))
    course.episode_reward_sum.resize_(1)
    with pytest.raises(ExceptionGroup, match='Reset snapshot restoration failed'):
        task.abort_simulator_reset()
    assert task._pending_reset is None
    assert_equal(before, task.command_manager.command)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize('type_mode,difficulty_mode', [
    ('Weighted', 'Curriculum'), ('Fixed', 'Curriculum'),
    ('Curriculum', 'Fixed'), ('Curriculum', 'Uniform'),
    ('Weighted', 'Uniform'), ('Curriculum', 'Curriculum'),
])
def test_independent_axis_policies_and_restore(runtime_context, model_context, type_mode, difficulty_mode):
    axes = {}
    for name, mode in (('category', type_mode), ('difficulty', difficulty_mode)):
        params = {'curriculum': 'types' if name == 'category' else 'levels'} if mode == 'Curriculum' else {}
        if mode == 'Fixed':
            params = {'values': {'rough': {'max_height': .1}}} if name == 'category' else {'values': [1.]}
        axes[name] = {'type': mode, 'params': params}
    task, backend = make_task(runtime_context, model_context, sampling='Fixed', axes=axes)
    term = terrain(task)
    assert term.get_curriculum_references(axes) == {
        axis['params']['curriculum'] for axis in axes.values() if axis['type'] == 'Curriculum'}
    for _ in range(8):
        reset(task, backend)
        for name, mode, dimension in (('category', type_mode, 'terrain_category'),
                                      ('difficulty', difficulty_mode, 'terrain_difficulty')):
            if mode == 'Curriculum':
                course = cast(BaseLpacReward, task.curriculum_manager.get_term(axes[name]['params']['curriculum']))
                assert set(course.buffer.term_slices) == {dimension}
            elif mode == 'Fixed':
                for patch in term.patches:
                    assert patch is not None
                    assert (patch.terrain_type == 'rough' if name == 'category' else patch.difficulty == 1.)
    state = task.checkpoint_state_dict()
    restored, physical = make_task(runtime_context, model_context, sampling='Fixed', axes=axes)
    restored.load_checkpoint_state_dict(state)
    reset(restored, physical)
    assert_equal(state, restored.checkpoint_state_dict())
    for _ in range(3):
        reset(task, backend)
        reset(restored, physical)
        assert_equal(task.checkpoint_state_dict(), restored.checkpoint_state_dict())
        assert term.patches == terrain(restored).patches


def test_only_difficulty_course_keeps_initial_fixed_type(runtime_context, model_context):
    task, backend = make_task(runtime_context, model_context, sampling='Fixed', axes={
        'difficulty': {'type': 'Curriculum', 'params': {'curriculum': 'levels'}}})
    reset(task, backend)
    original = [patch.terrain_type for patch in terrain(task).patches if patch is not None]
    for _ in range(12):
        reset(task, backend)
        assert [patch.terrain_type for patch in terrain(task).patches if patch is not None] == original
