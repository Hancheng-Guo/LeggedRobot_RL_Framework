"""Behavioral boundaries between consumers, candidate spaces and learning."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from envs.tasks.managers.command import CommandManager
from envs.tasks.managers.curriculum.spaces import build_curriculum_spaces
from envs.tasks.managers.curriculum import CurriculumManager
from envs.tasks.managers.curriculum.terms.reward import BaseLpacReward
from envs.tasks.managers.curriculum.types import ContinuousDimensionConfig, CurriculumSpaceConfig
from envs.tasks.locomotion import LocomotionTaskLogic
from envs.tasks.managers.command.terms.curriculum import Curriculum
from envs.tasks.managers.command.terms import get_command_class
from curriculum_helpers import build_curriculum_manager


def assert_state_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0, equal_nan=True)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for name in first:
            assert_state_equal(first[name], second[name])
    else:
        assert first == second


def config():
    return {"terms": {"speed": {"type": "Curriculum", "params": {
        "curriculum": "motion",
        "min_value": -1., "max_value": 1., "num_cell": 2,
    }}}}


def make(context, model_context, command_config=None):
    return build_curriculum_manager(
        context=context, model_context=model_context, num_envs=3,
        terms={"motion": {"type": "fast_lpac_reward", "params": {
            "reward_sources": ["reward"], "min_samples_per_cell": 1,
        }}},
        manager_configs={"command_manager_config": command_config or config(),
                         "reward_manager_config": {"terms": {}}},
    )


def update(manager):
    manager.update(raw_term_rewards={}, term_rewards={}, total_reward=torch.tensor([1., 2., 3.]))


def test_learning_modules_do_not_import_command_implementations():
    directory = Path(__file__).resolve().parents[1] / "envs/tasks/managers/curriculum"
    violations = []
    for path in directory.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.ImportFrom):
                if (node.module or "").startswith("envs.tasks.managers.command"):
                    violations.append(f"{path.name}:{node.lineno}")
    assert not violations


@pytest.mark.parametrize("legacy_config", [{}, {"temperature": 1.0}])
def test_curriculum_requires_explicit_type(legacy_config, runtime_context, model_context):
    with pytest.raises(ValueError, match="'type' is missing"):
        CurriculumManager(
            3, runtime_context, model_context,
            manager_configs={}, terms={"fast_lpac_reward": legacy_config},
        )


def test_curriculum_rejects_null_term_config(runtime_context, model_context):
    with pytest.raises(TypeError, match="must be a dict"):
        CurriculumManager(
            3, runtime_context, model_context,
            manager_configs={}, terms={"fast_lpac_reward": None},  # type: ignore[dict-item]
        )


def test_course_builds_spaces_without_command_config(runtime_context, model_context):
    space = CurriculumSpaceConfig({"value": ContinuousDimensionConfig(0., 2., 2)})
    manager = CurriculumManager(
        3, runtime_context, model_context,
        terms={"course": {"type": "lpac_reward", "params": {
            "reward_sources": ["reward"],
        }}},
        manager_configs={"reward_manager_config": {"terms": {}}},
        space_configs=[{"course": space}],
    )
    manager.resample()
    term = cast(BaseLpacReward, manager.get_term("course"))
    before = term.buffer.cell_starts.clone()
    space.dimensions["value"] = ContinuousDimensionConfig(99., 100., 1)
    torch.testing.assert_close(term.buffer.cell_starts, before)


def test_command_only_needs_read_protocol(runtime_context, model_context):
    class ReadOnlySampler:
        def get_bounds(self, dimension, env_ids=None):
            count = 3 if env_ids is None else env_ids.numel()
            return torch.zeros(count, 1), torch.ones(count, 1)

    sampler = ReadOnlySampler()
    class Provider:
        def get_term(self, name):
            return sampler
    provider = Provider()
    command = CommandManager(3, runtime_context, model_context,
                             curriculum_manager=provider, **config())
    command.reset(torch.tensor([2, 0]))
    assert torch.all((command.command["speed"] >= 0) & (command.command["speed"] <= 1))


def test_command_manager_forwards_provider_without_resolving_it(runtime_context, model_context, monkeypatch):
    from envs.tasks.managers.command.terms.uniform import Uniform
    received = []
    class Provider:
        def get_term(self, name):
            raise AssertionError('The manager must not resolve the provider')
    class Consumer(Uniform):
        def __init__(self, *args, curriculum_manager=None, **kwargs):
            received.append(curriculum_manager)
            super().__init__(*args, **kwargs)
    monkeypatch.setattr('envs.tasks.managers.command.base.get_command_class', lambda name: Consumer)
    provider = Provider()
    CommandManager(3, runtime_context, model_context, curriculum_manager=provider,
                   terms={'value': {'type': 'Uniform', 'params': {'min_value': 0., 'max_value': 1.}}})
    assert received == [provider]


def test_same_command_class_keeps_distinct_instance_names(runtime_context, model_context):
    manager = CommandManager(3, runtime_context, model_context, terms={
        name: {'type': 'Uniform', 'params': {'min_value': 0., 'max_value': 1.}}
        for name in ('first', 'second')
    })
    assert type(manager.terms['first']) is type(manager.terms['second'])
    assert manager.terms['first'].name == 'first'
    assert manager.terms['second'].name == 'second'


def test_direct_term_construction_requires_an_explicit_name(runtime_context, model_context):
    from envs.tasks.managers.command.terms.uniform import Uniform
    with pytest.raises(TypeError, match='name'):
        Uniform(num_envs=3, context=runtime_context, model_context=model_context,
                min_value=0., max_value=1.)


@pytest.mark.parametrize('name', [None, '', 3])
def test_term_name_cannot_be_empty_or_invalid(runtime_context, model_context, name):
    from envs.tasks.managers.command.terms.uniform import Uniform
    with pytest.raises(ValueError, match='non-empty string'):
        Uniform(name=name, num_envs=3, context=runtime_context, model_context=model_context,
                min_value=0., max_value=1.)


@pytest.mark.parametrize("params", [{}, {"curriculum": None}, {"curriculum": ""}, {"curriculum": 1}])
def test_course_reference_never_falls_back_to_class_attributes(params):
    class CommandWithOldAttribute(Curriculum):
        curriculum_term_name = "motion"

    with pytest.raises(ValueError, match="non-empty 'curriculum'"):
        CommandWithOldAttribute.get_curriculum_references(params)
    assert CommandWithOldAttribute.get_curriculum_references({"curriculum": "explicit"}) == {"explicit"}


@pytest.mark.parametrize("type_name", [
    "LpacSampleOnReset", "SymmetricLpacSampleOnReset",
    "FastLpacSampleOnReset", "SymmetricFastLpacSampleOnReset",
])
def test_algorithm_specific_command_aliases_are_not_supported(type_name):
    with pytest.raises(KeyError):
        get_command_class(type_name)


def test_settlement_does_not_sample_and_reads_do_not_advance_rng(runtime_context, model_context):
    manager = make(runtime_context, model_context)
    manager.resample()
    term = cast(BaseLpacReward, manager.get_term("motion"))
    update(manager)
    before = torch.random.get_rng_state()
    with pytest.raises(RuntimeError, match="Settle active"):
        manager.resample()
    assert torch.equal(before, torch.random.get_rng_state())
    manager.finalize_episodes(torch.tensor([2, 0]))
    assert torch.equal(before, torch.random.get_rng_state())
    assert term.buffer.assigned_cell_ids[[2, 0]].tolist() == [-1, -1]
    with pytest.raises(RuntimeError, match="has not been sampled"):
        term.get_bounds("speed", torch.tensor([0]))
    manager.resample(torch.tensor([2, 0]))
    before = torch.random.get_rng_state()
    first = term.get_bounds("speed")
    second = term.get_bounds("speed")
    assert torch.equal(before, torch.random.get_rng_state())
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("course_type", ["lpac_reward", "fast_lpac_reward"])
def test_evaluation_preserves_statistics_and_probabilities(runtime_context, model_context, course_type):
    manager = make(runtime_context, model_context)
    # The behavior is shared by both algorithms; construct the requested one.
    if course_type == "lpac_reward":
        manager = CurriculumManager(3, runtime_context, model_context,
            terms={"motion": {"type": course_type, "params": {
                "reward_sources": ["reward"], "min_samples_per_cell": 1}}},
            manager_configs={"reward_manager_config": {"terms": {}}},
            space_configs=[CommandManager.get_curriculum_space_config(config())])
    manager.resample()
    update(manager)
    manager.finalize_episodes()
    manager.resample()
    before = deepcopy(manager.checkpoint_state_dict())
    manager.set_training_mode(False)
    for _ in range(3):
        update(manager)
        manager.finalize_episodes()
        manager.resample()
    assert_state_equal(before, manager.checkpoint_state_dict())
    manager.set_training_mode(True)
    update(manager)
    manager.finalize_episodes()
    assert manager.collect_metrics()["curriculum/motion/statistics_revision"] > 0


def test_same_algorithm_instances_have_independent_spaces(runtime_context, model_context):
    commands = config()
    commands["terms"]["turn"] = deepcopy(commands["terms"]["speed"])
    commands["terms"]["turn"]["params"]["curriculum"] = "turning"
    spaces = build_curriculum_spaces([CommandManager.get_curriculum_space_config(commands)], {"motion", "turning"}, runtime_context)
    assert set(spaces["motion"].term_slices) == {"speed"}
    assert set(spaces["turning"].term_slices) == {"turn"}
    manager = build_curriculum_manager(
        num_envs=3, context=runtime_context, model_context=model_context,
        terms={name: {"type": "fast_lpac_reward", "params": {
            "reward_sources": ["reward"]}} for name in ("motion", "turning")},
        manager_configs={"command_manager_config": commands,
                         "reward_manager_config": {"terms": {}}},
    )
    manager.resample()
    command = CommandManager(3, runtime_context, model_context,
                             curriculum_manager=manager, **commands)
    command.reset()
    for reference, dimension in (("motion", "speed"), ("turning", "turn")):
        term = cast(BaseLpacReward, manager.get_term(reference))
        lower, upper = term.get_bounds(dimension)
        assert torch.all((command.command[dimension] >= lower) & (command.command[dimension] <= upper))
    commands["constraints"] = {"turn": {"operator": "<=", "expression": "{speed}"}}
    with pytest.raises(ValueError, match="crosses curriculum instances"):
        build_curriculum_spaces([CommandManager.get_curriculum_space_config(commands)], {"motion", "turning"}, runtime_context)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_candidate_builder_rejects_invalid_cell_counts(runtime_context, value):
    commands = config()
    commands["terms"]["speed"]["params"]["num_cell"] = value
    with pytest.raises(ValueError, match="must be positive"):
        build_curriculum_spaces([CommandManager.get_curriculum_space_config(commands)], {"motion"}, runtime_context)


def test_unrelated_commands_stay_independent(runtime_context, model_context):
    commands = config()
    commands["terms"]["constant"] = {"type": "Uniform", "params": {
        "min_value": 3., "max_value": 3.,
    }}
    manager = make(runtime_context, model_context, commands)
    manager.resample()
    command = CommandManager(3, runtime_context, model_context,
                             curriculum_manager=manager, **commands)
    command.reset()
    torch.testing.assert_close(command.command["constant"], torch.full((3, 1), 3.))


def test_empty_selection_does_not_resample_or_consume_rng(runtime_context, model_context):
    manager = make(runtime_context, model_context)
    manager.resample()
    term = cast(BaseLpacReward, manager.get_term("motion"))
    assignments = term.buffer.assigned_cell_ids.clone()
    before = torch.random.get_rng_state()
    manager.resample(torch.empty(0, dtype=torch.long))
    assert torch.equal(before, torch.random.get_rng_state())
    torch.testing.assert_close(assignments, term.buffer.assigned_cell_ids)


def test_task_passes_course_configuration_without_interpreting_it(runtime_context, model_context, monkeypatch):
    import envs.tasks.locomotion as locomotion_module

    captured = {}

    class ManagerSpy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(locomotion_module, "CurriculumManager", ManagerSpy)
    task = LocomotionTaskLogic(runtime_context)
    task.num_envs = 3
    task.model_context = model_context
    course_config = {"terms": {"custom": {"type": "owned_by_manager", "params": {}}}}
    task._build_curriculum_manager(course_config, {"command_manager_config": {"terms": {}}})
    assert captured["terms"] is course_config["terms"]
    assert "space_configs" in captured
    assert "sampling_spaces" not in captured


def test_task_wires_command_without_mutating_configuration(runtime_context, model_context):
    task = LocomotionTaskLogic(runtime_context)
    task.num_envs = 3
    task.model_context = model_context
    task.curriculum_manager = make(runtime_context, model_context)
    commands = config()
    before = deepcopy(commands)
    task._build_command_manager(commands)
    assert commands == before
    term = task.command_manager.terms["speed"]
    assert isinstance(term, Curriculum)
    assert term.curriculum_sampler is task.curriculum_manager.get_term("motion")

@pytest.mark.parametrize('base_name', ['command', 'randomization'])
def test_consumer_term_defaults_share_state_and_lifecycle_contract(
    runtime_context, model_context, base_name,
):
    from envs.tasks.managers.command.terms.base import BaseCommandTerm
    from envs.tasks.managers.randomization.terms.base import BaseRandomizationTerm

    base = cast(Any, BaseCommandTerm if base_name == 'command' else BaseRandomizationTerm)

    class Consumer(base):
        def matches_config(self, params):
            return not params

        def update(self, task_context):
            pass

        def reset(self, env_ids=None):
            pass

    kwargs = {'model_context': model_context} if base_name == 'command' else {}
    term = Consumer(name='consumer', num_envs=3, context=runtime_context, **kwargs)
    assert term.name == 'consumer'
    assert term.get_curriculum_references({}) == set()
    assert term.get_curriculum_space_config('consumer', {}) == {}
    assert term.checkpoint_state_dict() == {}
    assert term.playback_state_dict(1) == {}
    term.config_update(curriculum_manager=object())
    term.set_training_mode(False)
    assert not hasattr(term, 'curriculum_manager')
    assert not hasattr(term, 'config')

    Consumer.CHECKPOINT_STATE_FIELDS = ('values',)
    term.values = torch.tensor([1., 2., 3.])
    saved = term.checkpoint_state_dict()
    term.values.zero_()
    term.load_checkpoint_state_dict(saved)
    torch.testing.assert_close(term.values, torch.tensor([1., 2., 3.]))
    with pytest.raises(NotImplementedError, match='playback_state_dict'):
        term.playback_state_dict(1)


def test_command_manager_aggregates_term_owned_dimensions(monkeypatch):
    from envs.tasks.managers.curriculum.types import DiscreteDimensionConfig

    class CustomTerm:
        @classmethod
        def get_curriculum_space_config(cls, name, params):
            # No min_value/max_value: the manager cannot assume continuous commands.
            return {'course': CurriculumSpaceConfig(dimensions={
                name: DiscreteDimensionConfig(tuple(params['choices'])),
            }, persistent_assignments=True)}

    monkeypatch.setattr('envs.tasks.managers.command.base.get_command_class', lambda name: CustomTerm)
    descriptions = CommandManager.get_curriculum_space_config({'terms': {
        'first': {'type': 'custom', 'params': {'choices': [1., 2.]}},
        'second': {'type': 'custom', 'params': {'choices': [3., 4.]}},
    }})
    space = descriptions['course']
    assert set(space.dimensions) == {'first', 'second'}
    assert space.dimensions['second'] == DiscreteDimensionConfig((3., 4.))
    assert space.persistent_assignments
