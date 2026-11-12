"""Manager descriptions combine before any shared curriculum cells are built."""
from copy import deepcopy

import pytest
import torch

from envs.tasks.base import BaseTaskLogic
from envs.tasks.locomotion import LocomotionTaskLogic
from envs.tasks.managers import CommandManager
from envs.tasks.managers.curriculum.spaces import build_curriculum_spaces
from envs.tasks.managers.curriculum.types import (
    ContinuousDimensionConfig, DiscreteDimensionConfig, CurriculumSpaceConfig,
)


def contribution(dimensions, *, source=None, filters=()):
    return {"course": CurriculumSpaceConfig(
        dimensions=dimensions, source_config=source or {}, filters=filters,
    )}


def build(context, *contributions):
    return build_curriculum_spaces(contributions, {"course"}, context)


@pytest.mark.parametrize("manager_type", LocomotionTaskLogic.CURRICULUM_SPACE_PROVIDERS.values())
def test_space_providers_expose_config_interface(manager_type):
    assert manager_type.get_curriculum_space_config({"terms": {}}) == {}
    assert manager_type.get_curriculum_space_config(None) == {}


def test_command_returns_description_without_building_tables():
    config = {"terms": {"speed": {"type": "Curriculum", "params": {
        "curriculum": "course",
        "min_value": -1., "max_value": 1., "num_cell": 2,
    }}}}
    before = deepcopy(config)
    state = torch.random.get_rng_state()
    space = CommandManager.get_curriculum_space_config(config)["course"]
    assert space.dimensions["speed"] == ContinuousDimensionConfig(-1., 1., 2)
    assert not hasattr(space, "cell_starts")
    assert config == before
    assert torch.equal(state, torch.random.get_rng_state())


def test_manager_contributions_form_one_joint_space(runtime_context):
    command = contribution({"speed": ContinuousDimensionConfig(-1., 1., 2)},
                           source={"command": {"range": [-1., 1.]}})
    other = contribution({"condition": DiscreteDimensionConfig((0., 1., 2.))},
                         source={"condition": {"values": [0, 1, 2]}})
    result = build(runtime_context, command, other)["course"]
    space = result
    assert space.dimension_names == ("speed", "condition")
    assert space.term_slices == {"speed": slice(0, 1), "condition": slice(1, 2)}
    torch.testing.assert_close(space.cell_starts, torch.tensor([
        [-1., 0.], [-1., 1.], [-1., 2.], [0., 0.], [0., 1.], [0., 2.],
    ]))
    torch.testing.assert_close(space.cell_interval, torch.tensor([1., 0.]))
    assert set(result.source_config) == {"command", "condition"}


def test_command_filter_preserves_other_manager_columns(runtime_context):
    config = {"terms": {"speed": {"type": "Curriculum", "params": {
        "curriculum": "course",
        "min_value": -1., "max_value": 1., "num_cell": 2,
    }}}, "constraints": {"speed": {"operator": ">=", "expression": "0.0"}}}
    other = contribution({"condition": DiscreteDimensionConfig((0., 1., 2.))})
    space = build(runtime_context, CommandManager.get_curriculum_space_config(config), other)["course"]
    torch.testing.assert_close(space.cell_starts, torch.tensor([[0., 0.], [0., 1.], [0., 2.]]))


def test_task_collects_all_providers_without_manager_instances(runtime_context):
    class Provider:
        @classmethod
        def get_curriculum_space_config(cls, config):
            return contribution({"choice": DiscreteDimensionConfig(tuple(config["choices"]))})

    class Task(LocomotionTaskLogic):
        CURRICULUM_SPACE_PROVIDERS = {"custom_config": Provider}

    descriptions = Task.get_curriculum_space_configs({"custom_config": {"choices": [0., 1.]}})
    assert build(runtime_context, *descriptions)["course"].cell_starts.shape == (2, 1)


def test_base_task_does_not_own_curriculum(runtime_context):
    task = BaseTaskLogic(runtime_context)
    assert not hasattr(task, "curriculum_manager")
    assert not hasattr(task, "CURRICULUM_SPACE_PROVIDERS")
    assert not hasattr(task, "get_curriculum_space_configs")


def test_duplicate_dimensions_are_rejected(runtime_context):
    item = contribution({"speed": ContinuousDimensionConfig(0., 1., 2)})
    with pytest.raises(ValueError, match="Duplicate dimensions"):
        build(runtime_context, item, item)


def test_source_conflicts_are_rejected_between_providers(runtime_context):
    first = contribution({"x": ContinuousDimensionConfig(0., 1., 2)}, source={"owner": 1})
    second = contribution({"y": ContinuousDimensionConfig(0., 1., 2)}, source={"owner": 2})
    with pytest.raises(ValueError, match="Conflicting.*source"):
        build(runtime_context, first, second)


@pytest.mark.parametrize("discrete", [False, True])
def test_joint_cell_limit_is_checked_before_cartesian_product(runtime_context, monkeypatch, discrete):
    from envs.tasks.managers.curriculum.spaces import _MAX_CELLS
    first = contribution({"x": ContinuousDimensionConfig(0., 1., 1000)})
    count = _MAX_CELLS // 1000 + 1
    dimension = (
        DiscreteDimensionConfig(tuple(float(value) for value in range(count)))
        if discrete else ContinuousDimensionConfig(0., 1., count)
    )
    second = contribution({"y": dimension})
    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("Oversized joint cells must be rejected before allocation.")
    monkeypatch.setattr(torch, "cartesian_prod", unexpected_allocation)
    with pytest.raises(ValueError, match=f"internal cell limit of {_MAX_CELLS}"):
        build(runtime_context, first, second)


def test_joint_space_can_exceed_former_default_limit(runtime_context):
    item = contribution({"x": ContinuousDimensionConfig(0., 1., 1500)})
    assert build(runtime_context, item)["course"].cell_starts.shape == (1500, 1)


def test_unknown_course_is_rejected_at_assembly(runtime_context):
    item = contribution({"x": ContinuousDimensionConfig(0., 1., 2)})
    with pytest.raises(ValueError, match="unknown curriculum"):
        build_curriculum_spaces([item], set(), runtime_context)


@pytest.mark.parametrize("values", [(), (1., 1.), (float("nan"),), (float("inf"),)])
def test_invalid_discrete_candidates_are_rejected(runtime_context, values):
    with pytest.raises(ValueError, match="unique finite"):
        build(runtime_context, contribution({"x": DiscreteDimensionConfig(values)}))


def test_filters_cannot_drop_foreign_dimensions(runtime_context):
    def invalid_filter(starts, intervals):
        return {"x": starts["x"]}

    first = contribution({"x": ContinuousDimensionConfig(0., 1., 2)}, filters=(invalid_filter,))
    second = contribution({"y": DiscreteDimensionConfig((0., 1.))})
    with pytest.raises(ValueError, match="preserve every dimension"):
        build(runtime_context, first, second)


def test_filters_from_different_managers_compose(runtime_context):
    def positive(starts, intervals):
        mask = starts["x"][:, 0] >= 0
        return {name: value[mask] for name, value in starts.items()}

    def exclude_middle(starts, intervals):
        mask = starts["choice"][:, 0] != 1
        return {name: value[mask] for name, value in starts.items()}

    first = contribution({"x": ContinuousDimensionConfig(-1., 1., 2)}, filters=(positive,))
    second = contribution({"choice": DiscreteDimensionConfig((0., 1., 2.))}, filters=(exclude_middle,))
    result = build(runtime_context, first, second)["course"]
    torch.testing.assert_close(result.cell_starts, torch.tensor([[0., 0.], [0., 2.]]))


def test_filters_cannot_desynchronize_manager_columns(runtime_context):
    def invalid_filter(starts, intervals):
        return {"x": starts["x"][:1], "y": starts["y"]}

    first = contribution({"x": ContinuousDimensionConfig(0., 1., 2)}, filters=(invalid_filter,))
    second = contribution({"y": DiscreteDimensionConfig((0., 1.))})
    with pytest.raises(ValueError, match="aligned"):
        build(runtime_context, first, second)
