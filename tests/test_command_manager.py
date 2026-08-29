import pytest
import torch

# Importing the task module establishes the existing TaskContext import order.
from envs.tasks import base as task_base
from envs.tasks.managers.command.base import CommandManager


def make_manager(runtime_context, model_context):
    return CommandManager(
        num_envs=3,
        context=runtime_context,
        model_context=model_context,
        terms={
            "x": {
                "type": "UniformOnReset",
                "params": {"min_value": 2.0, "max_value": 2.0},
            },
            "y": {
                "type": "UniformOnReset",
                "params": {"min_value": 10.0, "max_value": 10.0},
            },
        },
        constraints={
            "x": {"operator": "<=", "expression": "1.0"},
            "y": {"operator": "<=", "expression": "{x} * 2.0"},
        },
    )


def test_command_constraints_are_applied_in_order(
    runtime_context,
    model_context,
):
    manager = make_manager(runtime_context, model_context)
    manager.reset()

    torch.testing.assert_close(manager.command["x"], torch.ones(3, 1))
    torch.testing.assert_close(
        manager.command["y"],
        torch.full((3, 1), 2.0),
    )


def test_command_manager_partial_reset_preserves_other_envs(
    runtime_context,
    model_context,
):
    manager = make_manager(runtime_context, model_context)
    manager.reset()
    manager.command["x"][0] = -5.0
    manager.command["y"][0] = -6.0

    manager.reset(torch.tensor([1, 2]))

    assert manager.command["x"][0] == -5.0
    assert manager.command["y"][0] == -6.0
    torch.testing.assert_close(
        manager.command["x"][1:],
        torch.ones(2, 1),
    )
    torch.testing.assert_close(
        manager.command["y"][1:],
        torch.full((2, 1), 2.0),
    )


def test_multidimensional_command_uses_shared_range_and_constraint(
    runtime_context,
    model_context,
):
    manager = CommandManager(
        num_envs=128,
        context=runtime_context,
        model_context=model_context,
        terms={
            "foot_phase": {
                "type": "UniformOnReset",
                "params": {
                    "dim": 4,
                    "min_value": -1.0,
                    "max_value": 1.0,
                },
            },
        },
        constraints={
            "foot_phase": {
                "operator": "<=",
                "expression": "0.5",
            },
        },
    )
    torch.manual_seed(0)

    manager.reset()

    command = manager.command["foot_phase"]
    assert command.shape == (128, 4)
    assert manager.output_dim == 4
    assert torch.all(command >= -1.0)
    assert torch.all(command <= 0.5)
    assert not torch.equal(command[:, 0], command[:, 1])


def test_command_dim_defaults_to_one(runtime_context, model_context):
    manager = make_manager(runtime_context, model_context)

    assert manager.output_dim == 2
    assert all(value.shape == (3, 1) for value in manager.command.values())


@pytest.mark.parametrize(
    ("operator", "initial_value", "direction", "expected_direction"),
    [
        ("<=", 1.0, None, None),
        (">=", -1.0, None, None),
        ("<", 1.0, None, "negative"),
        (">", -1.0, None, "positive"),
        ("!=", 0.0, None, "positive"),
        ("!=", 0.0, "negative", "negative"),
    ],
)
def test_command_constraint_operators(
    runtime_context,
    model_context,
    operator,
    initial_value,
    direction,
    expected_direction,
):
    constraint = {"operator": operator, "expression": "0.0"}
    if direction is not None:
        constraint["direction"] = direction
    manager = CommandManager(
        num_envs=3,
        context=runtime_context,
        model_context=model_context,
        terms={
            "x": {
                "type": "UniformOnReset",
                "params": {
                    "min_value": initial_value,
                    "max_value": initial_value,
                },
            },
        },
        constraints={"x": constraint},
    )

    manager.reset()

    boundary = torch.tensor(0.0, dtype=runtime_context.dtype)
    if expected_direction is None:
        expected_value = boundary
    else:
        target = torch.tensor(
            torch.inf if expected_direction == "positive" else -torch.inf,
            dtype=runtime_context.dtype,
        )
        expected_value = torch.nextafter(boundary, target)
    torch.testing.assert_close(
        manager.command["x"],
        expected_value.expand(3, 1),
        rtol=0.0,
        atol=0.0,
    )


def test_not_equal_constraint_rejects_invalid_direction(
    runtime_context,
    model_context,
):
    with pytest.raises(ValueError, match="positive.*negative"):
        CommandManager(
            num_envs=1,
            context=runtime_context,
            model_context=model_context,
            terms={
                "x": {
                    "type": "UniformOnReset",
                    "params": {"min_value": 0.0, "max_value": 1.0},
                },
            },
            constraints={
                "x": {
                    "operator": "!=",
                    "expression": "0.0",
                    "direction": "up",
                },
            },
        )
