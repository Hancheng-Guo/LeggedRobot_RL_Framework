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
