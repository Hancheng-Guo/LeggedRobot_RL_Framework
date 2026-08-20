import torch

from envs.tasks.managers.action.base import ActionManager


def test_action_manager_processes_terms_in_pipeline_order(
    runtime_context,
    model_context,
):
    manager = ActionManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        terms={
            "hard_clamp": {"min_value": -1.0, "max_value": 1.0},
            "linear_map": {},
        },
    )

    control, info = manager.process(
        torch.tensor([[2.0, -2.0], [0.0, 0.5]])
    )

    expected = torch.tensor([[2.0, -2.0], [1.0, 1.0]])
    torch.testing.assert_close(control, expected)
    torch.testing.assert_close(info["action/control"], expected)


def test_action_manager_tracks_history_and_partial_reset(
    runtime_context,
    model_context,
):
    manager = ActionManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        terms={"hard_clamp": {"min_value": -1.0, "max_value": 1.0}},
    )

    first = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    second = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    manager.process(first)
    manager.process(second)

    torch.testing.assert_close(manager.action, second)
    torch.testing.assert_close(manager.last_action, first)

    manager.reset(torch.tensor([1]))
    torch.testing.assert_close(manager.action[0], second[0])
    torch.testing.assert_close(manager.last_action[0], first[0])
    assert torch.count_nonzero(manager.action[1]) == 0
    assert torch.count_nonzero(manager.last_action[1]) == 0

