import torch

from envs.tasks.managers.reward.base import RewardManager
from envs.tasks.utils.context import TaskContext


def make_reward_context() -> TaskContext:
    return TaskContext(
        state={},
        command={},
        action=torch.tensor([[1.0, 3.0], [2.0, 2.0]]),
        last_action=torch.tensor([[0.0, 1.0], [1.0, 1.0]]),
        episode_step=torch.zeros(2, dtype=torch.long),
    )


def test_reward_manager_computes_weighted_reward_and_reuses_buffer(
    runtime_context,
    model_context,
):
    manager = RewardManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        terms={"action_diff_l2": {"weight": 2.0}},
    )
    buffer = manager.get_term_reward("action_diff_l2")

    reward, means = manager.compute(make_reward_context())

    expected = torch.tensor([5.0, 2.0])
    torch.testing.assert_close(reward, expected)
    torch.testing.assert_close(buffer, expected)
    assert manager.get_term_reward("action_diff_l2") is buffer
    torch.testing.assert_close(means["action_diff_l2"], expected.mean())


def test_reward_manager_partial_reset(
    runtime_context,
    model_context,
):
    manager = RewardManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        terms={"action_diff_l2": {}},
    )
    manager.compute(make_reward_context())
    previous = manager.get_term_reward("action_diff_l2")[0].clone()

    manager.reset(torch.tensor([1]))

    torch.testing.assert_close(
        manager.get_term_reward("action_diff_l2")[0],
        previous,
    )
    assert manager.get_term_reward("action_diff_l2")[1] == 0.0

