import torch
from typing import cast

from envs.tasks.managers.reward.base import RewardManager
from envs.tasks.managers.reward.terms.tracking import (
    TrackLinearVelocityXyErrorIntegralL2,
)
from envs.tasks.utils.context import TaskContext


def make_reward_context() -> TaskContext:
    return TaskContext(
        state={},
        command={},
        action=torch.tensor([[1.0, 3.0], [2.0, 2.0]]),
        last_action=torch.tensor([[0.0, 1.0], [1.0, 1.0]]),
        episode_step=torch.zeros(2, dtype=torch.long),
        step_dt=0.02,
    )


def make_state_reward_context() -> TaskContext:
    return TaskContext(
        state={
            "qpos": torch.tensor([
                [1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.55, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 2.5],
            ]),
            "qvel": torch.tensor([
                [0.2, 0.0, 0.0, 0.0, 1.0, 2.0, 0.5, 0.3],
                [0.4, 0.0, 0.0, 0.0, 0.5, 0.0, -0.5, 0.6],
            ]),
            "base_lin_vel_body": torch.tensor([
                [1.0, 2.0, 0.5],
                [0.5, 0.0, -0.5],
            ]),
            "base_ang_vel_body": torch.tensor([
                [0.0, 0.0, 0.5],
                [0.0, 0.0, -0.5],
            ]),
            "actuator_force": torch.tensor([
                [2.0, 3.0],
                [4.0, 5.0],
            ]),
            "contact_geom_ids": torch.tensor([
                [[3, 0], [2, 0]],
                [[1, 0], [-1, -1]],
            ]),
            "contact_forces": torch.zeros(2, 2, 6),
            "geom_xpos": torch.zeros(2, 4, 3),
            "geom_xvel": torch.zeros(2, 4, 6),
        },
        command={
            "lin_vel_x": torch.tensor([[0.0], [0.0]]),
            "lin_vel_y": torch.tensor([[2.0], [0.0]]),
            "ang_vel_z": torch.tensor([[0.5], [-0.5]]),
        },
        action=torch.zeros(2, 2),
        last_action=torch.zeros(2, 2),
        episode_step=torch.ones(2, dtype=torch.long),
        step_dt=0.02,
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

    reward, info = manager.compute(make_reward_context())

    expected = torch.tensor([5.0, 2.0])
    torch.testing.assert_close(reward, expected)
    torch.testing.assert_close(buffer, expected)
    assert manager.get_term_reward("action_diff_l2") is buffer
    torch.testing.assert_close(
        info["reward/action_diff_l2"],
        expected,
    )
    torch.testing.assert_close(info["reward"], expected.mean())


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


def test_new_reward_terms_compute_batched_tensors(
    runtime_context,
    model_context,
):
    manager = RewardManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        terms={
            "base_height_l2": {
                "params": {"target_height": 0.45},
            },
            "joint_limit_violation_l1": {},
            "illegal_contact_l1": {},
            "track_linear_velocity_xy_l2_exp": {},
            "joint_power_abs": {},
        },
    )

    reward, info = manager.compute(make_state_reward_context())

    assert reward.shape == (2,)
    assert info["reward/base_height_l2"].shape == (2,)
    torch.testing.assert_close(
        info["reward/base_height_l2"],
        torch.tensor([0.01, 0.01]),
    )
    torch.testing.assert_close(
        info["reward/joint_limit_violation_l1"],
        torch.tensor([0.5, 0.5]),
    )
    torch.testing.assert_close(
        info["reward/illegal_contact_l1"],
        torch.tensor([1.0, 1.0]),
    )


def test_integral_reward_term_resets_internal_buffer(
    runtime_context,
    model_context,
):
    manager = RewardManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        terms={
            "track_linear_velocity_xy_error_integral_l2": {
                "params": {"integral_length": 3},
            },
        },
    )

    manager.compute(make_state_reward_context())
    term = cast(
        TrackLinearVelocityXyErrorIntegralL2,
        manager.terms["track_linear_velocity_xy_error_integral_l2"],
    )
    assert term.error_history.any()

    manager.reset(torch.tensor([1]))

    assert term.error_history[0].any()
    assert not term.error_history[1].any()
