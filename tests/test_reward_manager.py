from dataclasses import replace
from typing import cast

import torch

from envs.tasks.managers.reward.base import RewardManager
from envs.tasks.managers.reward.terms.foot import (
    FootStateDurationCommandWeighedExp,
)
from envs.tasks.managers.reward.terms.gait import QuadrupedalGaitPhaseL2Exp
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


def make_phase_gait_context() -> TaskContext:
    qpos = torch.zeros(2, 9)
    qpos[:, 1] = 1.0
    qpos[:, 7] = 0.5
    geom_xpos = torch.zeros(2, 7, 3)
    geom_xpos[:, [3, 6], 2] = 0.7
    geom_xpos[:, [4, 5], 2] = 0.55

    return TaskContext(
        state={
            "qpos": qpos,
            "geom_xpos": geom_xpos,
        },
        command={
            "foot_phase_real": torch.ones(2, 4),
            "foot_phase_imag": torch.zeros(2, 4),
            "half_period_duration": torch.full((2,), 0.04),
        },
        action=torch.zeros(2, 2),
        last_action=torch.zeros(2, 2),
        episode_step=torch.ones(2, dtype=torch.long),
        step_dt=0.02,
    )


def make_quadrupedal_foot_context() -> TaskContext:
    contact_forces = torch.zeros(2, 4, 6)
    contact_forces[0, 0, 0] = 20.0
    contact_forces[0, 1, 0] = 20.0
    contact_forces[1, 0, 0] = 20.0

    return TaskContext(
        state={
            "contact_geom_ids": torch.tensor([
                [[3, 0], [4, 0], [-1, -1], [-1, -1]],
                [[5, 0], [-1, -1], [-1, -1], [-1, -1]],
            ]),
            "contact_forces": contact_forces,
        },
        command={
            "lin_vel_x": torch.zeros(2, 1),
            "lin_vel_y": torch.zeros(2, 1),
            "ang_vel_z": torch.zeros(2, 1),
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
            "joint_power_l1": {},
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


def test_foot_state_duration_terms_use_new_names_and_update_each_foot(
    runtime_context,
    model_context,
):
    quadrupedal_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        geom_foot_ids=torch.tensor([3, 4, 5, 6]),
    )
    manager = RewardManager(
        num_envs=2,
        context=runtime_context,
        model_context=quadrupedal_model_context,
        terms={
            "foot_state_duration_command_weighed_exp": {},
            "foot_state_duration_cubic_command_weighed_exp": {},
        },
    )
    task_context = make_quadrupedal_foot_context()

    reward, info = manager.compute(task_context)
    term = cast(
        FootStateDurationCommandWeighedExp,
        manager.terms["foot_state_duration_command_weighed_exp"],
    )

    assert reward.shape == (2,)
    assert info["reward/foot_state_duration_command_weighed_exp"].shape == (2,)
    assert info["reward/foot_state_duration_cubic_command_weighed_exp"].shape == (2,)
    torch.testing.assert_close(
        term.duration,
        torch.tensor([
            [0.00, 0.00, 0.02, 0.02],
            [0.02, 0.02, 0.00, 0.02],
        ]),
    )


def test_foot_state_duration_ignores_low_force_contacts(
    runtime_context,
    model_context,
):
    quadrupedal_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        geom_foot_ids=torch.tensor([3, 4, 5, 6]),
    )
    manager = RewardManager(
        num_envs=2,
        context=runtime_context,
        model_context=quadrupedal_model_context,
        terms={
            "foot_state_duration_command_weighed_exp": {},
        },
    )
    task_context = make_quadrupedal_foot_context()
    task_context.state["contact_forces"].zero_()

    manager.compute(task_context)
    term = cast(
        FootStateDurationCommandWeighedExp,
        manager.terms["foot_state_duration_command_weighed_exp"],
    )

    torch.testing.assert_close(
        term.duration,
        torch.full((2, 4), 0.02),
    )


def test_quadrupedal_foot_velocity_diff_matches_diagonal_feet(
    runtime_context,
    model_context,
):
    quadrupedal_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        geom_foot_ids=torch.tensor([3, 4, 5, 6]),
    )
    manager = RewardManager(
        num_envs=2,
        context=runtime_context,
        model_context=quadrupedal_model_context,
        terms={"quadrupedal_foot_velocity_diff_l2": {}},
    )
    task_context = make_quadrupedal_foot_context()
    task_context.state["geom_xvel"] = torch.zeros(2, 7, 6)
    task_context.state["geom_xvel"][0, 3, 3:5] = torch.tensor([1.0, 2.0])
    task_context.state["geom_xvel"][0, 6, 3:5] = torch.tensor([1.0, 4.0])
    task_context.state["geom_xvel"][0, 4, 3:5] = torch.tensor([3.0, 0.0])
    task_context.state["geom_xvel"][0, 5, 3:5] = torch.tensor([1.0, 0.0])
    task_context.state["geom_xvel"][1, 3, 3:5] = torch.tensor([0.5, 0.5])
    task_context.state["geom_xvel"][1, 6, 3:5] = torch.tensor([0.5, 0.5])
    task_context.state["geom_xvel"][1, 4, 3:5] = torch.tensor([2.0, -1.0])
    task_context.state["geom_xvel"][1, 5, 3:5] = torch.tensor([2.0, -1.0])

    reward, info = manager.compute(task_context)

    expected = torch.tensor([8.0, 0.0])
    torch.testing.assert_close(reward, expected)
    torch.testing.assert_close(
        info["reward/quadrupedal_foot_velocity_diff_l2"],
        expected,
    )


def test_quadrupedal_phase_gait_updates_phase_steps(
    runtime_context,
    model_context,
):
    phase_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        geom_foot_ids=torch.tensor([3, 4, 5, 6]),
    )
    manager = RewardManager(
        num_envs=2,
        context=runtime_context,
        model_context=phase_model_context,
        terms={
            "quadrupedal_gait_phase_l2_exp": {
                "params": {"target_height": 0.2},
            },
        },
    )
    term = cast(
        QuadrupedalGaitPhaseL2Exp,
        manager.terms["quadrupedal_gait_phase_l2_exp"],
    )
    task_context = make_phase_gait_context()

    reward, _ = manager.compute(task_context)
    assert reward.shape == (2,)
    torch.testing.assert_close(
        term.foot_phase_steps,
        torch.tensor([
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ]),
    )

    manager.compute(task_context)
    torch.testing.assert_close(
        term.foot_phase_steps,
        torch.tensor([
            [0.0, 2.0, 2.0, 0.0],
            [0.0, 2.0, 2.0, 0.0],
        ]),
    )

    manager.compute(task_context)
    torch.testing.assert_close(
        term.foot_phase_steps,
        torch.tensor([
            [1.0, 3.0, 3.0, 1.0],
            [1.0, 3.0, 3.0, 1.0],
        ]),
    )

    manager.reset(torch.tensor([1]))
    assert term.foot_phase_steps[0].any()
    assert not term.foot_phase_steps[1].any()


def test_quadrupedal_phase_gait_uses_base_plane_distance(
    runtime_context,
    model_context,
):
    phase_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        geom_foot_ids=torch.tensor([3, 4, 5, 6]),
    )
    term = QuadrupedalGaitPhaseL2Exp(
        num_envs=1,
        context=runtime_context,
        model_context=phase_model_context,
        target_height=0.2,
    )
    task_context = make_phase_gait_context()
    task_context.state["qpos"] = torch.zeros(1, 9)
    task_context.state["qpos"][:, 1] = 2.0 ** -0.5
    task_context.state["qpos"][:, 3] = 2.0 ** -0.5
    task_context.state["geom_xpos"] = torch.zeros(1, 7, 3)
    task_context.state["geom_xpos"][:, [3, 4, 5, 6], 0] = 0.2
    task_context.command["foot_phase_real"] = torch.ones(1, 4)
    task_context.command["foot_phase_imag"] = torch.zeros(1, 4)
    task_context.command["half_period_duration"] = torch.full((1,), 0.04)

    torch.testing.assert_close(
        term._foot_height_from_base_plane(task_context),
        torch.full((1, 4), 0.2),
    )
