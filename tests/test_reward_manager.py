from dataclasses import replace
from typing import cast

import pytest
import torch

from envs.tasks.managers.reward.base import RewardManager
from envs.tasks.managers.reward.terms.foot import (
    FootLiftHeightCommandWeightedExp,
    FootStateDurationCommandWeighedExp,
)
from envs.tasks.managers.reward.terms.gait import (
    QuadrupedalGaitPhaseL2Exp,
    TrotLoopDurationTanh,
)
from envs.tasks.managers.reward.terms.tracking import (
    TrackLinearVelocityXL2Exp,
    TrackLinearVelocityXL2ExpAndLogcosh,
    TrackLinearVelocityYL2Exp,
    TrackLinearVelocityYL2ExpAndLogcosh,
    TrackLinearVelocityXyErrorIntegralL2,
)
from envs.tasks.utils.context import TaskContext
from envs.simulators.utils.state import SimulatorState


def make_simulator_state(
    num_envs: int = 2,
    **overrides: torch.Tensor,
) -> SimulatorState:
    values = {
        "qpos": torch.empty(num_envs, 0),
        "qvel": torch.empty(num_envs, 0),
        "qacc": torch.empty(num_envs, 0),
        "ctrl": torch.empty(num_envs, 0),
        "geom_xpos": torch.empty(num_envs, 0, 3),
        "geom_xvel": torch.empty(num_envs, 0, 6),
        "actuator_force": torch.empty(num_envs, 0),
        "base_lin_vel_body": torch.empty(num_envs, 3),
        "base_ang_vel_body": torch.empty(num_envs, 3),
        "contact_geom_ids": torch.empty(num_envs, 0, 2, dtype=torch.long),
        "contact_forces": torch.empty(num_envs, 0, 6),
        "foot_ground_contact": torch.empty(
            num_envs, 0, 0, dtype=torch.bool
        ),
    }
    values.update(overrides)
    return SimulatorState(**values)


def make_reward_context() -> TaskContext:
    return TaskContext(
        state=make_simulator_state(),
        command={},
        action=torch.tensor([[1.0, 3.0], [2.0, 2.0]]),
        last_action=torch.tensor([[0.0, 1.0], [1.0, 1.0]]),
        episode_step=torch.zeros(2, dtype=torch.long),
        step_dt=0.02,
    )


def make_state_reward_context() -> TaskContext:
    return TaskContext(
        state=make_simulator_state(
            qpos=torch.tensor([
                [1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.55, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 2.5],
            ]),
            qvel=torch.tensor([
                [0.2, 0.0, 0.0, 0.0, 1.0, 2.0, 0.5, 0.3],
                [0.4, 0.0, 0.0, 0.0, 0.5, 0.0, -0.5, 0.6],
            ]),
            base_lin_vel_body=torch.tensor([
                [1.0, 2.0, 0.5],
                [0.5, 0.0, -0.5],
            ]),
            base_ang_vel_body=torch.tensor([
                [0.0, 0.0, 0.5],
                [0.0, 0.0, -0.5],
            ]),
            actuator_force=torch.tensor([
                [2.0, 3.0],
                [4.0, 5.0],
            ]),
            contact_geom_ids=torch.tensor([
                [[3, 0], [2, 0]],
                [[1, 0], [-1, -1]],
            ]),
            contact_forces=torch.zeros(2, 2, 6),
            geom_xpos=torch.zeros(2, 4, 3),
            geom_xvel=torch.zeros(2, 4, 6),
        ),
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
        state=make_simulator_state(
            qpos=qpos,
            geom_xpos=geom_xpos,
        ),
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
        state=make_simulator_state(
            contact_geom_ids=torch.tensor([
                [[3, 0], [4, 0], [-1, -1], [-1, -1]],
                [[5, 0], [-1, -1], [-1, -1], [-1, -1]],
            ]),
            foot_ground_contact=torch.tensor([
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ], dtype=torch.bool),
            contact_forces=contact_forces,
        ),
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
            "joint_position_diff_l2": {},
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
    assert info["reward/joint_position_diff_l2"].shape == (2,)


def test_axis_specific_linear_velocity_rewards(
    runtime_context,
    model_context,
):
    task_context = make_state_reward_context()
    terms = (
        TrackLinearVelocityXL2Exp(
            context=runtime_context,
            model_context=model_context,
        ),
        TrackLinearVelocityYL2Exp(
            context=runtime_context,
            model_context=model_context,
        ),
        TrackLinearVelocityXL2ExpAndLogcosh(
            context=runtime_context,
            model_context=model_context,
            logcosh_weight=0.25,
        ),
        TrackLinearVelocityYL2ExpAndLogcosh(
            context=runtime_context,
            model_context=model_context,
            logcosh_weight=0.25,
        ),
    )

    x_error = torch.tensor([1.0, 0.5])
    y_error = torch.zeros(2)
    expected_x_exp = torch.exp(-x_error.square())
    expected_y_exp = torch.exp(-y_error.square())
    expected_x_logcosh = (
        0.75 * expected_x_exp
        + 0.25 * (1.0 - torch.log(torch.cosh(2.0 * x_error)))
    )
    expected_y_logcosh = torch.ones(2)

    expected = (
        expected_x_exp,
        expected_y_exp,
        expected_x_logcosh,
        expected_y_logcosh,
    )
    for term, term_expected in zip(terms, expected):
        reward = term.compute(task_context)
        assert reward.shape == (2,)
        torch.testing.assert_close(reward, term_expected)


@pytest.mark.parametrize("std", [0.0, -1.0, float("nan"), float("inf")])
def test_axis_specific_linear_velocity_rewards_reject_invalid_std(
    runtime_context,
    model_context,
    std,
):
    for term_class in (
        TrackLinearVelocityXL2Exp,
        TrackLinearVelocityYL2Exp,
        TrackLinearVelocityXL2ExpAndLogcosh,
        TrackLinearVelocityYL2ExpAndLogcosh,
    ):
        with pytest.raises(ValueError, match="positive and finite"):
            term_class(
                context=runtime_context,
                model_context=model_context,
                std=std,
            )


@pytest.mark.parametrize(
    "logcosh_weight",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_axis_specific_linear_velocity_rewards_reject_invalid_logcosh_weight(
    runtime_context,
    model_context,
    logcosh_weight,
):
    for term_class in (
        TrackLinearVelocityXL2ExpAndLogcosh,
        TrackLinearVelocityYL2ExpAndLogcosh,
    ):
        with pytest.raises(ValueError, match=r"within \[0, 1\]"):
            term_class(
                context=runtime_context,
                model_context=model_context,
                logcosh_weight=logcosh_weight,
            )


def test_base_height_l2_normalizes_and_clamps_error(
    runtime_context,
    model_context,
):
    manager = RewardManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        terms={
            "base_height_l2": {
                "params": {
                    "target_height": 0.45,
                    "std": 0.05,
                    "max_normalized_error": 2.0,
                },
            },
        },
    )

    reward, _ = manager.compute(make_state_reward_context())

    torch.testing.assert_close(reward, torch.tensor([4.0, 4.0]))


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
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
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
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
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
    task_context.state.contact_forces.zero_()
    task_context.state.foot_ground_contact.zero_()

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
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
    )
    manager = RewardManager(
        num_envs=2,
        context=runtime_context,
        model_context=quadrupedal_model_context,
        terms={"quadrupedal_foot_velocity_diff_l2": {}},
    )
    task_context = make_quadrupedal_foot_context()
    task_context.state.geom_xvel = torch.zeros(2, 7, 6)
    task_context.state.geom_xvel[0, 3, 3:5] = torch.tensor([1.0, 2.0])
    task_context.state.geom_xvel[0, 6, 3:5] = torch.tensor([1.0, 4.0])
    task_context.state.geom_xvel[0, 4, 3:5] = torch.tensor([3.0, 0.0])
    task_context.state.geom_xvel[0, 5, 3:5] = torch.tensor([1.0, 0.0])
    task_context.state.geom_xvel[1, 3, 3:5] = torch.tensor([0.5, 0.5])
    task_context.state.geom_xvel[1, 6, 3:5] = torch.tensor([0.5, 0.5])
    task_context.state.geom_xvel[1, 4, 3:5] = torch.tensor([2.0, -1.0])
    task_context.state.geom_xvel[1, 5, 3:5] = torch.tensor([2.0, -1.0])

    reward, info = manager.compute(task_context)

    expected = torch.tensor([8.0, 0.0])
    torch.testing.assert_close(reward, expected)
    torch.testing.assert_close(
        info["reward/quadrupedal_foot_velocity_diff_l2"],
        expected,
    )


def test_foot_lift_height_reward_is_gated_by_command(
    runtime_context,
    model_context,
):
    quadrupedal_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
    )
    term = FootLiftHeightCommandWeightedExp(
        context=runtime_context,
        model_context=quadrupedal_model_context,
        target_height=0.08,
        height_std=0.03,
        command_std=0.5,
    )
    task_context = make_quadrupedal_foot_context()
    task_context.state.geom_xpos = torch.zeros(2, 7, 3)
    task_context.state.geom_xpos[:, 3:7, 2] = 0.08
    task_context.command["lin_vel_x"][0] = 0.5

    reward = term.compute(task_context)

    command_gate = 1.0 - torch.exp(torch.tensor(-1.0))
    torch.testing.assert_close(
        reward,
        torch.tensor([0.5, 0.0]) * command_gate,
    )

    task_context.state.geom_xpos[:, 3:7, 2] = 0.0
    low_reward = term.compute(task_context)
    assert low_reward[0] < reward[0] * 0.01
    assert reward[1] == 0.0


@pytest.mark.parametrize("command_std", [0.0, -1.0])
def test_foot_lift_height_reward_rejects_invalid_command_std(
    runtime_context,
    model_context,
    command_std,
):
    with pytest.raises(ValueError, match="'command_std' must be positive"):
        FootLiftHeightCommandWeightedExp(
            context=runtime_context,
            model_context=model_context,
            target_height=0.08,
            command_std=command_std,
        )


def test_quadrupedal_phase_gait_updates_phase_steps(
    runtime_context,
    model_context,
):
    phase_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
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


def test_trot_loop_duration_tracks_valid_contact_sequence_and_resets(
    runtime_context,
    model_context,
):
    gait_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
    )
    term = TrotLoopDurationTanh(
        num_envs=2,
        context=runtime_context,
        model_context=gait_model_context,
        growth_rate=2.0,
    )
    task_context = TaskContext(
        state=make_simulator_state(
            contact_geom_ids=torch.tensor([
                [[3, 0], [4, 0], [5, 0], [6, 0]],
                [[0, 3], [0, 4], [0, 5], [0, 6]],
            ]),
            foot_ground_contact=torch.eye(
                4, dtype=torch.bool,
            ).repeat(2, 1, 1),
            contact_forces=torch.zeros(2, 4, 6),
        ),
        command={
            "lin_vel_x": torch.tensor([[1.0], [0.0]]),
            "lin_vel_y": torch.zeros(2, 1),
            "ang_vel_z": torch.zeros(2, 1),
        },
        action=torch.zeros(2, 2),
        last_action=torch.zeros(2, 2),
        episode_step=torch.ones(2, dtype=torch.long),
        step_dt=0.02,
    )
    task_context.state.contact_forces[..., 0] = 20.0

    first_reward = term.compute(task_context)
    torch.testing.assert_close(
        first_reward,
        torch.tanh(torch.full((2,), 0.04)),
    )

    held_reward = term.compute(task_context)
    torch.testing.assert_close(
        held_reward,
        torch.tanh(torch.full((2,), 0.08)),
    )

    task_context.state.contact_geom_ids[0] = torch.tensor([
        [3, 0], [6, 0], [-1, -1], [-1, -1],
    ])
    task_context.state.foot_ground_contact[0] = torch.tensor([
        [1, 0, 0, 0], [0, 0, 0, 1],
        [0, 0, 0, 0], [0, 0, 0, 0],
    ], dtype=torch.bool)
    transitioned_reward = term.compute(task_context)
    torch.testing.assert_close(
        transitioned_reward,
        torch.tanh(torch.full((2,), 0.12)),
    )

    term.reset(torch.tensor([1]))
    torch.testing.assert_close(
        term.gait_loop_duration,
        torch.tensor([0.06, 0.0]),
    )
    assert term.gait_is_moving[0] is True
    assert term.gait_is_moving[1] is None


def test_trot_loop_duration_resets_on_invalid_contact_sequence(
    runtime_context,
    model_context,
):
    gait_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
    )
    term = TrotLoopDurationTanh(
        num_envs=1,
        context=runtime_context,
        model_context=gait_model_context,
        growth_rate=2.0,
    )
    task_context = TaskContext(
        state=make_simulator_state(
            num_envs=1,
            foot_ground_contact=torch.ones(1, 1, 4, dtype=torch.bool),
        ),
        command={
            "lin_vel_x": torch.ones(1, 1),
            "lin_vel_y": torch.zeros(1, 1),
            "ang_vel_z": torch.zeros(1, 1),
        },
        action=torch.zeros(1, 2),
        last_action=torch.zeros(1, 2),
        episode_step=torch.ones(1, dtype=torch.long),
        step_dt=0.02,
    )

    valid_reward = term.compute(task_context)
    torch.testing.assert_close(
        valid_reward,
        torch.tanh(torch.tensor([0.04])),
    )

    task_context.state.foot_ground_contact = torch.tensor(
        [[[1, 0, 0, 0]]], dtype=torch.bool,
    )
    invalid_reward = term.compute(task_context)
    torch.testing.assert_close(invalid_reward, torch.zeros(1))
    torch.testing.assert_close(term.gait_loop_duration, torch.zeros(1))

    task_context.state.foot_ground_contact.fill_(True)
    resumed_reward = term.compute(task_context)
    torch.testing.assert_close(
        resumed_reward,
        torch.tanh(torch.tensor([0.04])),
    )


def test_trot_loop_duration_treats_small_commands_as_idle(
    runtime_context,
    model_context,
):
    gait_model_context = replace(
        model_context,
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
    )
    term = TrotLoopDurationTanh(
        num_envs=1,
        context=runtime_context,
        model_context=gait_model_context,
    )
    task_context = TaskContext(
        state=make_simulator_state(
            num_envs=1,
            foot_ground_contact=torch.tensor(
                [[[True, False, False, True]]],
            ),
        ),
        command={
            "lin_vel_x": torch.tensor([[0.05]]),
            "lin_vel_y": torch.zeros(1, 1),
            "ang_vel_z": torch.zeros(1, 1),
        },
        action=torch.zeros(1, 2),
        last_action=torch.zeros(1, 2),
        episode_step=torch.ones(1, dtype=torch.long),
        step_dt=0.02,
    )

    idle_reward = term.compute(task_context)
    torch.testing.assert_close(idle_reward, torch.zeros(1))

    task_context.command["lin_vel_x"].fill_(0.1)
    moving_reward = term.compute(task_context)
    torch.testing.assert_close(
        moving_reward,
        torch.tanh(torch.tensor([0.02])),
    )


def test_quadrupedal_phase_gait_uses_base_plane_distance(
    runtime_context,
    model_context,
):
    phase_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
    )
    term = QuadrupedalGaitPhaseL2Exp(
        num_envs=1,
        context=runtime_context,
        model_context=phase_model_context,
        target_height=0.2,
    )
    task_context = make_phase_gait_context()
    task_context.state.qpos = torch.zeros(1, 9)
    task_context.state.qpos[:, 1] = 2.0 ** -0.5
    task_context.state.qpos[:, 3] = 2.0 ** -0.5
    task_context.state.geom_xpos = torch.zeros(1, 7, 3)
    task_context.state.geom_xpos[:, [3, 4, 5, 6], 0] = 0.2
    task_context.command["foot_phase_real"] = torch.ones(1, 4)
    task_context.command["foot_phase_imag"] = torch.zeros(1, 4)
    task_context.command["half_period_duration"] = torch.full((1,), 0.04)

    torch.testing.assert_close(
        term._foot_height_from_base_plane(task_context),
        torch.full((1, 4), 0.2),
    )
