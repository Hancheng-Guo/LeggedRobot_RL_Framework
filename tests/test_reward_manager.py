import pytest
import torch
from dataclasses import replace
from typing import cast

from envs.tasks.managers.reward import RewardManager
from envs.tasks.managers.task_state import TaskStateManager
from envs.tasks.managers.reward.terms.foot import (
    FootDurationCubicCommandTanhWeightedExp,
    FootLiftHeightDiffCommandGatedL2,
    FootLiftHeightDiffCommandWeightedExp,
    FootStateDurationCommandWeightedExp,
)
from envs.tasks.managers.reward.terms.gait import (
    QuadrupedalCommandAdaptiveGaitPhaseHeightL2,
    QuadrupedalCommandAdaptiveGaitPhaseHeightL2Exp,
    TrotLoopDurationTanh,
)
from envs.tasks.managers.reward.terms.tracking import (
    TrackAngularVelocityZErrorIntegralL2,
    TrackLinearVelocityXL2Exp,
    TrackLinearVelocityXL2ExpAndLogcosh,
    TrackLinearVelocityYL2Exp,
    TrackLinearVelocityYL2ExpAndLogcosh,
    TrackLinearVelocityXErrorIntegralL2,
    TrackLinearVelocityYErrorIntegralL2,
)
from envs.tasks.utils import TaskContext
from envs.simulators.utils import SimulatorState


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
            num_envs, 0, dtype=torch.bool
        ),
        "foot_contact_normal_force": torch.empty(num_envs, 0),
    }
    values.update(overrides)
    return SimulatorState(**values)


def make_reward_context() -> TaskContext:
    return TaskContext(
        state=make_simulator_state(),
        command={},
        last_command={},
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
        last_command={
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
    geom_xpos[:, [3, 4, 5, 6], 2] = 0.3

    return TaskContext(
        state=make_simulator_state(
            qpos=qpos,
            geom_xpos=geom_xpos,
        ),
        command={
            "lin_vel_x": torch.zeros(2, 1),
            "lin_vel_y": torch.zeros(2, 1),
            "ang_vel_z": torch.zeros(2, 1),
            "foot_phase_real": torch.ones(2, 4),
            "foot_phase_imag": torch.zeros(2, 4),
        },
        last_command={
            "lin_vel_x": torch.zeros(2, 1),
            "lin_vel_y": torch.zeros(2, 1),
            "ang_vel_z": torch.zeros(2, 1),
            "foot_phase_real": torch.ones(2, 4),
            "foot_phase_imag": torch.zeros(2, 4),
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
                [1, 1, 0, 0],
                [0, 0, 1, 0],
            ], dtype=torch.bool),
            contact_forces=contact_forces,
        ),
        command={
            "lin_vel_x": torch.zeros(2, 1),
            "lin_vel_y": torch.zeros(2, 1),
            "ang_vel_z": torch.zeros(2, 1),
        },
        last_command={
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

    expected = torch.tensor([10.0, 8.0])
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
            "track_linear_velocity_x_error_integral_l2": {
                "params": {"integral_length": 3},
            },
        },
    )

    manager.compute(make_state_reward_context())
    term = cast(
        TrackLinearVelocityXErrorIntegralL2,
        manager.terms["track_linear_velocity_x_error_integral_l2"],
    )
    assert term.error_history.any()

    manager.reset(torch.tensor([1]))

    assert term.error_history[0].any()
    assert not term.error_history[1].any()


def test_integral_tracking_terms_average_signed_error_over_valid_window(
    runtime_context,
    model_context,
):
    manager = RewardManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        terms={
            "track_linear_velocity_x_error_integral_l2": {
                "params": {"integral_length": 2},
            },
            "track_linear_velocity_y_error_integral_l2": {
                "params": {"integral_length": 2},
            },
            "track_angular_velocity_z_error_integral_l2": {
                "params": {"integral_length": 2},
            },
        },
    )
    context = make_state_reward_context()
    context.step_dt = 0.1
    for command in context.command.values():
        command.zero_()
    context.state.base_lin_vel_body[:, :2] = torch.tensor([-1.0, -2.0])
    context.state.base_ang_vel_body[:, 2] = -1.0

    _, first = manager.compute(context)
    torch.testing.assert_close(
        first["reward/track_linear_velocity_x_error_integral_l2"],
        torch.full((2,), 1.0),
    )
    torch.testing.assert_close(
        first["reward/track_linear_velocity_y_error_integral_l2"],
        torch.full((2,), 4.0),
    )
    torch.testing.assert_close(
        first["reward/track_angular_velocity_z_error_integral_l2"],
        torch.full((2,), 1.0),
    )

    context.state.base_lin_vel_body[0, :2] = torch.tensor([1.0, 2.0])
    context.state.base_ang_vel_body[0, 2] = 1.0
    _, second = manager.compute(context)
    torch.testing.assert_close(
        second["reward/track_linear_velocity_x_error_integral_l2"],
        torch.tensor([0.0, 1.0]),
    )
    torch.testing.assert_close(
        second["reward/track_linear_velocity_y_error_integral_l2"],
        torch.tensor([0.0, 4.0]),
    )
    torch.testing.assert_close(
        second["reward/track_angular_velocity_z_error_integral_l2"],
        torch.tensor([0.0, 1.0]),
    )

    _, third = manager.compute(context)
    torch.testing.assert_close(
        third["reward/track_linear_velocity_x_error_integral_l2"],
        torch.full((2,), 1.0),
    )
    torch.testing.assert_close(
        third["reward/track_linear_velocity_y_error_integral_l2"],
        torch.full((2,), 4.0),
    )
    torch.testing.assert_close(
        third["reward/track_angular_velocity_z_error_integral_l2"],
        torch.full((2,), 1.0),
    )

    manager.reset(torch.tensor([0]))
    x_term = cast(
        TrackLinearVelocityXErrorIntegralL2,
        manager.terms["track_linear_velocity_x_error_integral_l2"],
    )
    y_term = cast(
        TrackLinearVelocityYErrorIntegralL2,
        manager.terms["track_linear_velocity_y_error_integral_l2"],
    )
    z_term = cast(
        TrackAngularVelocityZErrorIntegralL2,
        manager.terms["track_angular_velocity_z_error_integral_l2"],
    )
    assert not x_term.error_history[0].any()
    assert x_term.error_history[1].any()
    assert not y_term.error_history[0].any()
    assert y_term.error_history[1].any()
    assert not z_term.error_history[0].any()
    assert z_term.error_history[1].any()
    for term in (x_term, y_term, z_term):
        torch.testing.assert_close(term.integral_count, torch.tensor([0, 2]))

    context.state.base_lin_vel_body[0, 0] = -2.0
    _, after_reset = manager.compute(context)
    torch.testing.assert_close(
        after_reset["reward/track_linear_velocity_x_error_integral_l2"],
        torch.tensor([4.0, 1.0]),
    )
    for term in (x_term, y_term, z_term):
        torch.testing.assert_close(term.integral_count, torch.tensor([1, 2]))


@pytest.mark.parametrize("integral_length", [0, -1, 1.5, True])
def test_integral_tracking_terms_reject_invalid_length(
    runtime_context,
    model_context,
    integral_length,
):
    for term_class in (
        TrackLinearVelocityXErrorIntegralL2,
        TrackLinearVelocityYErrorIntegralL2,
        TrackAngularVelocityZErrorIntegralL2,
    ):
        with pytest.raises(ValueError, match="positive integer"):
            term_class(
                context=runtime_context,
                model_context=model_context,
                num_envs=2,
                integral_length=integral_length,
            )


def test_foot_state_duration_terms_track_joint_contact_state(
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
            "foot_state_duration_command_weighted_exp": {},
            "foot_state_duration_cubic_command_weighted_exp": {},
        },
    )
    task_context = make_quadrupedal_foot_context()

    reward, info = manager.compute(task_context)
    term = cast(
        FootStateDurationCommandWeightedExp,
        manager.terms["foot_state_duration_command_weighted_exp"],
    )

    assert reward.shape == (2,)
    assert info["reward/foot_state_duration_command_weighted_exp"].shape == (2,)
    assert info["reward/foot_state_duration_cubic_command_weighted_exp"].shape == (2,)
    torch.testing.assert_close(
        term.duration,
        torch.zeros(2),
    )

    task_context.command["lin_vel_x"].fill_(1.0)
    _, second_info = manager.compute(task_context)
    torch.testing.assert_close(term.duration, torch.full((2,), 0.02))
    torch.testing.assert_close(
        second_info["reward/foot_state_duration_command_weighted_exp"],
        torch.exp(torch.full((2,), -0.02)),
    )
    torch.testing.assert_close(
        second_info["reward/foot_state_duration_cubic_command_weighted_exp"],
        torch.exp(torch.full((2,), -(0.02 ** 3))),
    )

    # Changing one foot changes the joint state and resets only that env.
    task_context.state.foot_ground_contact[0, 0] = False
    manager.compute(task_context)
    torch.testing.assert_close(term.duration, torch.tensor([0.0, 0.04]))

    manager.compute(task_context)
    torch.testing.assert_close(term.duration, torch.tensor([0.02, 0.06]))

    manager.reset(torch.tensor([0]))
    torch.testing.assert_close(term.duration, torch.tensor([0.0, 0.06]))


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
            "foot_state_duration_command_weighted_exp": {},
        },
    )
    task_context = make_quadrupedal_foot_context()
    task_context.state.contact_forces.zero_()
    task_context.state.foot_ground_contact.zero_()

    manager.compute(task_context)
    term = cast(
        FootStateDurationCommandWeightedExp,
        manager.terms["foot_state_duration_command_weighted_exp"],
    )

    torch.testing.assert_close(
        term.duration,
        torch.full((2,), 0.02),
    )


def test_foot_duration_cubic_command_tanh_weighted_exp_tracks_each_foot(
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
        terms={"foot_duration_cubic_command_tanh_weighted_exp": {}},
    )
    task_context = make_quadrupedal_foot_context()
    task_context.command["lin_vel_x"][:, 0] = torch.tensor([1.0, 2.0])

    reward, _ = manager.compute(task_context)
    term = manager.terms["foot_duration_cubic_command_tanh_weighted_exp"]
    assert isinstance(term, FootDurationCubicCommandTanhWeightedExp)
    assert reward.shape == (2,)
    torch.testing.assert_close(
        term.duration,
        torch.tensor([[0.0, 0.0, 0.02, 0.02], [0.02, 0.02, 0.0, 0.02]]),
    )
    expected = torch.exp(
        -torch.tanh(torch.tensor([1.0, 2.0])).unsqueeze(-1)
        * term.duration.pow(3)
    ).mean(dim=-1)
    torch.testing.assert_close(reward, expected)

    task_context.state.foot_ground_contact[0, 0] = False
    manager.compute(task_context)
    torch.testing.assert_close(
        term.duration[0],
        torch.tensor([0.0, 0.02, 0.04, 0.04]),
    )
    manager.reset(torch.tensor([0]))
    torch.testing.assert_close(term.duration[0], torch.zeros(4))


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

    expected = torch.tensor([8.0 / 6.0, 0.0])
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
    term = FootLiftHeightDiffCommandWeightedExp(
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
        FootLiftHeightDiffCommandWeightedExp(
            context=runtime_context,
            model_context=model_context,
            target_height=0.08,
            command_std=command_std,
        )


def test_foot_lift_height_diff_command_gated_l2_tracks_height_only_when_moving(
    runtime_context,
    model_context,
):
    quadrupedal_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
    )
    term = FootLiftHeightDiffCommandGatedL2(
        context=runtime_context,
        model_context=quadrupedal_model_context,
        target_height=0.1,
        height_std=0.1,
    )
    task_context = make_quadrupedal_foot_context()
    task_context.state.geom_xpos = torch.zeros(2, 7, 3)
    task_context.state.geom_xpos[:, 3:7, 2] = torch.tensor([
        [0.0, 0.0, 0.1, 0.2],
        [0.0, 0.0, 0.1, 0.0],
    ])
    task_context.command["lin_vel_x"][0] = 0.2

    reward = term.compute(task_context)

    # Only the moving environment's swinging feet contribute height error.
    torch.testing.assert_close(reward, torch.tensor([0.25, 0.0]))


@pytest.mark.parametrize("height_std", [0, -0.1, float("inf"), float("nan")])
def test_foot_lift_height_diff_command_gated_l2_rejects_invalid_height_std(
    runtime_context,
    model_context,
    height_std,
):
    with pytest.raises(ValueError, match="'height_std' must be positive and finite"):
        FootLiftHeightDiffCommandGatedL2(
            context=runtime_context,
            model_context=model_context,
            target_height=0.1,
            height_std=height_std,
        )


def make_gait_phase_state(
    runtime_context,
    task_context: TaskContext,
    **params,
) -> TaskStateManager:
    manager = TaskStateManager(
        num_envs=2,
        context=runtime_context,
        terms={"quadrupedal_gait_phase": params},
    )
    task_context.task_state = manager.values()
    return manager


def test_command_adaptive_gait_phase_height_stops_below_command_threshold(
    runtime_context,
    model_context,
):
    phase_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
    )
    term = QuadrupedalCommandAdaptiveGaitPhaseHeightL2(
        context=runtime_context,
        num_envs=2,
        model_context=phase_model_context,
        target_height=0.2,
    )
    task_context = make_phase_gait_context()
    state_manager = make_gait_phase_state(
        runtime_context, task_context, omega_min=2.0, omega_max=2.0,
    )
    phase = state_manager.terms["quadrupedal_gait_phase"].value
    task_context.step_dt = 0.1
    task_context.state.geom_xpos[:, [3, 4, 5, 6], 2] = 0.3
    task_context.command["lin_vel_x"] = torch.tensor([[0.0], [0.2]])
    task_context.command["lin_vel_y"] = torch.zeros(2, 1)
    task_context.command["ang_vel_z"] = torch.zeros(2, 1)

    for _ in range(2):
        state_manager.update(task_context)
        reward = term.compute(task_context)
        assert reward.shape == (2,)
        assert torch.isfinite(reward).all()
    torch.testing.assert_close(
        phase,
        torch.tensor([[0.0, 0.0], [0.4, 0.0]]),
    )

    task_context.command["lin_vel_x"] = torch.tensor([[0.2], [0.0]])
    state_manager.update(task_context)
    term.compute(task_context)
    torch.testing.assert_close(
        phase,
        torch.tensor([[0.2, 0.0], [0.6, 0.0]]),
    )

    state_manager.reset(torch.tensor([1]))
    term.compute(task_context)
    torch.testing.assert_close(phase[1], torch.zeros(2))


def test_command_adaptive_gait_phase_height_lands_and_resumes(
    runtime_context,
    model_context,
):
    phase_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
    )
    term = QuadrupedalCommandAdaptiveGaitPhaseHeightL2(
        context=runtime_context,
        num_envs=2,
        model_context=phase_model_context,
        target_height=0.2,
    )
    task_context = make_phase_gait_context()
    state_manager = make_gait_phase_state(
        runtime_context, task_context, omega_min=torch.pi, omega_max=torch.pi,
    )
    phase = state_manager.terms["quadrupedal_gait_phase"].value
    task_context.step_dt = 0.25
    task_context.state.geom_xpos[:, [3, 4, 5, 6], 2] = 0.3
    task_context.command["lin_vel_x"] = torch.tensor([[0.0], [0.2]])
    task_context.command["lin_vel_y"] = torch.zeros(2, 1)
    task_context.command["ang_vel_z"] = torch.zeros(2, 1)

    for _ in range(6):
        state_manager.update(task_context)
        term.compute(task_context)
    torch.testing.assert_close(phase[0], torch.zeros(2))
    torch.testing.assert_close(
        phase[1],
        torch.tensor([1.5 * torch.pi, 0.5 * torch.pi]),
    )
    task_context.state.geom_xpos[1, [3, 4, 5, 6], 2] = torch.tensor(
        [0.3, 0.4, 0.4, 0.3]
    )
    torch.testing.assert_close(term._get_phase_height(task_context)[1], torch.tensor(
        [-0.22, -0.1, -0.1, -0.22]
    ))

    task_context.command["lin_vel_x"][1] = 0.0
    for _ in range(6):
        state_manager.update(task_context)
        term.compute(task_context)
    torch.testing.assert_close(
        phase[1],
        torch.full((2,), 2.0 * torch.pi),
    )
    task_context.state.geom_xpos[1, [3, 4, 5, 6], 2] = 0.3
    torch.testing.assert_close(term.compute(task_context), torch.full((2,), 0.01))

    task_context.command["lin_vel_x"][1] = 0.2
    state_manager.update(task_context)
    term.compute(task_context)
    torch.testing.assert_close(
        phase[1],
        torch.tensor([2.25 * torch.pi, 2.0 * torch.pi]),
    )
    state_manager.reset(torch.tensor([1]))
    torch.testing.assert_close(phase, torch.zeros(2, 2))


def test_command_adaptive_gait_phase_height_clamps_at_cycle_boundary(
    runtime_context,
    model_context,
):
    phase_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
    )
    term = QuadrupedalCommandAdaptiveGaitPhaseHeightL2(
        context=runtime_context,
        num_envs=2,
        model_context=phase_model_context,
        target_height=0.2,
    )
    task_context = make_phase_gait_context()
    state_manager = make_gait_phase_state(
        runtime_context, task_context, omega_min=2.0, omega_max=2.0,
    )
    phase = state_manager.terms["quadrupedal_gait_phase"].value
    task_context.step_dt = 0.5
    task_context.state.geom_xpos[:, [3, 4, 5, 6], 2] = 0.3
    task_context.command["lin_vel_x"] = torch.full((2, 1), 0.2)
    task_context.command["lin_vel_y"] = torch.zeros(2, 1)
    task_context.command["ang_vel_z"] = torch.zeros(2, 1)

    state_manager.update(task_context)
    term.compute(task_context)
    state_manager.update(task_context)
    term.compute(task_context)
    task_context.command["lin_vel_x"].zero_()
    for _ in range(5):
        state_manager.update(task_context)
        term.compute(task_context)

    torch.testing.assert_close(
        phase[:, 0],
        torch.full((2,), 2.0 * torch.pi),
    )
    torch.testing.assert_close(phase[:, 1], torch.zeros(2))
    torch.testing.assert_close(term.compute(task_context), torch.full((2,), 0.01))


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"target_height": 0.0}, "target_height"),
        ({"std": 0.0}, "std"),
        ({"std": float("nan")}, "std"),
    ],
)
@pytest.mark.parametrize(
    "term_class",
    (
        QuadrupedalCommandAdaptiveGaitPhaseHeightL2,
        QuadrupedalCommandAdaptiveGaitPhaseHeightL2Exp,
    ),
)
def test_command_adaptive_gait_phase_height_rejects_invalid_params(
    runtime_context,
    model_context,
    params,
    match,
    term_class,
):
    phase_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
    )
    with pytest.raises(ValueError, match=match):
        term_class(
            context=runtime_context,
            num_envs=2,
            model_context=phase_model_context,
            **{"target_height": 0.2, **params},
        )


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"omega_min": 0.0}, "omega_min"),
        ({"omega_max": 0.5}, "omega_max"),
        ({"omega_max": float("inf")}, "omega_max"),
        ({"command_gain": -1.0}, "command_gain"),
        ({"command_gain": float("nan")}, "command_gain"),
    ],
)
def test_gait_phase_state_rejects_invalid_params(
    runtime_context,
    params,
    match,
):
    with pytest.raises(ValueError, match=match):
        TaskStateManager(
            context=runtime_context,
            num_envs=2,
            terms={"quadrupedal_gait_phase": params},
        )


def test_command_adaptive_gait_phase_height_registers_and_resets(
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
            "quadrupedal_command_adaptive_gait_phase_height_l2": {
                "weight": -1.0,
                "params": {"target_height": 0.2},
            },
        },
    )
    assert isinstance(
        manager.terms["quadrupedal_command_adaptive_gait_phase_height_l2"],
        QuadrupedalCommandAdaptiveGaitPhaseHeightL2,
    )
    task_context = make_phase_gait_context()
    state_manager = make_gait_phase_state(
        runtime_context, task_context, omega_min=2.0, omega_max=2.0,
    )
    phase = state_manager.terms["quadrupedal_gait_phase"].value
    task_context.step_dt = 0.1
    task_context.command["lin_vel_x"].fill_(0.2)

    state_manager.update(task_context)
    reward, _ = manager.compute(task_context)
    assert reward.shape == (2,)
    assert torch.all(reward <= 0.0)
    torch.testing.assert_close(phase[:, 0], torch.full((2,), 0.2))
    repeated_reward, _ = manager.compute(task_context)
    torch.testing.assert_close(repeated_reward, reward)
    torch.testing.assert_close(phase[:, 0], torch.full((2,), 0.2))

    manager.reset(torch.tensor([1]))
    torch.testing.assert_close(phase[:, 0], torch.full((2,), 0.2))
    state_manager.reset(torch.tensor([1]))
    torch.testing.assert_close(phase[0], torch.tensor([0.2, 0.0]))
    torch.testing.assert_close(phase[1], torch.zeros(2))


def test_command_adaptive_gait_phase_height_matches_lift_and_error(
    runtime_context,
    model_context,
):
    phase_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
    )
    term = QuadrupedalCommandAdaptiveGaitPhaseHeightL2(
        num_envs=2,
        context=runtime_context,
        model_context=phase_model_context,
        target_height=0.2,
    )
    task_context = make_phase_gait_context()
    state_manager = make_gait_phase_state(
        runtime_context, task_context, omega_min=torch.pi, omega_max=torch.pi,
    )
    phase = state_manager.terms["quadrupedal_gait_phase"].value
    task_context.step_dt = 0.5
    task_context.command["lin_vel_x"].fill_(0.2)
    task_context.state.geom_xpos[0, [3, 4, 5, 6], 2] = torch.tensor(
        [0.4, 0.3, 0.3, 0.4]
    )

    state_manager.update(task_context)
    torch.testing.assert_close(
        term.compute(task_context),
        torch.tensor([0.005, 0.13]),
    )
    torch.testing.assert_close(
        term._get_phase_height(task_context)[0],
        torch.tensor([-0.1, -0.22, -0.22, -0.1]),
    )


def test_command_adaptive_gait_phase_height_uses_model_foot_order(
    runtime_context,
    model_context,
):
    phase_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FR", "FL", "RR", "RL"),
        geom_body_ids=torch.arange(7),
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
    )
    term = QuadrupedalCommandAdaptiveGaitPhaseHeightL2(
        num_envs=2,
        context=runtime_context,
        model_context=phase_model_context,
        target_height=0.2,
    )
    task_context = make_phase_gait_context()
    state_manager = make_gait_phase_state(
        runtime_context, task_context, omega_min=torch.pi, omega_max=torch.pi,
    )
    phase = state_manager.terms["quadrupedal_gait_phase"].value
    task_context.step_dt = 0.5
    task_context.command["lin_vel_x"].fill_(0.2)
    state_manager.update(task_context)
    term.compute(task_context)

    torch.testing.assert_close(
        term._get_phase_height(task_context)[0],
        torch.tensor([-0.1, -0.22, -0.22, -0.1]),
    )


def test_command_adaptive_gait_phase_height_applies_per_foot_phase_shift(
    runtime_context,
    model_context,
):
    phase_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
    )
    term = QuadrupedalCommandAdaptiveGaitPhaseHeightL2(
        context=runtime_context,
        model_context=phase_model_context,
        target_height=0.2,
    )
    task_context = make_phase_gait_context()
    make_gait_phase_state(runtime_context, task_context)
    task_context.command["lin_vel_x"].fill_(0.2)
    task_context.command["foot_phase_real"][0, 0] = 0.0
    task_context.command["foot_phase_imag"][0, 0] = 1.0

    reward = term.compute(task_context)

    torch.testing.assert_close(reward, torch.tensor([0.07, 0.01]))
    torch.testing.assert_close(
        term._get_phase_height(task_context)[0],
        torch.tensor([-0.1, -0.22, -0.22, -0.22]),
    )


def test_command_adaptive_gait_phase_height_l2_exp_registers_and_averages(
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
            "quadrupedal_command_adaptive_gait_phase_height_l2_exp": {
                "weight": 1.5,
                "params": {"target_height": 0.2},
            },
        },
    )
    term = manager.terms["quadrupedal_command_adaptive_gait_phase_height_l2_exp"]
    assert isinstance(term, QuadrupedalCommandAdaptiveGaitPhaseHeightL2Exp)
    assert term.std == 0.05

    task_context = make_phase_gait_context()
    make_gait_phase_state(runtime_context, task_context)
    task_context.command["foot_phase_real"][0, 0] = 0.0
    task_context.command["foot_phase_imag"][0, 0] = 1.0

    reward, info = manager.compute(task_context)
    landed = torch.exp(torch.tensor(-0.16))
    lifted = torch.exp(torch.tensor(-4.0))
    expected = 1.5 * torch.stack(((lifted + 3 * landed) / 4, landed))
    torch.testing.assert_close(reward, expected)
    torch.testing.assert_close(
        info["reward/quadrupedal_command_adaptive_gait_phase_height_l2_exp"],
        expected,
    )


@pytest.mark.parametrize("name", ["foot_phase_real", "foot_phase_imag"])
def test_command_adaptive_gait_phase_height_checks_phase_shift_shape(
    runtime_context,
    model_context,
    name,
):
    phase_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
    )
    term = QuadrupedalCommandAdaptiveGaitPhaseHeightL2(
        context=runtime_context,
        model_context=phase_model_context,
        target_height=0.2,
    )
    task_context = make_phase_gait_context()
    make_gait_phase_state(runtime_context, task_context)
    task_context.command[name] = torch.zeros(2, 3)

    with pytest.raises(ValueError, match=f"'{name}' must have shape"):
        term.compute(task_context)


def test_command_adaptive_gait_phase_height_scales_with_command(
    runtime_context,
    model_context,
):
    phase_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
    )
    term = QuadrupedalCommandAdaptiveGaitPhaseHeightL2(
        num_envs=2,
        context=runtime_context,
        model_context=phase_model_context,
        target_height=0.2,
    )
    task_context = make_phase_gait_context()
    state_manager = make_gait_phase_state(runtime_context, task_context)
    phase = state_manager.terms["quadrupedal_gait_phase"].value
    task_context.step_dt = 0.05
    task_context.command["lin_vel_x"] = torch.tensor([[0.2], [1.0]])

    state_manager.update(task_context)
    term.compute(task_context)
    command_norm = torch.tensor([0.2, 1.0])
    expected_omega = 4.0 + 6.0 * torch.tanh(command_norm)
    torch.testing.assert_close(phase[:, 0], expected_omega * 0.05)
    torch.testing.assert_close(phase[:, 1], torch.zeros(2))


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
            foot_ground_contact=torch.ones(2, 4, dtype=torch.bool),
            contact_forces=torch.zeros(2, 4, 6),
        ),
        command={
            "lin_vel_x": torch.tensor([[1.0], [0.0]]),
            "lin_vel_y": torch.zeros(2, 1),
            "ang_vel_z": torch.zeros(2, 1),
        },
        last_command={
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
        1, 0, 0, 1,
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
            foot_ground_contact=torch.ones(1, 4, dtype=torch.bool),
        ),
        command={
            "lin_vel_x": torch.ones(1, 1),
            "lin_vel_y": torch.zeros(1, 1),
            "ang_vel_z": torch.zeros(1, 1),
        },
        last_command={
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
        [[1, 0, 0, 0]], dtype=torch.bool,
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
                [[True, False, False, True]],
            ),
        ),
        command={
            "lin_vel_x": torch.tensor([[0.05]]),
            "lin_vel_y": torch.zeros(1, 1),
            "ang_vel_z": torch.zeros(1, 1),
        },
        last_command={
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


def test_command_adaptive_gait_phase_height_uses_base_plane_distance(
    runtime_context,
    model_context,
):
    phase_model_context = replace(
        model_context,
        geom_names=("floor", "base", "thigh", "FL", "FR", "RL", "RR"),
        geom_body_ids=torch.arange(7),
        foot_geom_ids=torch.tensor([3, 4, 5, 6]),
    )
    term = QuadrupedalCommandAdaptiveGaitPhaseHeightL2(
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
    torch.testing.assert_close(
        term._foot_height_from_base_plane(task_context),
        torch.full((1, 4), 0.2),
    )
