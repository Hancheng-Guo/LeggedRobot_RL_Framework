import pytest
import torch
from dataclasses import replace

from envs.simulators.utils import SimulatorState
from envs.tasks import BaseTaskLogic
from envs.tasks.managers.observation import ObservationManager
from envs.tasks.managers.observation.terms import BaseObservationTerm
from envs.tasks.managers.observation.terms.action import LastAction
from envs.tasks.managers.observation.terms.foot import FootDurationTanh
from envs.tasks.managers.task_state import TaskStateManager
from envs.tasks.managers.observation.terms import (
    OBSERVATION_CLASS_MAP,
)
from envs.tasks.utils import TaskContext


def make_simulator_state(
    num_envs: int,
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
        "foot_ground_contact": torch.empty(num_envs, 0, dtype=torch.bool),
        "foot_contact_normal_force": torch.empty(num_envs, 0),
    }
    values.update(overrides)
    return SimulatorState(**values)


def make_task_context(num_envs: int = 2) -> TaskContext:
    qpos = torch.zeros(num_envs, 9)
    qpos[:, 1] = 1.0
    qpos[:, [0, 8]] = torch.tensor([0.25, -0.5])

    qvel = torch.zeros(num_envs, 8)
    qvel[:, 1:4] = torch.tensor([1.0, 2.0, 3.0])
    qvel[:, 4:7] = torch.tensor([0.5, -0.25, 0.1])
    qvel[:, [0, 7]] = torch.tensor([4.0, 5.0])

    return TaskContext(
        state=make_simulator_state(num_envs, **{
            "qpos": qpos,
            "qvel": qvel,
            "actuator_force": torch.tensor(
                [[2.0, -3.0]],
            ).repeat(num_envs, 1),
            "geom_xpos": torch.tensor(
                [[[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.2],
                  [0.0, 0.0, 0.4],
                  [0.0, 0.0, 0.125]]],
            ).repeat(num_envs, 1, 1),
            "contact_geom_ids": torch.tensor(
                [[[3, 0], [1, 2]]],
            ).repeat(num_envs, 1, 1),
            "foot_ground_contact": torch.tensor(
                [[True]],
            ).repeat(num_envs, 1),
            "foot_contact_normal_force": torch.full((num_envs, 1), 12.0),
            "contact_forces": torch.tensor(
                [[[12.0, 1.0, 2.0, 0.0, 0.0, 0.0],
                  [50.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            ).repeat(num_envs, 1, 1),
        }),
        command={
            "lin_vel_x": torch.full((num_envs, 1), 0.1),
            "lin_vel_y": torch.full((num_envs, 1), 0.2),
            "ang_vel_z": torch.full((num_envs, 1), 0.3),
        },
        last_command={
            "lin_vel_x": torch.full((num_envs, 1), 0.1),
            "lin_vel_y": torch.full((num_envs, 1), 0.2),
            "ang_vel_z": torch.full((num_envs, 1), 0.3),
        },
        action=torch.tensor([[0.6, -0.7]]).repeat(num_envs, 1),
        last_action=torch.zeros(num_envs, 2),
        episode_step=torch.arange(num_envs),
        step_dt=0.02,
    )


def make_manager(runtime_context, model_context, clip=None):
    return ObservationManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        command_dim=3,
        action_dim=2,
        clip=clip,
        terms={
            "base_angular_velocity": {"scale": 0.5},
            "projected_gravity": {},
            "command": {},
            "joint_position": {},
            "joint_velocity": {"scale": 0.1},
            "last_action": {},
        },
    )


def test_gait_phase_observation_reads_shared_state(
    runtime_context,
    model_context,
):
    task_context = make_task_context()
    state_manager = TaskStateManager(
        num_envs=2,
        context=runtime_context,
        terms={"quadrupedal_gait_phase": {"omega_min": 2.0, "omega_max": 2.0}},
    )
    task_context.task_state = state_manager.values()
    observation_manager = ObservationManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        command_dim=3,
        action_dim=2,
        terms={"quadrupedal_gait_phase": {}},
    )

    initial, _ = observation_manager.compute(task_context)
    torch.testing.assert_close(initial, torch.tensor([[0., 0., 1., 1.]] * 2))
    assert observation_manager.output_dim == 4

    state_manager.update(task_context)
    phase = state_manager.terms["quadrupedal_gait_phase"].value
    expected = torch.cat((phase.sin(), phase.cos()), dim=-1)
    observed, _ = observation_manager.compute(task_context)
    torch.testing.assert_close(observed, expected)
    repeated, _ = observation_manager.compute(task_context)
    torch.testing.assert_close(repeated, expected)

    state_manager.reset(torch.tensor([1]))
    task_context.env_ids = torch.tensor([1])
    task_context.task_state = state_manager.values(task_context.env_ids)
    selected, _ = observation_manager.compute(task_context, env_ids=task_context.env_ids)
    torch.testing.assert_close(selected, torch.tensor([[0., 0., 1., 1.]]))
    torch.testing.assert_close(phase[0], torch.tensor([0.04, 0.0]))

    task_context.command = {
        name: value[task_context.env_ids]
        for name, value in task_context.command.items()
    }
    task_context.command["lin_vel_x"].fill_(0.2)
    task_context.step_dt = 0.1
    state_manager.update(task_context)
    torch.testing.assert_close(phase[0], torch.tensor([0.04, 0.0]))
    torch.testing.assert_close(phase[1], torch.tensor([0.2, 0.0]))
    torch.testing.assert_close(
        task_context.task_state["quadrupedal_gait_phase"], phase[1:2],
    )


def test_gait_phase_observation_requires_state(runtime_context, model_context):
    manager = ObservationManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        command_dim=3,
        action_dim=2,
        terms={"quadrupedal_gait_phase": {}},
    )
    with pytest.raises(ValueError, match="quadrupedal_gait_phase"):
        manager.compute(make_task_context())


def test_task_exposes_gait_phase_without_reward(runtime_context, model_context):
    task = BaseTaskLogic(runtime_context)
    task.num_envs = 2
    task.model_context = model_context
    task._build_managers(
        action_manager_config={"terms": {}},
        command_manager_config={
            "terms": {
                name: {
                    "type": "UniformOnReset",
                    "params": {"min_value": 0.0, "max_value": 0.0},
                }
                for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
            },
        },
        task_state_manager_config={
            "terms": {
                "quadrupedal_gait_phase": {"omega_min": 2.0, "omega_max": 2.0},
            },
        },
        observation_manager_config={
            "terms": {"quadrupedal_gait_phase": {}},
        },
        reward_manager_config={"terms": {}},
        termination_manager_config={"terms": {}},
        constants={},
    )
    state = make_task_context().state
    task.command["lin_vel_x"].fill_(0.2)
    context = task.build_task_context(state, torch.ones(2, dtype=torch.long), 0.1)
    task.update_task_state(context)
    observation, _ = task.compute_observation(context)
    expected_phase = torch.tensor([[0.2, 0.0]] * 2)
    torch.testing.assert_close(context.task_state["quadrupedal_gait_phase"], expected_phase)
    torch.testing.assert_close(
        observation,
        torch.cat((expected_phase.sin(), expected_phase.cos()), dim=-1),
    )

    task.reset(torch.tensor([1]))
    selected = task.build_task_context(
        state, torch.zeros(1, dtype=torch.long), 0.1, env_ids=torch.tensor([1]),
    )
    reset_observation, _ = task.compute_observation(selected, env_ids=selected.env_ids)
    torch.testing.assert_close(reset_observation, torch.tensor([[0., 0., 1., 1.]]))


def test_observation_manager_scales_and_concatenates_in_config_order(
    runtime_context,
    model_context,
):
    manager = make_manager(runtime_context, model_context)
    observation, info = manager.compute(make_task_context())

    expected_row = torch.tensor([
        0.5, 1.0, 1.5,
        0.0, 0.0, -1.0,
        0.1, 0.2, 0.3,
        0.25, -0.5,
        0.4, 0.5,
        0.6, -0.7,
    ])
    torch.testing.assert_close(observation[0], expected_row)
    assert observation.shape == (2, 15)
    assert list(info) == [
        "observation/base_angular_velocity",
        "observation/projected_gravity",
        "observation/command",
        "observation/joint_position",
        "observation/joint_velocity",
        "observation/last_action",
    ]


def test_observation_manager_clips_final_observation(
    runtime_context,
    model_context,
):
    manager = make_manager(runtime_context, model_context, clip=0.25)
    observation, _ = manager.compute(make_task_context())
    assert torch.all(observation <= 0.25)
    assert torch.all(observation >= -0.25)


def test_last_action_history_tracks_steps_and_partial_reset(
    runtime_context,
    model_context,
):
    manager = ObservationManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        command_dim=3,
        action_dim=2,
        terms={"last_action": {"lags": 3}},
    )
    context = make_task_context()
    context.episode_step[:] = 0
    context.action.zero_()
    manager.compute(context)

    context.episode_step[:] = 1
    context.action[:] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    first, _ = manager.compute(context)
    torch.testing.assert_close(first, torch.tensor([
        [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
    ]))
    repeated, _ = manager.compute(context)
    torch.testing.assert_close(repeated, first)

    context.episode_step[:] = 2
    context.action[:] = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    second, _ = manager.compute(context)
    torch.testing.assert_close(second, torch.tensor([
        [5.0, 6.0, 1.0, 2.0, 0.0, 0.0],
        [7.0, 8.0, 3.0, 4.0, 0.0, 0.0],
    ]))

    manager.reset(torch.tensor([1]))
    reset_context = make_task_context(num_envs=1)
    reset_context.env_ids = torch.tensor([1])
    reset_context.episode_step[:] = 0
    reset_context.action.zero_()
    reset_observation, _ = manager.compute(
        reset_context, env_ids=reset_context.env_ids,
    )
    torch.testing.assert_close(reset_observation, torch.zeros(1, 6))
    term = manager.terms["last_action"]
    assert isinstance(term, LastAction)
    torch.testing.assert_close(
        term.history[0],
        torch.tensor([[5.0, 6.0], [1.0, 2.0], [0.0, 0.0]]),
    )

    reset_context.episode_step[:] = 1
    reset_context.action[:] = torch.tensor([[9.0, 10.0]])
    resumed, _ = manager.compute(reset_context, env_ids=reset_context.env_ids)
    torch.testing.assert_close(
        resumed, torch.tensor([[9.0, 10.0, 0.0, 0.0, 0.0, 0.0]]),
    )


def test_velocity_error_integral_observations_track_signed_history(
    runtime_context,
    model_context,
):
    manager = ObservationManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        command_dim=3,
        action_dim=2,
        terms={
            "track_linear_velocity_x_error_integral": {
                "integral_length": 2,
            },
            "track_linear_velocity_y_error_integral": {
                "integral_length": 2,
            },
            "track_angular_velocity_z_error_integral": {
                "integral_length": 2,
            },
        },
    )
    context = make_task_context()
    context.step_dt = 0.1
    context.episode_step.zero_()
    for command in context.command.values():
        command.zero_()
    context.state.base_lin_vel_body[:, :2] = torch.tensor([-1.0, -2.0])
    context.state.base_ang_vel_body[:, 2] = -1.0

    initial, _ = manager.compute(context)
    torch.testing.assert_close(initial, torch.zeros(2, 3))

    context.last_command = {
        name: value.clone() for name, value in context.command.items()
    }
    context.command["lin_vel_x"].fill_(5.0)
    context.command["ang_vel_z"].fill_(3.0)
    context.episode_step[:] = 1
    first, _ = manager.compute(context)
    torch.testing.assert_close(
        first, torch.tensor([[0.1, 0.2, 0.1], [0.1, 0.2, 0.1]]),
    )
    repeated, _ = manager.compute(context)
    torch.testing.assert_close(repeated, first)

    context.episode_step[:] = 2
    context.state.base_lin_vel_body[0, :2] = torch.tensor([1.0, 2.0])
    context.state.base_ang_vel_body[0, 2] = 1.0
    second, _ = manager.compute(context)
    torch.testing.assert_close(
        second, torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.4, 0.2]]),
    )

    context.episode_step[:] = 3
    third, _ = manager.compute(context)
    torch.testing.assert_close(
        third, torch.tensor([[-0.2, -0.4, -0.2], [0.2, 0.4, 0.2]]),
    )

    manager.reset(torch.tensor([1]))
    reset_context = make_task_context(num_envs=1)
    reset_context.env_ids = torch.tensor([1])
    reset_context.episode_step.zero_()
    reset_observation, _ = manager.compute(
        reset_context, env_ids=reset_context.env_ids,
    )
    torch.testing.assert_close(reset_observation, torch.zeros(1, 3))


@pytest.mark.parametrize("integral_length", [0, -1, 1.5, True])
def test_velocity_error_integral_observations_reject_invalid_length(
    runtime_context,
    model_context,
    integral_length,
):
    with pytest.raises(ValueError, match="positive integer"):
        ObservationManager(
            num_envs=2,
            context=runtime_context,
            model_context=model_context,
            command_dim=3,
            action_dim=2,
            terms={
                "track_linear_velocity_x_error_integral": {
                    "integral_length": integral_length,
                },
            },
        )


def test_tanh_velocity_error_integrals_map_signed_integrals(
    runtime_context,
    model_context,
):
    manager = ObservationManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        command_dim=3,
        action_dim=2,
        terms={
            "track_linear_velocity_x_error_integral": {
                "integral_length": 2,
            },
            "track_linear_velocity_y_error_integral": {
                "integral_length": 2,
            },
            "track_linear_velocity_x_error_integral_tanh": {
                "integral_length": 2, "alpha": 2.0,
            },
            "track_linear_velocity_y_error_integral_tanh": {
                "integral_length": 2, "alpha": 2.0,
            },
            "track_angular_velocity_z_error_integral": {
                "integral_length": 2,
            },
            "track_angular_velocity_z_error_integral_tanh": {
                "integral_length": 2, "alpha": 2.0,
            },
        },
    )
    context = make_task_context()
    context.step_dt = 0.1
    context.episode_step[:] = 1
    for command in context.last_command.values():
        command.zero_()
    context.state.base_lin_vel_body[:, :2] = torch.tensor([-1.0, 2.0])
    context.state.base_ang_vel_body[:, 2] = -3.0

    observation, _ = manager.compute(context)
    xy_integral = torch.tensor([[0.1, -0.2]]).repeat(2, 1)
    z_integral = torch.full((2, 1), 0.3)
    expected = torch.cat((
        xy_integral,
        torch.tanh(2.0 * xy_integral[:, :1]),
        torch.tanh(2.0 * xy_integral[:, 1:2]),
        z_integral,
        torch.tanh(2.0 * z_integral),
    ), dim=-1)
    torch.testing.assert_close(observation, expected)

    repeated, _ = manager.compute(context)
    torch.testing.assert_close(repeated, expected)


@pytest.mark.parametrize("alpha", [0, -1.0, float("nan"), float("inf"), True])
def test_tanh_velocity_error_integrals_reject_invalid_alpha(
    runtime_context,
    model_context,
    alpha,
):
    for term_name in (
        "track_linear_velocity_x_error_integral_tanh",
        "track_linear_velocity_y_error_integral_tanh",
        "track_angular_velocity_z_error_integral_tanh",
    ):
        with pytest.raises(ValueError, match="finite positive"):
            ObservationManager(
                num_envs=2,
                context=runtime_context,
                model_context=model_context,
                command_dim=3,
                action_dim=2,
                terms={term_name: {"alpha": alpha}},
            )


def test_observation_manager_validates_selected_env_count(
    runtime_context,
    model_context,
):
    manager = make_manager(runtime_context, model_context)
    context = make_task_context(num_envs=1)
    context.env_ids = torch.tensor([1])

    observation, _ = manager.compute(
        context,
        env_ids=context.env_ids,
    )
    assert observation.shape == (1, 15)

    context.env_ids = None
    with pytest.raises(ValueError, match="expected 2"):
        manager.compute(context)


def test_observation_manager_rejects_non_matrix_term(
    runtime_context,
    model_context,
):
    class InvalidShape(BaseObservationTerm):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.output_dim = 1

        def compute(self, task_context: TaskContext) -> torch.Tensor:
            return torch.zeros(task_context.action.shape[0])

    OBSERVATION_CLASS_MAP["invalid_shape"] = InvalidShape
    try:
        manager = ObservationManager(
            num_envs=2,
            context=runtime_context,
            model_context=model_context,
            command_dim=3,
            action_dim=2,
            terms={"invalid_shape": {}},
        )
        with pytest.raises(ValueError, match="must return a 2D tensor"):
            manager.compute(make_task_context())
    finally:
        OBSERVATION_CLASS_MAP.pop("invalid_shape", None)


def test_projected_gravity_uses_model_gravity(
    runtime_context,
    model_context,
):
    model_context = replace(
        model_context,
        gravity=torch.tensor([0.0, -9.81, 0.0]),
    )
    manager = ObservationManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        command_dim=3,
        action_dim=2,
        terms={"projected_gravity": {}},
    )

    observation, _ = manager.compute(make_task_context())

    torch.testing.assert_close(
        observation,
        torch.tensor([[0.0, -1.0, 0.0]]).repeat(2, 1),
    )


def test_projected_gravity_rejects_zero_model_gravity(
    runtime_context,
    model_context,
):
    model_context = replace(
        model_context,
        gravity=torch.zeros(3),
    )

    with pytest.raises(ValueError, match="non-zero gravity"):
        ObservationManager(
            num_envs=2,
            context=runtime_context,
            model_context=model_context,
            command_dim=3,
            action_dim=2,
            terms={"projected_gravity": {}},
        )


def test_additional_state_observation_terms(
    runtime_context,
    model_context,
):
    manager = ObservationManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        command_dim=3,
        action_dim=2,
        terms={
            "base_linear_velocity": {},
            "base_position": {},
            "base_height": {},
            "joint_position_diff": {},
            "actuator_force": {},
        },
    )

    observation, _ = manager.compute(make_task_context())

    expected = torch.tensor([
        0.5, -0.25, 0.1,
        0.0, 0.0, 0.0,
        0.0,
        0.25, -0.5,
        2.0, -3.0,
    ])
    torch.testing.assert_close(observation[0], expected)
    assert observation.shape == (2, 11)


def test_foot_contact_observation_terms(
    runtime_context,
    model_context,
):
    manager = ObservationManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        command_dim=3,
        action_dim=2,
        terms={
            "foot_contact_normal_force": {},
            "foot_contact_state": {},
        },
    )

    observation, _ = manager.compute(make_task_context())

    torch.testing.assert_close(
        observation,
        torch.tensor([[12.0, 1.0]]).repeat(2, 1),
    )


def test_foot_contact_normal_force_reads_simulator_summary(
    runtime_context,
    model_context,
):
    manager = ObservationManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        command_dim=3,
        action_dim=2,
        terms={"foot_contact_normal_force": {}},
    )
    task_context = make_task_context()
    task_context.state.foot_ground_contact[1] = False
    task_context.state.foot_contact_normal_force = torch.tensor([
        [15.0], [0.0],
    ])

    observation, _ = manager.compute(task_context)

    torch.testing.assert_close(observation, torch.tensor([[15.0], [0.0]]))


def test_foot_duration_tanh_uses_signed_contact_duration(
    runtime_context,
    model_context,
):
    manager = ObservationManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        command_dim=3,
        action_dim=2,
        terms={"foot_duration_tanh": {"alpha": 2.0}},
    )
    term = manager.terms["foot_duration_tanh"]
    assert isinstance(term, FootDurationTanh)
    task_context = make_task_context()

    first_observation, _ = manager.compute(task_context)
    torch.testing.assert_close(first_observation, torch.zeros(2, 1))

    second_observation, _ = manager.compute(task_context)
    torch.testing.assert_close(
        second_observation,
        torch.tanh(torch.full((2, 1), 0.04)),
    )

    task_context.state.foot_ground_contact.zero_()
    switched_observation, _ = manager.compute(task_context)
    torch.testing.assert_close(switched_observation, torch.zeros(2, 1))

    airborne_observation, _ = manager.compute(task_context)
    torch.testing.assert_close(
        airborne_observation,
        -torch.tanh(torch.full((2, 1), 0.04)),
    )

    manager.reset(torch.tensor([1]))
    assert (
        term.duration[0].item()
        == pytest.approx(0.02)
    )
    assert (
        term.duration[1].item()
        == pytest.approx(0.0)
    )
    reset_context = make_task_context(num_envs=1)
    reset_context.state.foot_ground_contact.zero_()
    reset_context.env_ids = torch.tensor([1])
    reset_observation, _ = manager.compute(
        reset_context,
        env_ids=reset_context.env_ids,
    )
    torch.testing.assert_close(reset_observation, torch.zeros(1, 1))


def test_foot_duration_tanh_updates_only_selected_envs(
    runtime_context,
    model_context,
):
    manager = ObservationManager(
        num_envs=3,
        context=runtime_context,
        model_context=model_context,
        command_dim=3,
        action_dim=2,
        terms={"foot_duration_tanh": {}},
    )
    term = manager.terms["foot_duration_tanh"]
    assert isinstance(term, FootDurationTanh)
    term.duration[:] = torch.tensor([[0.1], [0.2], [0.3]])
    term.last_foot_state.fill_(True)
    term.initialized.fill_(True)
    env_ids = torch.tensor([2, 0])
    task_context = make_task_context(num_envs=2)
    task_context.env_ids = env_ids

    observation, _ = manager.compute(task_context, env_ids=env_ids)

    torch.testing.assert_close(
        observation,
        torch.tanh(torch.tensor([[0.32], [0.12]])),
    )
    torch.testing.assert_close(
        term.duration,
        torch.tensor([[0.12], [0.2], [0.32]]),
    )


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("nan"), float("inf")])
def test_foot_duration_tanh_rejects_invalid_alpha(
    runtime_context,
    model_context,
    alpha,
):
    with pytest.raises(ValueError, match="finite positive"):
        ObservationManager(
            num_envs=2,
            context=runtime_context,
            model_context=model_context,
            command_dim=3,
            action_dim=2,
            terms={"foot_duration_tanh": {"alpha": alpha}},
        )


def test_foot_height_observation_term(
    runtime_context,
    model_context,
):
    manager = ObservationManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        command_dim=3,
        action_dim=2,
        terms={"foot_height": {}},
    )

    observation, _ = manager.compute(make_task_context())

    torch.testing.assert_close(
        observation,
        torch.full((2, 1), 0.125),
    )
