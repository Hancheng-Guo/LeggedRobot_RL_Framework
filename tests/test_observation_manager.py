import pytest
import torch
from dataclasses import replace

from envs.simulators.utils.state import SimulatorState
from envs.tasks.managers.observation.base import ObservationManager
from envs.tasks.managers.observation.terms.base import BaseObservationTerm
from envs.tasks.managers.observation.terms.action import LastAction
from envs.tasks.managers.observation.terms.foot import FootDurationTanh
from envs.tasks.managers.observation.terms.registry import (
    OBSERVATION_CLASS_MAP,
)
from envs.tasks.utils.context import TaskContext


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
        "foot_ground_contact": torch.empty(num_envs, 0, 0, dtype=torch.bool),
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
                [[[True], [False]]],
            ).repeat(num_envs, 1, 1),
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
