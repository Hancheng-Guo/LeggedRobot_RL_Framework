import pytest
import torch
from dataclasses import replace

from envs.tasks.managers.observation.base import ObservationManager
from envs.tasks.managers.observation.terms.base import BaseObservationTerm
from envs.tasks.managers.observation.terms.registry import (
    OBSERVATION_CLASS_MAP,
)
from envs.tasks.utils.context import TaskContext


def make_task_context(num_envs: int = 2) -> TaskContext:
    qpos = torch.zeros(num_envs, 9)
    qpos[:, 1] = 1.0
    qpos[:, [0, 8]] = torch.tensor([0.25, -0.5])

    qvel = torch.zeros(num_envs, 8)
    qvel[:, 1:4] = torch.tensor([1.0, 2.0, 3.0])
    qvel[:, [0, 7]] = torch.tensor([4.0, 5.0])

    return TaskContext(
        state={"qpos": qpos, "qvel": qvel},
        command={
            "lin_vel_x": torch.full((num_envs, 1), 0.1),
            "lin_vel_y": torch.full((num_envs, 1), 0.2),
            "ang_vel_z": torch.full((num_envs, 1), 0.3),
        },
        action=torch.tensor([[0.6, -0.7]]).repeat(num_envs, 1),
        last_action=torch.zeros(num_envs, 2),
        episode_step=torch.arange(num_envs),
    )


def make_manager(runtime_context, model_context, clip=None):
    return ObservationManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
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


def test_observation_manager_validates_selected_env_count(
    runtime_context,
    model_context,
):
    manager = make_manager(runtime_context, model_context)
    context = make_task_context(num_envs=1)

    observation, _ = manager.compute(
        context,
        env_ids=torch.tensor([1]),
    )
    assert observation.shape == (1, 15)

    with pytest.raises(ValueError, match="expected 2"):
        manager.compute(context)


def test_observation_manager_rejects_non_matrix_term(
    runtime_context,
    model_context,
):
    class InvalidShape(BaseObservationTerm):
        def compute(self, task_context: TaskContext) -> torch.Tensor:
            return torch.zeros(task_context.action.shape[0])

    OBSERVATION_CLASS_MAP["invalid_shape"] = InvalidShape
    try:
        manager = ObservationManager(
            num_envs=2,
            context=runtime_context,
            model_context=model_context,
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
            terms={"projected_gravity": {}},
        )
