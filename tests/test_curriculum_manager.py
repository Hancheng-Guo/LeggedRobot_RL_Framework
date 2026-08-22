import torch
import pytest
from typing import cast

from envs.tasks.managers.command.base import CommandManager
from envs.tasks.managers.command.terms.curriculum import (
    CurriculumSampleOnReset,
)
from envs.tasks.managers.curriculum.base import CurriculumManager
from envs.tasks.managers.curriculum.terms.command import CommandReward


def make_command_terms():
    return {
        "x": {
            "type": "CurriculumSampleOnReset",
            "params": {
                "min_value": -1.0,
                "max_value": 1.0,
                "num_bins": 3,
                "group": "motion",
                "noise_scale": 0.0,
            },
        },
        "yaw": {
            "type": "CurriculumSampleOnReset",
            "params": {
                "min_value": -2.0,
                "max_value": 2.0,
                "num_bins": 2,
                "group": "motion",
                "noise_scale": 0.0,
            },
        },
    }


def make_curriculum(runtime_context, model_context, num_envs=4):
    return CurriculumManager(
        num_envs=num_envs,
        context=runtime_context,
        model_context=model_context,
        terms={
            "command_reward": {
                "temperature": 1.0,
                "exploration": 0.0,
            },
        },
        manager_configs={
            "command_manager_config": {
                "terms": make_command_terms(),
                "constraints": {
                    "yaw": {"operator": "<=", "expression": "{x}"},
                },
            },
        },
    )


def test_curriculum_buffer_represents_joint_command_space(
    runtime_context,
    model_context,
):
    manager = make_curriculum(runtime_context, model_context)
    term = cast(CommandReward, manager.get_term("command_reward"))
    buffer = term.buffers["motion"]

    assert buffer.dimension_names == ("x", "yaw")
    assert buffer.command_values.shape == (3, 2)
    assert buffer.reward_sum.shape == (3,)
    assert buffer.assigned_cell_ids.shape == (4,)
    x_id = buffer.dimension_names.index("x")
    yaw_id = buffer.dimension_names.index("yaw")
    assert torch.all(
        buffer.command_values[:, yaw_id]
        <= buffer.command_values[:, x_id]
    )
    torch.testing.assert_close(
        buffer.command_values,
        torch.tensor([
            [-1.0, -2.0],
            [0.0, -2.0],
            [1.0, -2.0],
        ]),
    )


def test_lower_reward_cells_have_higher_sampling_probability(
    runtime_context,
    model_context,
):
    manager = make_curriculum(runtime_context, model_context)
    term = cast(CommandReward, manager.get_term("command_reward"))
    buffer = term.buffers["motion"]
    buffer.sample_count.fill_(1)
    buffer.reward_sum.copy_(
        torch.tensor([-2.0, -1.0, 0.0])
    )

    probabilities = term.probabilities("motion")

    assert torch.all(probabilities[:-1] > probabilities[1:])


def test_command_terms_share_one_joint_space_sample(
    runtime_context,
    model_context,
):
    curriculum = make_curriculum(runtime_context, model_context)
    command = CommandManager(
        num_envs=4,
        context=runtime_context,
        model_context=model_context,
        curriculum_manager=curriculum,
        terms=make_command_terms(),
        constraints={
            "yaw": {"operator": "<=", "expression": "{x}"},
        },
    )

    curriculum.reset()
    command.reset()

    term = cast(CommandReward, curriculum.get_term("command_reward"))
    buffer = term.buffers["motion"]
    expected = buffer.command_values[buffer.assigned_cell_ids]
    torch.testing.assert_close(command.command["x"], expected[:, 0:1])
    torch.testing.assert_close(command.command["yaw"], expected[:, 1:2])


def test_curriculum_accumulates_reward_by_assigned_joint_cell(
    runtime_context,
    model_context,
):
    manager = make_curriculum(runtime_context, model_context)
    term = cast(CommandReward, manager.get_term("command_reward"))
    buffer = term.buffers["motion"]
    buffer.assigned_cell_ids.copy_(torch.tensor([0, 1, 0, 2]))

    manager.update(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    torch.testing.assert_close(
        buffer.reward_sum,
        torch.tensor([4.0, 2.0, 4.0]),
    )
    torch.testing.assert_close(
        buffer.sample_count,
        torch.tensor([2, 1, 1]),
    )


def test_curriculum_rejects_constraint_across_groups(
    runtime_context,
    model_context,
):
    terms = make_command_terms()
    terms["yaw"]["params"]["group"] = "rotation"

    with pytest.raises(ValueError, match="crosses curriculum groups"):
        CurriculumManager(
            num_envs=4,
            context=runtime_context,
            model_context=model_context,
            terms={"command_reward": {}},
            manager_configs={
                "command_manager_config": {
                    "terms": terms,
                    "constraints": {
                        "yaw": {
                            "operator": "<=",
                            "expression": "{x}",
                        },
                    },
                },
            },
        )


def test_curriculum_command_noise_scale_uses_bin_width(
    runtime_context,
    model_context,
):
    num_envs = 20_000

    class FixedSampler:
        buffers = {}

        def resample(self, space_names, env_ids=None):
            pass

        def get_command(self, space_name, dimension, env_ids=None):
            selected_count = num_envs if env_ids is None else env_ids.numel()
            return torch.zeros(selected_count, 1)

    term = CurriculumSampleOnReset(
        num_envs=num_envs,
        context=runtime_context,
        model_context=model_context,
        curriculum_sampler=FixedSampler(),
        term_name="x",
        group="motion",
        min_value=-1.0,
        max_value=1.0,
        num_bins=3,
        noise_scale=0.5,
    )
    torch.manual_seed(0)

    term.reset()

    noise = term.command - term.command_center
    assert abs(noise.mean().item()) < 0.02
    assert abs(noise.std().item() - 0.5) < 0.02
