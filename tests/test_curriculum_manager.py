import torch
import pytest
from typing import cast

from envs.tasks.managers.command.base import CommandManager
# from envs.tasks.managers.command.terms.curriculum import LrpcSampleOnReset
from envs.tasks.managers.curriculum.base import CurriculumManager
from envs.tasks.managers.curriculum.terms.command import (
    LpacCommandReward,
    # LrpcCommandReward,
)


LRPC_DISABLED = pytest.mark.skip(
    reason="LRPC command curriculum is disabled."
)


def make_command_terms():
    return {
        "x": {
            "type": "LrpcSampleOnReset",
            "params": {
                "min_value": -1.0,
                "max_value": 1.0,
                "num_cell": 3,
                "group": "motion",
                "noise_scale": 0.0,
            },
        },
        "yaw": {
            "type": "LrpcSampleOnReset",
            "params": {
                "min_value": -2.0,
                "max_value": 2.0,
                "num_cell": 2,
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
            "lrpc_command_reward": {
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


def make_lpac_curriculum(runtime_context, model_context, num_envs=3):
    terms = {
        "x": {
            "type": "LpacSampleOnReset",
            "params": {
                "min_value": -1.0,
                "max_value": 1.0,
                "num_cell": 3,
                "group": "motion",
                "noise_scale": 0.0,
            },
        },
    }
    return CurriculumManager(
        num_envs=num_envs,
        context=runtime_context,
        model_context=model_context,
        terms={
            "lpac_command_reward": {
                "temperature": 1.0,
                "exploration": 0.0,
                "min_coverage": 1.0,
                "min_samples_per_cell": 1,
            },
        },
        manager_configs={
            "command_manager_config": {"terms": terms},
        },
    )


@LRPC_DISABLED
def test_curriculum_buffer_represents_joint_command_space(
    runtime_context,
    model_context,
):
    manager = make_curriculum(runtime_context, model_context)
    term = cast(
        LrpcCommandReward,  # pyright: ignore[reportUndefinedVariable]
        manager.get_term("lrpc_command_reward"),
    )
    buffer = term.buffers["motion"]

    assert buffer.dimension_names == ("x", "yaw")
    assert buffer.cell_starts.shape == (3, 2)
    assert buffer.reward_sum.shape == (3,)
    assert buffer.assigned_cell_ids.shape == (4,)
    x_id = buffer.dimension_names.index("x")
    yaw_id = buffer.dimension_names.index("yaw")
    assert torch.all(
        buffer.cell_starts[:, yaw_id]
        <= buffer.cell_starts[:, x_id]
    )
    torch.testing.assert_close(
        buffer.cell_starts,
        torch.tensor([
            [-1.0, -2.0],
            [0.0, -2.0],
            [1.0, -2.0],
        ]),
    )


@LRPC_DISABLED
def test_lower_reward_cells_have_higher_sampling_probability(
    runtime_context,
    model_context,
):
    manager = make_curriculum(runtime_context, model_context)
    term = cast(
        LrpcCommandReward,  # pyright: ignore[reportUndefinedVariable]
        manager.get_term("lrpc_command_reward"),
    )
    buffer = term.buffers["motion"]
    buffer.sample_count.fill_(1)
    buffer.reward_sum.copy_(
        torch.tensor([-2.0, -1.0, 0.0])
    )

    probabilities = term._get_probabilities("motion")

    assert torch.all(probabilities[:-1] > probabilities[1:])


def test_lpac_uses_first_window_as_zero_progress_baseline(
    runtime_context,
    model_context,
):
    manager = make_lpac_curriculum(runtime_context, model_context)
    term = cast(
        LpacCommandReward,
        manager.get_term("lpac_command_reward"),
    )
    buffer = term.buffers["motion"]
    buffer.assigned_cell_ids.copy_(torch.tensor([0, 1, 2]))

    manager.update(torch.tensor([1.0, 2.0, 3.0]))
    manager.reset(torch.tensor([0, 1, 2]))

    torch.testing.assert_close(
        term.last_reward_mean["motion"],
        torch.tensor([1.0, 2.0, 3.0]),
    )
    torch.testing.assert_close(
        term.learning_progress["motion"],
        torch.tensor([1.0, 2.0, 3.0]),
    )


def test_lpac_prioritizes_positive_progress_after_second_window(
    runtime_context,
    model_context,
):
    manager = make_lpac_curriculum(runtime_context, model_context)
    term = cast(
        LpacCommandReward,
        manager.get_term("lpac_command_reward"),
    )
    buffer = term.buffers["motion"]

    buffer.assigned_cell_ids.copy_(torch.tensor([0, 1, 2]))
    manager.update(torch.tensor([1.0, 2.0, 3.0]))
    manager.reset(torch.tensor([0, 1, 2]))

    buffer.assigned_cell_ids.copy_(torch.tensor([0, 1, 2]))
    manager.update(torch.tensor([2.0, 1.0, 3.0]))
    manager.reset(torch.tensor([0, 1, 2]))

    torch.testing.assert_close(
        term.learning_progress["motion"],
        torch.tensor([1.0, -1.0, 0.0]),
    )
    probabilities = term.probabilities["motion"]
    assert probabilities[0] > probabilities[2] > probabilities[1]


def test_lpac_waits_when_any_cell_is_below_sample_threshold(
    runtime_context,
    model_context,
):
    manager = make_lpac_curriculum(runtime_context, model_context)
    term = cast(
        LpacCommandReward,
        manager.get_term("lpac_command_reward"),
    )
    term.min_samples_per_cell = 2
    buffer = term.buffers["motion"]
    buffer.sample_count.copy_(torch.tensor([2, 2, 1]))
    buffer.reward_sum.copy_(torch.tensor([4.0, 6.0, 7.0]))

    manager.reset()

    torch.testing.assert_close(
        buffer.sample_count,
        torch.tensor([2, 2, 1]),
    )
    torch.testing.assert_close(
        buffer.reward_sum,
        torch.tensor([4.0, 6.0, 7.0]),
    )
    torch.testing.assert_close(
        term.last_reward_mean["motion"],
        torch.zeros(3),
    )


@LRPC_DISABLED
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

    term = cast(
        LrpcCommandReward,  # pyright: ignore[reportUndefinedVariable]
        curriculum.get_term("lrpc_command_reward"),
    )
    buffer = term.buffers["motion"]
    expected = buffer.cell_starts[buffer.assigned_cell_ids]
    torch.testing.assert_close(command.command["x"], expected[:, 0:1])
    torch.testing.assert_close(command.command["yaw"], expected[:, 1:2])


@LRPC_DISABLED
def test_curriculum_accumulates_reward_by_assigned_joint_cell(
    runtime_context,
    model_context,
):
    manager = make_curriculum(runtime_context, model_context)
    term = cast(
        LrpcCommandReward,  # pyright: ignore[reportUndefinedVariable]
        manager.get_term("lrpc_command_reward"),
    )
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


@LRPC_DISABLED
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
            terms={"lrpc_command_reward": {}},
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


@LRPC_DISABLED
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

    term = LrpcSampleOnReset(  # pyright: ignore[reportUndefinedVariable]
        num_envs=num_envs,
        context=runtime_context,
        model_context=model_context,
        curriculum_sampler=FixedSampler(),
        term_name="x",
        group="motion",
        min_value=-1.0,
        max_value=1.0,
        num_cell=3,
        noise_scale=0.5,
    )
    torch.manual_seed(0)

    term.reset()

    noise = term.command - term.command_center
    assert abs(noise.mean().item()) < 0.02
    assert abs(noise.std().item() - 0.5) < 0.02


@LRPC_DISABLED
def test_curriculum_supports_multidimensional_command_term(
    runtime_context,
    model_context,
):
    terms = {
        "foot_phase": {
            "type": "LrpcSampleOnReset",
            "params": {
                "dim": 2,
                "min_value": -1.0,
                "max_value": 1.0,
                "num_cell": 2,
                "group": "phase",
                "noise_scale": 0.0,
            },
        },
    }
    curriculum = CurriculumManager(
        num_envs=32,
        context=runtime_context,
        model_context=model_context,
        terms={"lrpc_command_reward": {}},
        manager_configs={
            "command_manager_config": {"terms": terms},
        },
    )
    command = CommandManager(
        num_envs=32,
        context=runtime_context,
        model_context=model_context,
        curriculum_manager=curriculum,
        terms=terms,
    )

    curriculum.reset()
    command.reset()

    term = cast(
        LrpcCommandReward,  # pyright: ignore[reportUndefinedVariable]
        curriculum.get_term("lrpc_command_reward"),
    )
    buffer = term.buffers["phase"]
    expected = buffer.cell_starts[buffer.assigned_cell_ids]

    assert buffer.dimension_names == ("foot_phase[0]", "foot_phase[1]")
    assert buffer.term_slices == {"foot_phase": slice(0, 2)}
    assert buffer.cell_starts.shape == (4, 2)
    assert command.command["foot_phase"].shape == (32, 2)
    torch.testing.assert_close(command.command["foot_phase"], expected)
    assert torch.any(expected[:, 0] != expected[:, 1])


@LRPC_DISABLED
def test_multidimensional_curriculum_noise_is_independent(
    runtime_context,
    model_context,
):
    num_envs = 10_000

    class FixedSampler:
        buffers = {}

        def resample(self, space_names, env_ids=None):
            pass

        def get_command(self, space_name, dimension, env_ids=None):
            selected_count = num_envs if env_ids is None else env_ids.numel()
            return torch.zeros(selected_count, 4)

    term = LrpcSampleOnReset(  # pyright: ignore[reportUndefinedVariable]
        num_envs=num_envs,
        context=runtime_context,
        model_context=model_context,
        curriculum_sampler=FixedSampler(),
        term_name="foot_phase",
        group="phase",
        dim=4,
        min_value=-1.0,
        max_value=1.0,
        num_cell=3,
        noise_scale=0.5,
    )
    torch.manual_seed(0)

    term.reset()

    noise = term.command - term.command_center
    assert noise.shape == (num_envs, 4)
    assert abs(noise.std().item() - 0.5) < 0.02
    assert not torch.equal(noise[:, 0], noise[:, 1])


@pytest.mark.parametrize(
    ("operator", "expression", "expected"),
    [
        ("<", "1.0", [-1.0, 0.0]),
        (">", "-1.0", [0.0, 1.0]),
        ("!=", "0.0", [-1.0, 1.0]),
    ],
)
@LRPC_DISABLED
def test_curriculum_filters_strict_and_not_equal_constraints(
    runtime_context,
    model_context,
    operator,
    expression,
    expected,
):
    terms = {
        "x": {
            "type": "LrpcSampleOnReset",
            "params": {
                "min_value": -1.0,
                "max_value": 1.0,
                "num_cell": 3,
                "group": "motion",
                "noise_scale": 0.0,
            },
        },
    }
    manager = CurriculumManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        terms={"lrpc_command_reward": {}},
        manager_configs={
            "command_manager_config": {
                "terms": terms,
                "constraints": {
                    "x": {
                        "operator": operator,
                        "expression": expression,
                    },
                },
            },
        },
    )

    term = cast(
        LrpcCommandReward,  # pyright: ignore[reportUndefinedVariable]
        manager.get_term("lrpc_command_reward"),
    )

    torch.testing.assert_close(
        term.buffers["motion"].cell_starts.squeeze(-1),
        torch.tensor(expected, dtype=runtime_context.dtype),
    )
