from typing import cast

import torch

from envs.tasks.managers.curriculum.base import CurriculumManager
from envs.tasks.managers.curriculum.terms.command import (
    FastLpacCommandReward,
    LpacCommandReward,
)


def make_manager(
    runtime_context,
    model_context,
    *,
    fast: bool,
    symmetric: bool = False,
) -> CurriculumManager:
    command_type = (
        "SymmetricFastLpacSampleOnReset"
        if fast and symmetric
        else "FastLpacSampleOnReset"
        if fast
        else "SymmetricLpacSampleOnReset"
        if symmetric
        else "LpacSampleOnReset"
    )
    curriculum_name = (
        "fast_lpac_command_reward" if fast else "lpac_command_reward"
    )
    curriculum_config = {
        "min_samples_per_cell": 1,
        "temperature": 1.0,
        "exploration": 0.0,
    }
    if fast:
        curriculum_config["min_coverage"] = 2.0 / 3.0

    return CurriculumManager(
        num_envs=3,
        context=runtime_context,
        model_context=model_context,
        terms={curriculum_name: curriculum_config},
        manager_configs={
            "command_manager_config": {
                "terms": {
                    "x": {
                        "type": command_type,
                        "params": {
                            "group": "motion",
                            "min_value": 0.0 if symmetric else -1.0,
                            "max_value": 3.0 if symmetric else 2.0,
                            "num_cell": 3,
                        },
                    },
                },
            },
        },
    )


def test_lpac_discovers_derived_command_terms(
    runtime_context,
    model_context,
) -> None:
    manager = make_manager(
        runtime_context,
        model_context,
        fast=False,
        symmetric=True,
    )

    term = cast(
        LpacCommandReward,
        manager.get_term("lpac_command_reward"),
    )
    assert term.buffers["motion"].term_slices == {"x": slice(0, 1)}


def test_fast_lpac_discovers_derived_command_terms(
    runtime_context,
    model_context,
) -> None:
    manager = make_manager(
        runtime_context,
        model_context,
        fast=True,
        symmetric=True,
    )

    term = cast(
        FastLpacCommandReward,
        manager.get_term("fast_lpac_command_reward"),
    )
    assert term.buffers["motion"].term_slices == {"x": slice(0, 1)}


def test_lpac_uses_zero_as_first_reward_baseline(
    runtime_context,
    model_context,
) -> None:
    manager = make_manager(runtime_context, model_context, fast=False)
    term = cast(LpacCommandReward, manager.get_term("lpac_command_reward"))
    buffer = term.buffers["motion"]

    assert torch.count_nonzero(term.last_reward_mean["motion"]) == 0
    buffer.sample_count.copy_(torch.tensor([1, 1, 0]))
    buffer.reward_sum.copy_(torch.tensor([1.0, 2.0, 0.0]))
    term._update_space("motion")

    assert torch.count_nonzero(term.last_reward_mean["motion"]) == 0
    torch.testing.assert_close(buffer.sample_count, torch.tensor([1, 1, 0]))

    buffer.sample_count[2] = 1
    buffer.reward_sum[2] = 3.0
    term._update_space("motion")

    torch.testing.assert_close(
        term.last_reward_mean["motion"],
        torch.tensor([1.0, 2.0, 3.0]),
    )
    torch.testing.assert_close(
        term.learning_progress["motion"],
        torch.tensor([1.0, 2.0, 3.0]),
    )
    assert torch.count_nonzero(buffer.sample_count) == 0


def test_lpac_updates_every_cell_in_each_complete_window(
    runtime_context,
    model_context,
) -> None:
    manager = make_manager(runtime_context, model_context, fast=False)
    term = cast(LpacCommandReward, manager.get_term("lpac_command_reward"))
    buffer = term.buffers["motion"]

    buffer.sample_count.fill_(1)
    buffer.reward_sum.copy_(torch.tensor([1.0, 2.0, 3.0]))
    term._update_space("motion")
    buffer.sample_count.fill_(1)
    buffer.reward_sum.copy_(torch.tensor([2.0, 1.0, 3.0]))
    term._update_space("motion")

    torch.testing.assert_close(
        term.learning_progress["motion"],
        torch.tensor([1.0, -1.0, 0.0]),
    )
    probabilities = term.probabilities["motion"]
    assert probabilities[0] > probabilities[2] > probabilities[1]


def test_fast_lpac_uses_ready_mean_for_missing_first_progress(
    runtime_context,
    model_context,
) -> None:
    manager = make_manager(runtime_context, model_context, fast=True)
    term = cast(
        FastLpacCommandReward,
        manager.get_term("fast_lpac_command_reward"),
    )
    buffer = term.buffers["motion"]
    buffer.sample_count.copy_(torch.tensor([1, 1, 0]))
    buffer.reward_sum.copy_(torch.tensor([2.0, 4.0, 0.0]))

    term._update_space("motion")

    torch.testing.assert_close(
        term.last_reward_mean["motion"],
        torch.tensor([2.0, 4.0, 3.0]),
    )
    torch.testing.assert_close(
        term.learning_progress["motion"],
        torch.tensor([2.0, 4.0, 3.0]),
    )
    torch.testing.assert_close(buffer.sample_count, torch.zeros(3, dtype=torch.long))


def test_fast_lpac_updates_ready_cells_independently_after_first_update(
    runtime_context,
    model_context,
) -> None:
    manager = make_manager(runtime_context, model_context, fast=True)
    term = cast(
        FastLpacCommandReward,
        manager.get_term("fast_lpac_command_reward"),
    )
    buffer = term.buffers["motion"]
    buffer.sample_count.copy_(torch.tensor([1, 1, 0]))
    buffer.reward_sum.copy_(torch.tensor([2.0, 4.0, 0.0]))
    term._update_space("motion")

    buffer.sample_count[2] = 1
    buffer.reward_sum[2] = 5.0
    term._update_space("motion")

    torch.testing.assert_close(
        term.last_reward_mean["motion"],
        torch.tensor([2.0, 4.0, 5.0]),
    )
    torch.testing.assert_close(
        term.learning_progress["motion"],
        torch.tensor([2.0, 4.0, 2.0]),
    )
    assert term.probabilities["motion"][1] > term.probabilities["motion"][[0, 2]].max()
    assert buffer.sample_count[2] == 0
