import torch
import pytest
from typing import Any
from typing import cast
from pathlib import Path

from envs.tasks.managers.curriculum import CurriculumManager
from envs.tasks.managers.curriculum.terms.command import (
    FastLpacCommandReward,
    LpacCommandReward,
)
from utils import load_yaml


def make_manager(
    runtime_context,
    model_context,
    *,
    fast: bool,
    symmetric: bool = False,
    normalize_by_episode_steps: bool = False,
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
    curriculum_config: dict[str, Any] = {
        "min_samples_per_cell": 1,
        "temperature": 1.0,
        "exploration": 0.0,
        "reward_sources": {"motion": [{"reward/action_diff_l2": 1.0}]},
    }
    if fast:
        curriculum_config["min_coverage"] = 2.0 / 3.0
    if normalize_by_episode_steps:
        curriculum_config["normalize_by_episode_steps"] = normalize_by_episode_steps

    return CurriculumManager(
        num_envs=3,
        context=runtime_context,
        model_context=model_context,
        terms={curriculum_name: curriculum_config},
        manager_configs={
            "reward_manager_config": {"terms": {"action_diff_l2": {}}},
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


@pytest.fixture(params=["lpac_command_reward", "fast_lpac_command_reward"])
def reward_source_config(request) -> dict[str, Any]:
    command_type = (
        "FastLpacSampleOnReset"
        if request.param == "fast_lpac_command_reward"
        else "LpacSampleOnReset"
    )
    return {
        "terms": {
            request.param: {
                "reward_sources": {
                    "motion": ["reward/action_diff_l2", {"reward/joint_velocity_l2": -2}],
                    "posture": [{"reward/action_diff_l2": 3.0}, {"reward/joint_velocity_l2": 0}],
                },
            },
        },
        "manager_configs": {
            "command_manager_config": {
                "terms": {
                    name: {
                        "type": command_type,
                        "params": {
                            "group": group,
                            "min_value": -1.0,
                            "max_value": 1.0,
                            "num_cell": 2,
                        },
                    }
                    for name, group in (("x", "motion"), ("yaw", "posture"))
                },
            },
            "reward_manager_config": {
                "terms": {
                    "action_diff_l2": {"weight": 99.0},
                    "joint_velocity_l2": {"weight": 0.0},
                },
            },
        },
    }


def test_reward_sources_are_independent_group_coefficients(
    runtime_context, model_context, reward_source_config,
) -> None:
    manager = CurriculumManager(
        num_envs=2, context=runtime_context, model_context=model_context,
        **reward_source_config,
    )
    name, config = next(iter(reward_source_config["terms"].items()))
    term = cast(LpacCommandReward | FastLpacCommandReward, manager.get_term(name))
    assert term.reward_sources == {
        "motion": {"reward/action_diff_l2": None, "reward/joint_velocity_l2": -2.0},
        "posture": {"reward/action_diff_l2": 3.0, "reward/joint_velocity_l2": 0.0},
    }
    config["reward_sources"]["posture"][0]["reward/action_diff_l2"] = 100.0
    config["reward_sources"]["motion"].append("reward")
    config["reward_sources"].clear()
    assert term.reward_sources["motion"]["reward/action_diff_l2"] is None
    assert "reward" not in term.reward_sources["motion"]
    assert term.reward_sources["posture"]["reward/action_diff_l2"] == 3.0
    assert set(term.reward_sources) == {"motion", "posture"}


@pytest.mark.parametrize(
    ("sources", "error", "message"),
    [
        (None, TypeError, "reward_sources.*dict"),
        ([], TypeError, "reward_sources.*dict"),
        ({1: {}}, TypeError, "group names.*strings"),
        ({}, ValueError, "missing groups.*motion.*posture"),
        ({"wrong": {}}, ValueError, "unknown groups.*wrong"),
        ({"motion": None, "posture": []}, ValueError, "motion.*non-empty list"),
        ({"motion": {}, "posture": []}, ValueError, "motion.*non-empty list"),
        (
            {"motion": {"terms": {"action_diff_l2": 1}}, "posture": []},
            ValueError, "motion.*non-empty list",
        ),
        ({"motion": [], "posture": []}, ValueError, "motion.*non-empty list"),
        ({"motion": [None], "posture": []}, ValueError, "motion.*single-entry dicts"),
        ({"motion": [{}], "posture": []}, ValueError, "motion.*single-entry dicts"),
        (
            {"motion": [{"reward/action_diff_l2": 1, "reward/joint_velocity_l2": 1}], "posture": []},
            ValueError, "motion.*single-entry dicts",
        ),
        (
            {"motion": ["reward/missing_reward"], "posture": []},
            ValueError, "motion.*unknown reward source 'reward/missing_reward'",
        ),
        (
            {"motion": ["action_diff_l2"], "posture": []},
            ValueError, "motion.*unknown reward source 'action_diff_l2'",
        ),
        (
            {"motion": ["reward/"], "posture": []},
            ValueError, "motion.*unknown reward source",
        ),
    ],
)
def test_reward_sources_reject_invalid_structure(
    runtime_context, model_context, reward_source_config, sources, error, message,
) -> None:
    config = next(iter(reward_source_config["terms"].values()))
    config["reward_sources"] = sources
    with pytest.raises(error, match=message):
        CurriculumManager(
            num_envs=2, context=runtime_context, model_context=model_context,
            **reward_source_config,
        )


@pytest.mark.parametrize(
    ("terms", "error", "message"),
    [
        ([{"": 1}], ValueError, "motion.*non-empty strings"),
        ([{1: 1}], ValueError, "motion.*non-empty strings"),
        ([{"reward/action_diff_l2": True}], TypeError, "action_diff_l2.*numeric"),
        ([{"reward/action_diff_l2": "1.0"}], TypeError, "action_diff_l2.*numeric"),
        ([{"reward/action_diff_l2": None}], TypeError, "action_diff_l2.*numeric"),
        ([{"reward/action_diff_l2": float("nan")}], ValueError, "action_diff_l2.*finite"),
        ([{"reward/action_diff_l2": float("inf")}], ValueError, "action_diff_l2.*finite"),
        ([{"reward/action_diff_l2": -float("inf")}], ValueError, "action_diff_l2.*finite"),
        (
            [{"reward/action_diff_l2": 0}, {"reward/joint_velocity_l2": 0.0}],
            ValueError, "motion.*all be zero",
        ),
        ([{"reward": 1.0}], ValueError, "reward.*does not support an override"),
        ([{"reward": None}], ValueError, "reward.*does not support an override"),
        (
            ["reward/action_diff_l2", {"reward/action_diff_l2": 0.5}],
            ValueError, "duplicate reward source",
        ),
        (["reward", "reward"], ValueError, "duplicate reward source"),
    ],
)
def test_reward_sources_reject_invalid_coefficients(
    runtime_context, model_context, reward_source_config, terms, error, message,
) -> None:
    config = next(iter(reward_source_config["terms"].values()))
    config["reward_sources"]["motion"] = terms
    with pytest.raises(error, match=message):
        CurriculumManager(
            num_envs=2, context=runtime_context, model_context=model_context,
            **reward_source_config,
        )


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        (["reward"], {"reward": None}),
        (
            ["reward", {"reward/action_diff_l2": 0.5}],
            {"reward": None, "reward/action_diff_l2": 0.5},
        ),
        (
            ["reward/joint_velocity_l2", {"reward/action_diff_l2": 0.0}],
            {"reward/joint_velocity_l2": None, "reward/action_diff_l2": 0.0},
        ),
        ([{"reward/action_diff_l2": 1.0}], {"reward/action_diff_l2": 1.0}),
    ],
)
def test_reward_sources_preserve_total_inherited_and_override_modes(
    runtime_context, model_context, reward_source_config, entries, expected,
) -> None:
    name, config = next(iter(reward_source_config["terms"].items()))
    config["reward_sources"]["motion"] = entries
    manager = CurriculumManager(
        num_envs=2, context=runtime_context, model_context=model_context,
        **reward_source_config,
    )
    term = cast(LpacCommandReward | FastLpacCommandReward, manager.get_term(name))
    assert term.reward_sources["motion"] == expected


def test_reward_sources_are_required(
    runtime_context, model_context, reward_source_config,
) -> None:
    config = next(iter(reward_source_config["terms"].values()))
    del config["reward_sources"]
    with pytest.raises(TypeError, match="reward_sources"):
        CurriculumManager(
            num_envs=2, context=runtime_context, model_context=model_context,
            **reward_source_config,
        )


@pytest.mark.parametrize("reward_config", [None, {}, {"terms": None}, {"terms": []}])
def test_reward_sources_require_reward_terms_config(
    runtime_context, model_context, reward_source_config, reward_config,
) -> None:
    reward_source_config["manager_configs"]["reward_manager_config"] = reward_config
    with pytest.raises(TypeError, match="reward_manager_config|reward manager"):
        CurriculumManager(
            num_envs=2, context=runtime_context, model_context=model_context,
            **reward_source_config,
        )


def test_reward_sources_require_reward_manager_config(
    runtime_context, model_context, reward_source_config,
) -> None:
    del reward_source_config["manager_configs"]["reward_manager_config"]
    with pytest.raises(ValueError, match="requires 'reward_manager_config'"):
        CurriculumManager(
            num_envs=2, context=runtime_context, model_context=model_context,
            **reward_source_config,
        )


@pytest.mark.parametrize("task_name", [
    "locomotion_unitree_go1_velocity",
    "locomotion_unitree_go1_phase",
    "locomotion_anymal_c_phase",
])
def test_task_configs_define_valid_curriculum_reward_sources(
    runtime_context, model_context, task_name,
) -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "tasks" / f"{task_name}.yaml"
    config = load_yaml(path)
    manager = CurriculumManager(
        num_envs=2, context=runtime_context, model_context=model_context,
        terms=config["curriculum_manager_config"]["terms"],
        manager_configs=config,
    )
    term = cast(FastLpacCommandReward, manager.get_term("fast_lpac_command_reward"))
    assert set(term.reward_sources) == set(term.spaces) == {"velocity"}


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


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_lpac_reset_records_episode_scores_for_different_lengths(
    runtime_context, model_context, fast, normalize,
) -> None:
    manager = make_manager(
        runtime_context, model_context, fast=fast,
        normalize_by_episode_steps=normalize,
    )
    name = "fast_lpac_command_reward" if fast else "lpac_command_reward"
    term = cast(LpacCommandReward | FastLpacCommandReward, manager.get_term(name))
    assert term.normalize_by_episode_steps is normalize
    # Keep the cell statistics available instead of consuming an update window.
    term.min_samples_per_cell = 10
    buffer = term.buffers["motion"]
    # Initial reset has no collected steps, including in normalized mode.
    manager.reset()
    torch.testing.assert_close(buffer.reward_sum, torch.zeros(3))
    torch.testing.assert_close(buffer.sample_count, torch.zeros(3, dtype=torch.long))
    buffer.assigned_cell_ids.copy_(torch.tensor([0, 1, 2]))
    reward = torch.tensor([1.0, -2.0, 3.0])

    manager.update(reward)
    manager.reset(torch.tensor([0]))
    torch.testing.assert_close(buffer.reward_sum, torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(term.episode_reward_sum, torch.tensor([0.0, -2.0, 3.0]))
    torch.testing.assert_close(term.episode_step_count, torch.tensor([0, 1, 1]))

    manager.update(reward)
    manager.reset(torch.tensor([1]))
    manager.update(reward)
    manager.reset(torch.tensor([2]))

    expected = [1.0, -2.0, 3.0] if normalize else [1.0, -4.0, 9.0]
    torch.testing.assert_close(buffer.reward_sum, torch.tensor(expected))
    torch.testing.assert_close(buffer.sample_count, torch.tensor([1, 1, 1]))
    torch.testing.assert_close(term.episode_reward_sum, torch.tensor([2.0, -2.0, 0.0]))
    torch.testing.assert_close(term.episode_step_count, torch.tensor([2, 1, 0]))


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_lpac_full_reset_averages_episode_scores_per_cell(
    runtime_context, model_context, fast, normalize,
) -> None:
    manager = make_manager(
        runtime_context, model_context, fast=fast,
        normalize_by_episode_steps=normalize,
    )
    name = "fast_lpac_command_reward" if fast else "lpac_command_reward"
    term = cast(LpacCommandReward | FastLpacCommandReward, manager.get_term(name))
    term.min_samples_per_cell = 10
    buffer = term.buffers["motion"]
    buffer.assigned_cell_ids.copy_(torch.tensor([0, 0, 1]))

    manager.update(torch.tensor([1.0, 2.0, 3.0]))
    manager.update(torch.tensor([1.0, 2.0, 3.0]))
    manager.reset()

    expected_sum = torch.tensor([3.0, 3.0, 0.0] if normalize else [6.0, 6.0, 0.0])
    expected_mean = torch.tensor([1.5, 3.0, 0.0] if normalize else [3.0, 6.0, 0.0])
    torch.testing.assert_close(buffer.reward_sum, expected_sum)
    torch.testing.assert_close(buffer.sample_count, torch.tensor([2, 1, 0]))
    torch.testing.assert_close(term._calculate_reward_mean("motion"), expected_mean)
    torch.testing.assert_close(term.episode_reward_sum, torch.zeros(3))
    torch.testing.assert_close(term.episode_step_count, torch.zeros(3, dtype=torch.long))

    # Resetting again without a new step must not count another episode.
    manager.reset()
    torch.testing.assert_close(buffer.reward_sum, expected_sum)
    torch.testing.assert_close(buffer.sample_count, torch.tensor([2, 1, 0]))


@pytest.mark.parametrize("normalize", [None, 0, 1, "true", [], {}])
def test_lpac_rejects_non_boolean_episode_normalization(
    runtime_context, model_context, reward_source_config, normalize,
) -> None:
    config = next(iter(reward_source_config["terms"].values()))
    config["normalize_by_episode_steps"] = normalize
    with pytest.raises(TypeError, match="normalize_by_episode_steps.*bool"):
        CurriculumManager(
            num_envs=2, context=runtime_context, model_context=model_context,
            **reward_source_config,
        )


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
