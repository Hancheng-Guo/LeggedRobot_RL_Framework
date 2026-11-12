from curriculum_helpers import build_curriculum_manager
import torch
import pytest
from typing import Any
from typing import cast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from envs.simulators.utils import SimulatorState
from envs.tasks.locomotion import LocomotionTaskLogic
from envs.tasks.managers.command import CommandManager
from envs.tasks.managers.curriculum import CurriculumManager
from envs.tasks.managers.reward import RewardManager
from envs.tasks.utils import TaskContext, TaskStepResult
from runners.callbacks.stage import StageCallback
from envs.tasks.managers.curriculum.terms.reward import (
    FastLpacReward,
    LpacReward,
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
    command_type = "SymmetricCurriculum" if symmetric else "Curriculum"
    curriculum_name = (
        "fast_lpac_reward" if fast else "lpac_reward"
    )
    curriculum_config: dict[str, Any] = {
        "min_samples_per_cell": 1,
        "temperature": 1.0,
        "exploration": 0.0,
        "reward_sources": [{"reward/action_diff_l2": 1.0}],
    }
    if fast:
        curriculum_config["min_coverage"] = 2.0 / 3.0
    if normalize_by_episode_steps:
        curriculum_config["normalize_by_episode_steps"] = normalize_by_episode_steps

    return build_curriculum_manager(
        num_envs=3,
        context=runtime_context,
        model_context=model_context,
        terms={curriculum_name: {"type": curriculum_name, "params": curriculum_config}},
        manager_configs={
            "reward_manager_config": {"terms": {"action_diff_l2": {}}},
            "command_manager_config": {
                "terms": {
                    "x": {
                        "type": command_type,
                        "params": {
                            "curriculum": curriculum_name,

                            "min_value": 0.0 if symmetric else -1.0,
                            "max_value": 3.0 if symmetric else 2.0,
                            "num_cell": 3,
                        },
                    },
                },
            },
        },
    )


@pytest.fixture(params=["lpac_reward", "fast_lpac_reward"])
def reward_source_config(request) -> dict[str, Any]:
    command_type = "Curriculum"
    return {
        "terms": {
            "motion": {"type": request.param, "params": {
                "reward_sources": ["reward/action_diff_l2", {"reward/joint_velocity_l2": -2}],
            }},
            "posture": {"type": request.param, "params": {
                "reward_sources": [{"reward/action_diff_l2": 3.0}, {"reward/joint_velocity_l2": 0}],
            }},
        },
        "manager_configs": {
            "command_manager_config": {"terms": {
                name: {"type": command_type, "params": {
                    "curriculum": curriculum_name,
                    "min_value": -1.0, "max_value": 1.0, "num_cell": 2,
                }}
                for name, curriculum_name in (("x", "motion"), ("yaw", "posture"))
            }},
            "reward_manager_config": {"terms": {
                "action_diff_l2": {"weight": 99.0},
                "joint_velocity_l2": {"weight": 0.0},
            }},
        },
    }


def test_reward_sources_are_independent_instance_coefficients(
    runtime_context, model_context, reward_source_config,
) -> None:
    manager = build_curriculum_manager(
        num_envs=2, context=runtime_context, model_context=model_context,
        **reward_source_config,
    )
    name, term_config = next(iter(reward_source_config["terms"].items()))
    config = term_config["params"]
    term = cast(LpacReward | FastLpacReward, manager.get_term(name))
    assert term.reward_sources == {
        "reward/action_diff_l2": None, "reward/joint_velocity_l2": -2.0,
    }
    config["reward_sources"][0] = "reward"
    config["reward_sources"].clear()
    assert term.reward_sources["reward/action_diff_l2"] is None
    assert "reward" not in term.reward_sources
    other = cast(LpacReward | FastLpacReward, manager.get_term("posture"))
    assert other.reward_sources["reward/action_diff_l2"] == 3.0


@pytest.mark.parametrize(
    ("sources", "error", "message"),
    [
        (None, ValueError, "reward_sources.*non-empty list"),
        ({}, ValueError, "reward_sources.*non-empty list"),
        ([], ValueError, "reward_sources.*non-empty list"),
        ({"motion": ["reward"]}, ValueError, "reward_sources.*non-empty list"),
        ([None], ValueError, "single-entry dicts"),
        ([{}], ValueError, "single-entry dicts"),
        ([{"reward/action_diff_l2": 1, "reward/joint_velocity_l2": 1}], ValueError, "single-entry dicts"),
        (["reward/missing_reward"], ValueError, "unknown reward source"),
        (["action_diff_l2"], ValueError, "unknown reward source"),
        (["reward/"], ValueError, "unknown reward source"),
    ],
)
def test_reward_sources_reject_invalid_structure(
    runtime_context, model_context, reward_source_config, sources, error, message,
) -> None:
    config = next(iter(reward_source_config["terms"].values()))["params"]
    config["reward_sources"] = sources
    with pytest.raises(error, match=message):
        build_curriculum_manager(
            num_envs=2, context=runtime_context, model_context=model_context,
            **reward_source_config,
        )


@pytest.mark.parametrize(
    ("terms", "error", "message"),
    [
        ([{"": 1}], ValueError, "reward_sources.*non-empty strings"),
        ([{1: 1}], ValueError, "reward_sources.*non-empty strings"),
        ([{"reward/action_diff_l2": True}], TypeError, "action_diff_l2.*numeric"),
        ([{"reward/action_diff_l2": "1.0"}], TypeError, "action_diff_l2.*numeric"),
        ([{"reward/action_diff_l2": None}], TypeError, "action_diff_l2.*numeric"),
        ([{"reward/action_diff_l2": float("nan")}], ValueError, "action_diff_l2.*finite"),
        ([{"reward/action_diff_l2": float("inf")}], ValueError, "action_diff_l2.*finite"),
        ([{"reward/action_diff_l2": -float("inf")}], ValueError, "action_diff_l2.*finite"),
        (
            [{"reward/action_diff_l2": 0}, {"reward/joint_velocity_l2": 0.0}],
            ValueError, "reward_sources.*all be zero",
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
    config = next(iter(reward_source_config["terms"].values()))["params"]
    config["reward_sources"] = terms
    with pytest.raises(error, match=message):
        build_curriculum_manager(
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
    name, term_config = next(iter(reward_source_config["terms"].items()))
    config = term_config["params"]
    config["reward_sources"] = entries
    manager = build_curriculum_manager(
        num_envs=2, context=runtime_context, model_context=model_context,
        **reward_source_config,
    )
    term = cast(LpacReward | FastLpacReward, manager.get_term(name))
    assert term.reward_sources == expected


def test_reward_sources_are_required(
    runtime_context, model_context, reward_source_config,
) -> None:
    config = next(iter(reward_source_config["terms"].values()))["params"]
    del config["reward_sources"]
    with pytest.raises(TypeError, match="reward_sources"):
        build_curriculum_manager(
            num_envs=2, context=runtime_context, model_context=model_context,
            **reward_source_config,
        )


@pytest.mark.parametrize("reward_config", [None, {}, {"terms": None}, {"terms": []}])
def test_reward_sources_require_reward_terms_config(
    runtime_context, model_context, reward_source_config, reward_config,
) -> None:
    reward_source_config["manager_configs"]["reward_manager_config"] = reward_config
    with pytest.raises(TypeError, match="reward_manager_config|reward manager"):
        build_curriculum_manager(
            num_envs=2, context=runtime_context, model_context=model_context,
            **reward_source_config,
        )


def test_reward_sources_require_reward_manager_config(
    runtime_context, model_context, reward_source_config,
) -> None:
    del reward_source_config["manager_configs"]["reward_manager_config"]
    with pytest.raises(ValueError, match="requires 'reward_manager_config'"):
        build_curriculum_manager(
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
    manager = build_curriculum_manager(
        num_envs=2, context=runtime_context, model_context=model_context,
        terms=config["curriculum_manager_config"]["terms"],
        manager_configs=config,
    )
    term = cast(FastLpacReward, manager.get_term("locomotion"))
    assert term.reward_sources
    assert set(term.buffer.term_slices) >= {"lin_vel_x", "ang_vel_z"}


@pytest.mark.parametrize("step_dt", [0.01, 0.1])
@pytest.mark.parametrize(
    ("sources", "expected"),
    [
        (["reward/action_diff_l2"], [10.0, 8.0]),
        ([{"reward/action_diff_l2": 0.5}], [2.5, 2.0]),
        ([{"reward/joint_velocity_l2": -2.0}], [-20.0, -10.0]),
        (
            [{"reward/action_diff_l2": 0.0}, "reward/joint_velocity_l2"],
            [-30.0, -15.0],
        ),
        (["reward"], [-20.0, -7.0]),
        (["reward", {"reward/action_diff_l2": 0.5}], [-17.5, -5.0]),
    ],
)
def test_task_routes_real_reward_sources_without_dt_scaling(
    runtime_context, model_context, reward_source_config, monkeypatch,
    step_dt, sources, expected,
) -> None:
    name, term_config = next(iter(reward_source_config["terms"].items()))
    config = term_config["params"]
    config["min_samples_per_cell"] = 10
    config["reward_sources"] = sources
    reward_config = reward_source_config["manager_configs"]["reward_manager_config"]
    reward_config["terms"]["action_diff_l2"]["weight"] = 2.0
    reward_config["terms"]["joint_velocity_l2"]["weight"] = -3.0
    manager = build_curriculum_manager(
        num_envs=2, context=runtime_context, model_context=model_context,
        **reward_source_config,
    )
    term = cast(LpacReward | FastLpacReward, manager.get_term(name))
    for instance in manager.terms.values():
        buffer = cast(LpacReward | FastLpacReward, instance).buffer
        buffer.assigned_cell_ids.copy_(torch.tensor([0, 1]))

    task = LocomotionTaskLogic(runtime_context)
    task.curriculum_manager = manager
    task.reward_manager = RewardManager(
        num_envs=2, context=runtime_context, model_context=model_context,
        **reward_config,
    )
    command_update = Mock(return_value={"command/updated": True})
    task.command_manager = cast(CommandManager, SimpleNamespace(update=command_update))
    qvel = torch.zeros(2, 8)
    qvel[:, [0, 7]] = torch.tensor([[2.0, 4.0], [1.0, 3.0]])
    context = TaskContext(
        state=cast(SimulatorState, SimpleNamespace(qvel=qvel)),
        command={}, last_command={},
        action=torch.tensor([[1.0, 3.0], [2.0, 2.0]]),
        last_action=torch.zeros(2, 2),
        episode_step=torch.ones(2, dtype=torch.long), step_dt=step_dt,
    )
    compute_spies = []
    for reward_term in task.reward_manager.terms.values():
        compute = Mock(wraps=reward_term.compute)
        monkeypatch.setattr(reward_term, "compute", compute)
        compute_spies.append(compute)

    reward, reward_info = task.compute_reward(context)
    result = TaskStepResult(
        reward=reward * step_dt,
        terminated=torch.zeros(2, dtype=torch.bool),
        truncated=torch.zeros(2, dtype=torch.bool),
    )
    info = task.post_step(context, result)

    torch.testing.assert_close(reward, torch.tensor([-20.0, -7.0]))
    torch.testing.assert_close(reward_info["reward"], reward.mean())
    torch.testing.assert_close(term.episode_reward_sum, torch.tensor(expected))
    torch.testing.assert_close(cast(LpacReward | FastLpacReward, manager.get_term("posture")).episode_reward_sum, torch.tensor([15.0, 12.0]))
    assert info["command/updated"] is True
    command_update.assert_called_once_with(context)
    assert all(compute.call_count == 1 for compute in compute_spies)
    # Composition must not overwrite RewardManager's reusable inputs.
    torch.testing.assert_close(task.reward_manager.raw_term_rewards["action_diff_l2"], torch.tensor([5.0, 4.0]))
    torch.testing.assert_close(task.reward_manager.term_rewards["joint_velocity_l2"], torch.tensor([-30.0, -15.0]))

    manager.reset(torch.tensor([0]))
    manager.resample(torch.tensor([0]))
    torch.testing.assert_close(term.buffer.reward_sum, torch.tensor([expected[0], 0.0]))
    torch.testing.assert_close(cast(LpacReward | FastLpacReward, manager.get_term("posture")).buffer.reward_sum, torch.tensor([15.0, 0.0]))
    torch.testing.assert_close(term.episode_reward_sum, torch.tensor([0.0, expected[1]]))
    torch.testing.assert_close(cast(LpacReward | FastLpacReward, manager.get_term("posture")).episode_reward_sum, torch.tensor([0.0, 12.0]))


def test_total_reward_source_uses_supplied_vector_without_term_summation(
    runtime_context, model_context, reward_source_config,
) -> None:
    name, term_config = next(iter(reward_source_config["terms"].items()))
    config = term_config["params"]
    config["min_samples_per_cell"] = 10
    config["reward_sources"] = ["reward"]
    reward_source_config["terms"]["posture"]["params"]["reward_sources"] = ["reward"]
    manager = build_curriculum_manager(
        num_envs=2, context=runtime_context, model_context=model_context,
        **reward_source_config,
    )
    term = cast(LpacReward | FastLpacReward, manager.get_term(name))
    for instance in manager.terms.values():
        buffer = cast(LpacReward | FastLpacReward, instance).buffer
        buffer.assigned_cell_ids.copy_(torch.tensor([0, 1]))
    total_reward = torch.tensor([2.0, -5.0])
    # A total-only source needs no per-term data and must not rebuild the sum.
    manager.update(raw_term_rewards={}, term_rewards={}, total_reward=total_reward)
    manager.reset()
    manager.resample()

    for instance in manager.terms.values():
        buffer = cast(LpacReward | FastLpacReward, instance).buffer
        torch.testing.assert_close(buffer.reward_sum, total_reward)
    torch.testing.assert_close(total_reward, torch.tensor([2.0, -5.0]))


@pytest.mark.parametrize("normalize", [False, True])
def test_instance_rewards_settle_independently_and_clear_invalid_assignments(
    runtime_context, model_context, reward_source_config, normalize,
) -> None:
    name, term_config = next(iter(reward_source_config["terms"].items()))
    config = term_config["params"]
    config["min_samples_per_cell"] = 10
    config["normalize_by_episode_steps"] = normalize
    reward_source_config["terms"]["posture"]["params"]["normalize_by_episode_steps"] = normalize
    manager = build_curriculum_manager(
        num_envs=2, context=runtime_context, model_context=model_context,
        **reward_source_config,
    )
    term = cast(LpacReward | FastLpacReward, manager.get_term(name))
    term.buffer.assigned_cell_ids.copy_(torch.tensor([-1, 1]))
    cast(LpacReward | FastLpacReward, manager.get_term("posture")).buffer.assigned_cell_ids.copy_(torch.tensor([0, 1]))
    raw = {"action_diff_l2": torch.tensor([2.0, 4.0]), "joint_velocity_l2": torch.tensor([3.0, 5.0])}
    weighted = {"action_diff_l2": torch.tensor([20.0, 40.0]), "joint_velocity_l2": torch.zeros(2)}
    for _ in range(2):
        manager.update(
            raw_term_rewards=raw, term_rewards=weighted,
            total_reward=torch.tensor([20.0, 40.0]),
        )
    manager.reset(torch.tensor([0]))
    manager.resample(torch.tensor([0]))

    expected_posture = 6.0 if normalize else 12.0
    torch.testing.assert_close(term.buffer.sample_count, torch.tensor([0, 0]))
    torch.testing.assert_close(cast(LpacReward | FastLpacReward, manager.get_term("posture")).buffer.reward_sum, torch.tensor([expected_posture, 0.0]))
    torch.testing.assert_close(cast(LpacReward | FastLpacReward, manager.get_term("posture")).buffer.sample_count, torch.tensor([1, 0]))
    torch.testing.assert_close(term.episode_step_count, torch.tensor([0, 2]))
    torch.testing.assert_close(term.episode_reward_sum, torch.tensor([0.0, 60.0]))
    torch.testing.assert_close(cast(LpacReward | FastLpacReward, manager.get_term("posture")).episode_reward_sum, torch.tensor([0.0, 24.0]))

    manager.reset()
    manager.resample()
    expected_motion = 30.0 if normalize else 60.0
    torch.testing.assert_close(term.buffer.reward_sum, torch.tensor([0.0, expected_motion]))
    torch.testing.assert_close(cast(LpacReward | FastLpacReward, manager.get_term("posture")).buffer.reward_sum, torch.tensor([expected_posture, 2 * expected_posture]))
    assert all(not cast(LpacReward | FastLpacReward, instance).episode_reward_sum.any() for instance in manager.terms.values())
    assert torch.count_nonzero(term.episode_step_count) == 0


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
        LpacReward,
        manager.get_term("lpac_reward"),
    )
    assert term.buffer.term_slices == {"x": slice(0, 1)}


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
        FastLpacReward,
        manager.get_term("fast_lpac_reward"),
    )
    assert term.buffer.term_slices == {"x": slice(0, 1)}


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_lpac_reset_records_episode_scores_for_different_lengths(
    runtime_context, model_context, fast, normalize,
) -> None:
    manager = make_manager(
        runtime_context, model_context, fast=fast,
        normalize_by_episode_steps=normalize,
    )
    name = "fast_lpac_reward" if fast else "lpac_reward"
    term = cast(LpacReward | FastLpacReward, manager.get_term(name))
    assert term.normalize_by_episode_steps is normalize
    # Keep the cell statistics available instead of consuming an update window.
    term.min_samples_per_cell = 10
    buffer = term.buffer
    # Initial reset has no collected steps, including in normalized mode.
    manager.reset()
    manager.resample()
    torch.testing.assert_close(buffer.reward_sum, torch.zeros(3))
    torch.testing.assert_close(buffer.sample_count, torch.zeros(3, dtype=torch.long))
    buffer.assigned_cell_ids.copy_(torch.tensor([0, 1, 2]))
    reward = torch.tensor([1.0, -2.0, 3.0])

    manager.update(
        raw_term_rewards={"action_diff_l2": reward},
        term_rewards={"action_diff_l2": reward},
        total_reward=reward,
    )
    manager.reset(torch.tensor([0]))
    manager.resample(torch.tensor([0]))
    torch.testing.assert_close(buffer.reward_sum, torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(term.episode_reward_sum, torch.tensor([0.0, -2.0, 3.0]))
    torch.testing.assert_close(term.episode_step_count, torch.tensor([0, 1, 1]))

    manager.update(
        raw_term_rewards={"action_diff_l2": reward},
        term_rewards={"action_diff_l2": reward},
        total_reward=reward,
    )
    manager.reset(torch.tensor([1]))
    manager.resample(torch.tensor([1]))
    manager.update(
        raw_term_rewards={"action_diff_l2": reward},
        term_rewards={"action_diff_l2": reward},
        total_reward=reward,
    )
    manager.reset(torch.tensor([2]))
    manager.resample(torch.tensor([2]))

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
    name = "fast_lpac_reward" if fast else "lpac_reward"
    term = cast(LpacReward | FastLpacReward, manager.get_term(name))
    term.min_samples_per_cell = 10
    buffer = term.buffer
    buffer.assigned_cell_ids.copy_(torch.tensor([0, 0, 1]))

    manager.update(
        raw_term_rewards={"action_diff_l2": torch.tensor([1.0, 2.0, 3.0])},
        term_rewards={"action_diff_l2": torch.tensor([1.0, 2.0, 3.0])},
        total_reward=torch.tensor([1.0, 2.0, 3.0]),
    )
    manager.update(
        raw_term_rewards={"action_diff_l2": torch.tensor([1.0, 2.0, 3.0])},
        term_rewards={"action_diff_l2": torch.tensor([1.0, 2.0, 3.0])},
        total_reward=torch.tensor([1.0, 2.0, 3.0]),
    )
    manager.reset()
    manager.resample()

    expected_sum = torch.tensor([3.0, 3.0, 0.0] if normalize else [6.0, 6.0, 0.0])
    expected_mean = torch.tensor([1.5, 3.0, 0.0] if normalize else [3.0, 6.0, 0.0])
    torch.testing.assert_close(buffer.reward_sum, expected_sum)
    torch.testing.assert_close(buffer.sample_count, torch.tensor([2, 1, 0]))
    torch.testing.assert_close(term._calculate_reward_mean(), expected_mean)
    torch.testing.assert_close(term.episode_reward_sum, torch.zeros(3))
    torch.testing.assert_close(term.episode_step_count, torch.zeros(3, dtype=torch.long))

    # Resetting again without a new step must not count another episode.
    manager.reset()
    manager.resample()
    torch.testing.assert_close(buffer.reward_sum, expected_sum)
    torch.testing.assert_close(buffer.sample_count, torch.tensor([2, 1, 0]))


@pytest.mark.parametrize("normalize", [None, 0, 1, "true", [], {}])
def test_lpac_rejects_non_boolean_episode_normalization(
    runtime_context, model_context, reward_source_config, normalize,
) -> None:
    config = next(iter(reward_source_config["terms"].values()))["params"]
    config["normalize_by_episode_steps"] = normalize
    with pytest.raises(TypeError, match="normalize_by_episode_steps.*bool"):
        build_curriculum_manager(
            num_envs=2, context=runtime_context, model_context=model_context,
            **reward_source_config,
        )


@pytest.mark.parametrize("fast", [False, True])
def test_curriculum_sampling_metrics_are_collected_after_settlement_as_snapshots(
    runtime_context, model_context, fast,
) -> None:
    manager = make_manager(runtime_context, model_context, fast=fast)
    name = "fast_lpac_reward" if fast else "lpac_reward"
    term = cast(LpacReward | FastLpacReward, manager.get_term(name))
    prefix = f"curriculum/{name}/"
    initial = manager.collect_metrics()

    term.buffer.assigned_cell_ids.copy_(torch.tensor([0, 1, 2]))
    reward = torch.tensor([1.0, 2.0, 3.0])
    assert manager.update(
        raw_term_rewards={"action_diff_l2": reward},
        term_rewards={"action_diff_l2": reward}, total_reward=reward,
    ) == {}
    manager.reset()
    settled = manager.collect_metrics()
    torch.testing.assert_close(settled[prefix + "learning_progress"], reward)
    torch.testing.assert_close(settled[prefix + "probabilities"], term.probabilities)
    assert {prefix + "learning_progress", prefix + "probabilities"} <= manager.get_metric_names()
    assert not initial[prefix + "learning_progress"].any()

    # Later settlements must not rewrite previously collected progress.
    term.buffer.assigned_cell_ids.copy_(torch.tensor([0, 1, 2]))
    next_reward = torch.tensor([4.0, 1.0, 5.0])
    manager.update(
        raw_term_rewards={"action_diff_l2": next_reward},
        term_rewards={"action_diff_l2": next_reward}, total_reward=next_reward,
    )
    manager.reset()
    torch.testing.assert_close(settled[prefix + "learning_progress"], reward)

    # Consumers may modify their snapshot without changing the sampler.
    current = manager.collect_metrics()
    probabilities = term.probabilities.clone()
    current[prefix + "probabilities"].zero_()
    torch.testing.assert_close(term.probabilities, probabilities)
    manager.set_training_mode(False)
    torch.testing.assert_close(manager.collect_metrics()[prefix + "probabilities"], probabilities)


@pytest.mark.parametrize("fast", [False, True])
def test_cell_statistics_require_real_episode_samples_before_becoming_valid(
    runtime_context, model_context, fast,
) -> None:
    manager = make_manager(runtime_context, model_context, fast=fast)
    name = "fast_lpac_reward" if fast else "lpac_reward"
    term = cast(LpacReward | FastLpacReward, manager.get_term(name))
    term.min_samples_per_cell = 2
    prefix = f"curriculum/{name}/"
    initial = manager.collect_metrics()
    assert initial[prefix + "coverage"].item() == 0
    assert initial[prefix + "valid_cell_count"].item() == 0
    assert initial[prefix + "num_cells"].item() == 3
    assert initial[prefix + "all_cells_valid"].item() == 0
    assert prefix + "reward_mean" not in initial
    manager.reset()
    manager.resample()
    assert manager.collect_metrics()[prefix + "statistics_revision"].item() == 0

    buffer = term.buffer
    buffer.assigned_cell_ids[0] = 0
    for reward in (torch.tensor([2.0, 0.0, 0.0]), torch.tensor([4.0, 0.0, 0.0])):
        manager.update(
            raw_term_rewards={"action_diff_l2": reward},
            term_rewards={"action_diff_l2": reward}, total_reward=reward,
        )
        manager.reset(torch.tensor([0]))
        manager.resample(torch.tensor([0]))
        if reward[0] == 2.0:
            assert manager.collect_metrics()[prefix + "coverage"].item() == 0
        buffer.assigned_cell_ids[0] = 0

    info = manager.collect_metrics()
    assert info[prefix + "coverage"].item() == pytest.approx(1 / 3)
    assert info[prefix + "reward_mean"].item() == 3.0
    assert info[prefix + "reward_min"].item() == 3.0
    assert info[prefix + "reward_std"].item() == 0.0
    assert info[prefix + "statistics_revision"].item() == 1
    # Statistics are available even while LPAC is waiting for other cells.
    assert not torch.any(term.learning_progress)


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_cell_statistics_exclude_inferred_baselines_and_survive_window_clears(
    runtime_context, model_context, fast, normalize,
) -> None:
    manager = make_manager(
        runtime_context, model_context, fast=fast,
        normalize_by_episode_steps=normalize,
    )
    name = "fast_lpac_reward" if fast else "lpac_reward"
    term = cast(LpacReward | FastLpacReward, manager.get_term(name))
    prefix = f"curriculum/{name}/"
    buffer = term.buffer
    buffer.assigned_cell_ids.copy_(torch.tensor([0, 1, -1]))
    reward = torch.tensor([2.0, 4.0, 0.0])
    for _ in range(2):
        manager.update(
            raw_term_rewards={"action_diff_l2": reward},
            term_rewards={"action_diff_l2": reward}, total_reward=reward,
        )
    manager.reset()
    manager.resample()
    scale = 1.0 if normalize else 2.0
    first = manager.collect_metrics()
    assert first[prefix + "coverage"].item() == pytest.approx(2 / 3)
    assert first[prefix + "reward_mean"].item() == 3.0 * scale
    assert first[prefix + "reward_min"].item() == 2.0 * scale
    assert first[prefix + "reward_max"].item() == 4.0 * scale
    assert first[prefix + "reward_std"].item() == scale
    stats = term.cell_statistics
    assert stats.valid.tolist() == [True, True, False]
    assert torch.isnan(stats.reward_mean[2])
    if fast:
        assert torch.isfinite(term.last_reward_mean[2])
    manager.reset()
    manager.resample()
    assert manager.collect_metrics()[prefix + "statistics_revision"].item() == 1

    buffer.assigned_cell_ids[0] = 2
    reward = torch.tensor([9.0, 0.0, 0.0])
    manager.update(
        raw_term_rewards={"action_diff_l2": reward},
        term_rewards={"action_diff_l2": reward}, total_reward=reward,
    )
    manager.reset(torch.tensor([0]))
    manager.resample(torch.tensor([0]))
    second = manager.collect_metrics()
    assert second[prefix + "coverage"].item() == 1.0
    assert second[prefix + "all_cells_valid"].item() == 1.0
    assert second[prefix + "statistics_revision"].item() == 2
    assert second[prefix + "reward_mean"].item() == pytest.approx((6 * scale + 9) / 3)
    # Old info is a snapshot; sampling buffer resets do not erase cell scores.
    assert first[prefix + "coverage"].item() == pytest.approx(2 / 3)
    assert first[prefix + "statistics_revision"].item() == 1
    torch.testing.assert_close(stats.reward_mean, torch.tensor([2 * scale, 4 * scale, 9.0]))


def test_cell_statistics_use_equal_cell_weights_and_independent_instances(
    runtime_context, model_context, reward_source_config,
) -> None:
    name, term_config = next(iter(reward_source_config["terms"].items()))
    config = term_config["params"]
    config["min_samples_per_cell"] = 1
    reward_source_config["terms"]["posture"]["params"]["min_samples_per_cell"] = 1
    config["reward_sources"] = ["reward"]
    reward_source_config["terms"]["posture"]["params"]["reward_sources"] = ["reward"]
    manager = build_curriculum_manager(
        num_envs=3, context=runtime_context, model_context=model_context,
        **reward_source_config,
    )
    term = cast(LpacReward | FastLpacReward, manager.get_term(name))
    term.buffer.assigned_cell_ids.copy_(torch.tensor([0, 0, 1]))
    cast(LpacReward | FastLpacReward, manager.get_term("posture")).buffer.assigned_cell_ids.copy_(torch.tensor([0, 1, 1]))
    manager.update(raw_term_rewards={}, term_rewards={}, total_reward=torch.tensor([2.0, 4.0, 9.0]))
    manager.reset()
    manager.resample()
    info = manager.collect_metrics()
    assert info[f"curriculum/{name}/reward_mean"].item() == 6.0
    assert info[f"curriculum/posture/reward_mean"].item() == 4.25
    assert info[f"curriculum/{name}/coverage"].item() == 1.0


@pytest.mark.parametrize("fast", [False, True])
def test_stage_consumes_only_fresh_fully_covered_cell_statistics(
    runtime_context, model_context, fast,
) -> None:
    manager = make_manager(runtime_context, model_context, fast=fast)
    name = "fast_lpac_reward" if fast else "lpac_reward"
    term = cast(LpacReward | FastLpacReward, manager.get_term(name))
    prefix = f"curriculum/{name}/"
    callback = StageCallback(condition=[{
        "metric": prefix + "reward_min", "method": "min",
        "operator": ">=", "threshold": 3.0, "window": 2,
        "revision_metric": prefix + "statistics_revision",
        "validity_metric": prefix + "all_cells_valid",
    }])
    buffer = term.buffer
    buffer.assigned_cell_ids.copy_(torch.tensor([0, 1, 2]))
    reward = torch.full((3,), 3.0)
    manager.update(
        raw_term_rewards={"action_diff_l2": reward},
        term_rewards={"action_diff_l2": reward}, total_reward=reward,
    )
    manager.reset(torch.tensor([0]))
    manager.resample(torch.tensor([0]))
    callback._on_step_end(manager.collect_metrics())
    assert not callback._conditions[0].values
    manager.reset(torch.tensor([1, 2]))
    manager.resample(torch.tensor([1, 2]))
    info = manager.collect_metrics()
    assert info[prefix + "all_cells_valid"].item() == 1.0
    for _ in range(3):
        callback._on_step_end(info)
    assert list(callback._conditions[0].values) == [3.0]
    assert not callback.stop_training

    buffer.assigned_cell_ids.copy_(torch.tensor([0, 1, 2]))
    reward = torch.full((3,), 4.0)
    manager.update(
        raw_term_rewards={"action_diff_l2": reward},
        term_rewards={"action_diff_l2": reward}, total_reward=reward,
    )
    manager.reset()
    manager.resample()
    callback._on_step_end(manager.collect_metrics())
    assert callback.stop_training


@pytest.mark.parametrize("fast", [False, True])
def test_stage_waits_for_first_curriculum_evaluation(
    runtime_context, model_context, fast,
) -> None:
    manager = make_manager(runtime_context, model_context, fast=fast)
    name = "fast_lpac_reward" if fast else "lpac_reward"
    prefix = f"curriculum/{name}/"
    assert prefix + "reward_min" not in manager.collect_metrics()
    assert prefix + "reward_min" in manager.get_metric_names()
    callback = StageCallback(condition=[{
        "metric": prefix + "reward_min", "method": "min",
        "operator": ">=", "threshold": 2.0, "window": 2,
        "revision_metric": prefix + "statistics_revision", "validity_metric": prefix + "all_cells_valid",
    }])
    callback.set_runner(SimpleNamespace(environment=SimpleNamespace(
        task=SimpleNamespace(curriculum_manager=manager),
    )))
    callback._on_train_start()
    assert not callback._conditions[0].values
    callback._on_step_end(manager.collect_metrics())
    assert not callback.stop_training


def test_lpac_uses_zero_as_first_reward_baseline(
    runtime_context,
    model_context,
) -> None:
    manager = make_manager(runtime_context, model_context, fast=False)
    term = cast(LpacReward, manager.get_term("lpac_reward"))
    buffer = term.buffer

    assert torch.count_nonzero(term.last_reward_mean) == 0
    buffer.sample_count.copy_(torch.tensor([1, 1, 0]))
    buffer.reward_sum.copy_(torch.tensor([1.0, 2.0, 0.0]))
    term._update_space()

    assert torch.count_nonzero(term.last_reward_mean) == 0
    torch.testing.assert_close(buffer.sample_count, torch.tensor([1, 1, 0]))

    buffer.sample_count[2] = 1
    buffer.reward_sum[2] = 3.0
    term._update_space()

    torch.testing.assert_close(
        term.last_reward_mean,
        torch.tensor([1.0, 2.0, 3.0]),
    )
    torch.testing.assert_close(
        term.learning_progress,
        torch.tensor([1.0, 2.0, 3.0]),
    )
    assert torch.count_nonzero(buffer.sample_count) == 0


def test_lpac_updates_every_cell_in_each_complete_window(
    runtime_context,
    model_context,
) -> None:
    manager = make_manager(runtime_context, model_context, fast=False)
    term = cast(LpacReward, manager.get_term("lpac_reward"))
    buffer = term.buffer

    buffer.sample_count.fill_(1)
    buffer.reward_sum.copy_(torch.tensor([1.0, 2.0, 3.0]))
    term._update_space()
    buffer.sample_count.fill_(1)
    buffer.reward_sum.copy_(torch.tensor([2.0, 1.0, 3.0]))
    term._update_space()

    torch.testing.assert_close(
        term.learning_progress,
        torch.tensor([1.0, -1.0, 0.0]),
    )
    probabilities = term.probabilities
    assert probabilities[0] > probabilities[2] > probabilities[1]


def test_fast_lpac_uses_ready_mean_for_missing_first_progress(
    runtime_context,
    model_context,
) -> None:
    manager = make_manager(runtime_context, model_context, fast=True)
    term = cast(
        FastLpacReward,
        manager.get_term("fast_lpac_reward"),
    )
    buffer = term.buffer
    buffer.sample_count.copy_(torch.tensor([1, 1, 0]))
    buffer.reward_sum.copy_(torch.tensor([2.0, 4.0, 0.0]))

    term._update_space()

    torch.testing.assert_close(
        term.last_reward_mean,
        torch.tensor([2.0, 4.0, 3.0]),
    )
    torch.testing.assert_close(
        term.learning_progress,
        torch.tensor([2.0, 4.0, 3.0]),
    )
    torch.testing.assert_close(buffer.sample_count, torch.zeros(3, dtype=torch.long))


def test_fast_lpac_updates_ready_cells_independently_after_first_update(
    runtime_context,
    model_context,
) -> None:
    manager = make_manager(runtime_context, model_context, fast=True)
    term = cast(
        FastLpacReward,
        manager.get_term("fast_lpac_reward"),
    )
    buffer = term.buffer
    buffer.sample_count.copy_(torch.tensor([1, 1, 0]))
    buffer.reward_sum.copy_(torch.tensor([2.0, 4.0, 0.0]))
    term._update_space()

    buffer.sample_count[2] = 1
    buffer.reward_sum[2] = 5.0
    term._update_space()

    torch.testing.assert_close(
        term.last_reward_mean,
        torch.tensor([2.0, 4.0, 5.0]),
    )
    torch.testing.assert_close(
        term.learning_progress,
        torch.tensor([2.0, 4.0, 2.0]),
    )
    assert term.probabilities[1] > term.probabilities[[0, 2]].max()
    assert buffer.sample_count[2] == 0
