from copy import deepcopy
from dataclasses import replace
from typing import cast

import pytest
import torch

from envs.tasks.managers.curriculum import CurriculumManager
from envs.tasks.managers.curriculum.terms.command import BaseLpacCommandReward


@pytest.fixture(params=[False, True], ids=["lpac", "fast_lpac"])
def make_manager(request, runtime_context, model_context):
    fast = request.param
    name = "fast_lpac_command_reward" if fast else "lpac_command_reward"
    command_type = "FastLpacSampleOnReset" if fast else "LpacSampleOnReset"

    def make(*, num_envs=3, context=runtime_context, reward_weight=2.0, **options):
        config = {
            "min_samples_per_cell": 2,
            "temperature": 1.0,
            "exploration": 0.2,
            "reward_sources": {
                "motion": ["reward"],
                "posture": [{"reward/action_diff_l2": 0.5}],
            },
        }
        config.update(options)
        return CurriculumManager(
            num_envs=num_envs, context=context, model_context=model_context,
            terms={name: config},
            manager_configs={
                "command_manager_config": {
                    "terms": {
                        key: {"type": command_type, "params": {
                            "group": group, "min_value": 0.0,
                            "max_value": 3.0, "num_cell": 3,
                        }}
                        for key, group in (("x", "motion"), ("yaw", "posture"))
                    },
                },
                "reward_manager_config": {"terms": {
                    "action_diff_l2": {"weight": reward_weight},
                }},
            },
        )

    return make


def get_term(manager):
    return cast(BaseLpacCommandReward, next(iter(manager.terms.values())))


def update(manager, values):
    reward = torch.tensor(values, dtype=manager.context.dtype, device=manager.context.device)
    manager.update(
        raw_term_rewards={"action_diff_l2": reward},
        term_rewards={"action_diff_l2": 2 * reward}, total_reward=2 * reward,
    )


def warm_up(manager):
    term = get_term(manager)
    for _ in range(2):
        for buffer in term.buffers.values():
            buffer.assigned_cell_ids.copy_(torch.tensor([0, 1, 2]))
        update(manager, [1.0, 2.0, 3.0])
        manager.reset()


def assert_same_checkpoint(expected, actual):
    assert expected.keys() == actual.keys()
    assert expected["terms"].keys() == actual["terms"].keys()
    for name, saved in expected["terms"].items():
        restored = actual["terms"][name]
        assert saved["type"] == restored["type"]
        assert saved["config"] == restored["config"]
        assert saved["groups"].keys() == restored["groups"].keys()
        for group, values in saved["groups"].items():
            current = restored["groups"][group]
            assert values["dimension_names"] == current["dimension_names"]
            assert values["term_slices"] == current["term_slices"]
            for key in ("cell_starts", "cell_interval"):
                torch.testing.assert_close(values[key], current[key], check_dtype=False)
            assert values["tensors"].keys() == current["tensors"].keys()
            for key, tensor in values["tensors"].items():
                torch.testing.assert_close(tensor, current["tensors"][key], equal_nan=True, check_dtype=False)


@pytest.mark.parametrize("normalize", [False, True])
def test_checkpoint_round_trip_preserves_batches_but_discards_active_episodes(make_manager, tmp_path, normalize):
    source = make_manager(normalize_by_episode_steps=normalize)
    warm_up(source)
    source_term = get_term(source)
    for buffer in source_term.buffers.values():
        buffer.assigned_cell_ids.copy_(torch.tensor([0, 1, 2]))
    update(source, [4.0, 5.0, 6.0])
    source.reset(torch.tensor([0]))
    update(source, [7.0, 8.0, 9.0])
    assert source_term.episode_step_count.tolist() == [1, 2, 2]

    state = source.checkpoint_state_dict()
    path = tmp_path / "curriculum.pt"
    torch.save(state, path)
    loaded = torch.load(path, weights_only=True)
    target = make_manager(normalize_by_episode_steps=normalize)
    warm_up(target)
    update(target, [100.0, 100.0, 100.0])
    target.load_checkpoint_state_dict(loaded)
    assert_same_checkpoint(state, target.checkpoint_state_dict())
    target_term = get_term(target)
    assert target_term.episode_step_count.tolist() == [0, 0, 0]
    for name, buffer in target_term.buffers.items():
        assert buffer.assigned_cell_ids.tolist() == [-1, -1, -1]
        assert torch.count_nonzero(target_term.episode_reward_sum[name]) == 0
        assert target_term.cell_statistics[name].sample_count[0] == 1
        assert target_term.cell_statistics[name].revision.item() == 1
    for name, value in source.collect_metrics().items():
        torch.testing.assert_close(value, target.collect_metrics()[name])

    # Fresh assignments use restored probabilities, without committing old episodes.
    torch.manual_seed(23)
    source_term.resample(set(source_term.buffers))
    torch.manual_seed(23)
    target.reset()
    for name, buffer in source_term.buffers.items():
        torch.testing.assert_close(buffer.assigned_cell_ids, target_term.buffers[name].assigned_cell_ids)
    assert_same_checkpoint(state, target.checkpoint_state_dict())


def test_initial_checkpoint_preserves_unknown_cell_means(make_manager):
    source = make_manager()
    target = make_manager()
    target.load_checkpoint_state_dict(source.checkpoint_state_dict())
    assert_same_checkpoint(source.checkpoint_state_dict(), target.checkpoint_state_dict())
    assert all(torch.isnan(stats.reward_mean).all() for stats in get_term(target).cell_statistics.values())


def test_checkpoint_is_independent_of_source_and_restored_tensors(make_manager):
    source = make_manager()
    warm_up(source)
    state = source.checkpoint_state_dict()
    target = make_manager()
    target.load_checkpoint_state_dict(state)
    saved = next(iter(state["terms"].values()))
    saved["groups"]["motion"]["tensors"]["cell_reward_mean"].add_(100)
    saved["config"]["reward_sources"]["motion"]["reward"] = 99
    assert_same_checkpoint(source.checkpoint_state_dict(), target.checkpoint_state_dict())
    assert get_term(source).reward_sources["motion"]["reward"] is None


def test_loading_allows_new_environment_count_and_float_dtype(make_manager, runtime_context):
    source = make_manager()
    warm_up(source)
    target = make_manager(num_envs=5, context=replace(runtime_context, dtype=torch.float64))
    target.load_checkpoint_state_dict(source.checkpoint_state_dict())
    assert_same_checkpoint(source.checkpoint_state_dict(), target.checkpoint_state_dict())
    assert get_term(target).episode_step_count.shape == (5,)
    assert get_term(target).last_reward_mean["motion"].dtype == torch.float64
    target.reset()
    assert all(buffer.assigned_cell_ids.shape == (5,) for buffer in get_term(target).buffers.values())


@pytest.mark.parametrize("options", [
    {"normalize_by_episode_steps": True}, {"temperature": 2.0},
    {"exploration": 0.5}, {"min_samples_per_cell": 3},
    {"reward_weight": 3.0},
    {"reward_sources": {"motion": [{"reward/action_diff_l2": 0.25}], "posture": ["reward"]}},
])
def test_changed_scoring_or_sampling_config_is_rejected(make_manager, options):
    source = make_manager()
    warm_up(source)
    target = make_manager(**options)
    before = target.checkpoint_state_dict()
    with pytest.raises(ValueError, match="configuration does not match"):
        target.load_checkpoint_state_dict(source.checkpoint_state_dict())
    assert_same_checkpoint(before, target.checkpoint_state_dict())


@pytest.mark.parametrize("corruption", [
    "term", "type", "group", "dimensions", "grid", "shape", "dtype",
    "negative_count", "probabilities", "valid_mean", "tensor_fields",
])
def test_invalid_checkpoint_does_not_partially_restore_groups(make_manager, corruption):
    source = make_manager()
    warm_up(source)
    state = deepcopy(source.checkpoint_state_dict())
    name = next(iter(state["terms"]))
    term_state = state["terms"][name]
    group = term_state["groups"]["posture"]
    tensors = group["tensors"]
    if corruption == "term":
        state["terms"]["missing"] = state["terms"].pop(name)
    elif corruption == "type":
        term_state["type"] = "OtherCurriculum"
    elif corruption == "group":
        term_state["groups"].pop("posture")
    elif corruption == "dimensions":
        group["dimension_names"] = ("unknown",)
    elif corruption == "grid":
        group["cell_starts"].add_(1)
    elif corruption == "shape":
        tensors["sample_count"] = torch.zeros(2, dtype=torch.long)
    elif corruption == "dtype":
        tensors["sample_count"] = tensors["sample_count"].float()
    elif corruption == "negative_count":
        tensors["statistics_sample_count"][0] = -1
    elif corruption == "probabilities":
        tensors["probabilities"].zero_()
    elif corruption == "valid_mean":
        tensors["cell_reward_mean"][0] = torch.nan
    else:
        tensors.pop("learning_progress")
    target = make_manager()
    update(target, [10.0, 10.0, 10.0])
    before = target.checkpoint_state_dict()
    pending = get_term(target).episode_reward_sum["motion"].clone()
    with pytest.raises(ValueError):
        target.load_checkpoint_state_dict(state)
    assert_same_checkpoint(before, target.checkpoint_state_dict())
    torch.testing.assert_close(get_term(target).episode_reward_sum["motion"], pending)


def test_completed_batches_continue_from_restored_pending_episodes(make_manager):
    source = make_manager()
    term = get_term(source)
    for buffer in term.buffers.values():
        buffer.assigned_cell_ids.copy_(torch.tensor([0, 1, 2]))
    update(source, [2.0, 4.0, 6.0])
    source.reset()
    target = make_manager()
    target.load_checkpoint_state_dict(source.checkpoint_state_dict())
    for manager in (source, target):
        for buffer in get_term(manager).buffers.values():
            buffer.assigned_cell_ids.copy_(torch.tensor([0, 1, 2]))
        update(manager, [4.0, 6.0, 8.0])
        manager.reset()
    assert_same_checkpoint(source.checkpoint_state_dict(), target.checkpoint_state_dict())
    torch.testing.assert_close(get_term(target).cell_statistics["motion"].reward_mean, torch.tensor([6.0, 10.0, 14.0]))


def test_manager_validates_every_term_before_restoring_any(runtime_context, model_context):
    def make():
        return CurriculumManager(
            num_envs=3, context=runtime_context, model_context=model_context,
            terms={
                name: {"min_samples_per_cell": 2, "reward_sources": {group: ["reward"]}}
                for name, group in (("lpac_command_reward", "motion"), ("fast_lpac_command_reward", "posture"))
            },
            manager_configs={
                "command_manager_config": {"terms": {
                    key: {"type": command_type, "params": {
                        "group": group, "min_value": 0.0,
                        "max_value": 3.0, "num_cell": 3,
                    }}
                    for key, command_type, group in (
                        ("x", "LpacSampleOnReset", "motion"),
                        ("yaw", "FastLpacSampleOnReset", "posture"),
                    )
                }},
                "reward_manager_config": {"terms": {"action_diff_l2": {"weight": 2.0}}},
            },
        )

    source = make()
    for _ in range(2):
        for raw_term in source.terms.values():
            term = cast(BaseLpacCommandReward, raw_term)
            for buffer in term.buffers.values():
                buffer.assigned_cell_ids.copy_(torch.tensor([0, 1, 2]))
        update(source, [1.0, 2.0, 3.0])
        source.reset()
    state = source.checkpoint_state_dict()
    state["terms"]["fast_lpac_command_reward"]["groups"]["posture"]["tensors"]["sample_count"] = torch.zeros(2, dtype=torch.long)
    target = make()
    before = target.checkpoint_state_dict()
    with pytest.raises(ValueError, match="tensor shape"):
        target.load_checkpoint_state_dict(state)
    assert_same_checkpoint(before, target.checkpoint_state_dict())
