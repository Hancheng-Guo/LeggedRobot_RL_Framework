from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from envs.tasks.managers.curriculum import CurriculumManager
from envs.tasks.managers.curriculum.terms.command import BaseLpacCommandReward
from envs import VectorEnv
from envs.tasks import BaseTaskLogic
from envs.tasks.locomotion import LocomotionTaskLogic
from rl.algorithms import OnPolicyAlgorithm
from runners import OnPolicyRunner, RestoreMode
from runners.callbacks import BaseCallback, StageCallback
from runners.types import TrainStopReason


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


@pytest.fixture
def make_runner(runtime_context):
    def make(manager=None):
        runner = OnPolicyRunner(runtime_context)
        runner.current_iteration = 4
        runner.max_iterations = 5
        runner.rollout_length = 1
        obs = torch.zeros(3, 1)

        def reset():
            if manager is not None:
                manager.reset()
            return obs

        def step(action):
            if manager is not None:
                update(manager, [10.0, 20.0, 30.0])
                manager.reset()
            return (obs, obs, torch.ones(3), torch.ones(3, dtype=torch.bool),
                    torch.zeros(3, dtype=torch.bool), {})

        environment = VectorEnv(runtime_context)
        if manager is None:
            environment.task = BaseTaskLogic(runtime_context)
        else:
            task = LocomotionTaskLogic(runtime_context)
            task.curriculum_manager = manager
            environment.task = task
        environment.num_envs = 3
        environment.reset = reset
        environment.step = step
        runner.environment = environment
        runner.algorithm = cast(OnPolicyAlgorithm, SimpleNamespace(
            checkpoint_state_dict=lambda: {},
            load_checkpoint_state_dict=lambda state, load_optimizer: None,
            save_module_artifacts=lambda path: [],
            set_train_mode=lambda: None, set_eval_mode=lambda: None,
            act=lambda obs, deterministic: SimpleNamespace(action=torch.zeros(3, 1)),
            reset_policy_state=lambda env_ids: None,
        ))
        return runner
    return make


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


@pytest.mark.parametrize("mode", list(RestoreMode))
def test_runner_saves_and_loads_curriculum(make_manager, make_runner, tmp_path, mode):
    source = make_manager()
    warm_up(source)
    # Keep both a completed partial batch and unfinished episodes at save time.
    update(source, [4.0, 5.0, 6.0])
    source.reset(torch.tensor([0]))
    update(source, [7.0, 8.0, 9.0])
    expected = source.checkpoint_state_dict()
    path = make_runner(source).save(tmp_path / "runner.pt")
    payload = torch.load(path, weights_only=False)
    assert_same_checkpoint(expected, payload["runner"]["environment"]["task"]["curriculum"])

    target = make_manager()
    runner = make_runner(target)
    runner.prepare_checkpoint_load(path)
    runner.load(mode=mode)
    assert runner.current_iteration == 4
    assert_same_checkpoint(expected, target.checkpoint_state_dict())
    term = get_term(target)
    assert not term.episode_step_count.any()
    for name, buffer in term.buffers.items():
        assert (buffer.assigned_cell_ids == -1).all()
        assert not term.episode_reward_sum[name].any()

    # train() starts with an environment reset even when the budget is exhausted.
    runner.train()
    assert_same_checkpoint(expected, target.checkpoint_state_dict())
    for buffer in term.buffers.values():
        assert (buffer.assigned_cell_ids >= 0).all()


def test_runner_rejects_incompatible_curriculum(make_manager, make_runner, tmp_path):
    source = make_manager()
    warm_up(source)
    path = make_runner(source).save(tmp_path / "runner.pt")
    target = make_manager(normalize_by_episode_steps=True)
    before = target.checkpoint_state_dict()
    runner = make_runner(target)
    runner.prepare_checkpoint_load(path)
    with pytest.raises(ValueError, match="configuration does not match"):
        runner.load()
    assert_same_checkpoint(before, target.checkpoint_state_dict())


def test_runner_rejects_curriculum_state_without_manager(make_manager, make_runner, tmp_path):
    path = make_runner(make_manager()).save(tmp_path / "runner.pt")
    runner = make_runner()
    runner.prepare_checkpoint_load(path)
    with pytest.raises(ValueError, match="task does not support"):
        runner.load()


@pytest.mark.parametrize("has_key", [False, True])
def test_runner_loads_without_curriculum_state(make_manager, make_runner, tmp_path, has_key):
    state = make_runner().checkpoint_state_dict()
    assert state["environment"] == {"task": {}}
    state["environment"] = None
    if not has_key:
        state.pop("environment")
    path = tmp_path / "runner.pt"
    torch.save({"runner": state}, path)
    target = make_manager()
    runner = make_runner(target)
    before = target.checkpoint_state_dict()
    runner.prepare_checkpoint_load(path)
    runner.load()
    assert_same_checkpoint(before, target.checkpoint_state_dict())


def test_evaluation_does_not_replace_training_curriculum_checkpoint(make_manager, make_runner, tmp_path):
    manager = make_manager()
    warm_up(manager)
    runner = make_runner(manager)
    expected = manager.checkpoint_state_dict()
    runner.test(num_episodes=6)
    assert get_term(manager).cell_statistics["motion"].revision.item() > 1
    path = runner.save(tmp_path / "after_evaluation.pt")
    saved = torch.load(path, weights_only=False)["runner"]["environment"]["task"]["curriculum"]
    assert_same_checkpoint(expected, saved)
    # Repeated evaluations keep the original training snapshot.
    runner.test(num_episodes=6)
    assert_same_checkpoint(expected, runner.checkpoint_state_dict()["environment"]["task"]["curriculum"])
    runner.train()
    assert runner._checkpoint_environment_snapshot is None
    assert_same_checkpoint(expected, manager.checkpoint_state_dict())


def test_new_checkpoint_load_discards_evaluation_snapshot(make_manager, make_runner, tmp_path):
    source = make_manager()
    warm_up(source)
    path = make_runner(source).save(tmp_path / "runner.pt")
    target = make_manager()
    runner = make_runner(target)
    runner.test(num_episodes=3)
    assert runner._checkpoint_environment_snapshot is not None
    runner.prepare_checkpoint_load(path)
    runner.load()
    runner.train()
    assert_same_checkpoint(source.checkpoint_state_dict(), target.checkpoint_state_dict())


@pytest.mark.parametrize("layer", ["environment", "task"])
@pytest.mark.parametrize("state", [None, {"unknown": {}}])
def test_checkpoint_layers_reject_invalid_state(make_manager, make_runner, layer, state):
    runner = make_runner(make_manager())
    target = runner.environment if layer == "environment" else runner.environment.task
    with pytest.raises((TypeError, ValueError)):
        target.load_checkpoint_state_dict(state)


def test_task_without_persistent_state_uses_empty_checkpoint(runtime_context):
    task = BaseTaskLogic(runtime_context)
    assert task.checkpoint_state_dict() == {}
    task.load_checkpoint_state_dict({})
    with pytest.raises(TypeError, match="must be a mapping"):
        task.load_checkpoint_state_dict(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="does not support"):
        task.load_checkpoint_state_dict({"curriculum": {}})


def attach_stage(runner, manager, *, window=3):
    name = next(iter(manager.terms))
    callback = StageCallback(condition=[{
        "metric": f"curriculum/{name}/motion/reward_min",
        "method": "min", "operator": ">=", "threshold": 0.0,
        "window": window, "sample_on_update": True,
        "require_all_cells_valid": True,
    }])
    callback.set_runner(runner)
    runner.callbacks = [callback]
    callback._on_train_start()
    return callback


def test_runner_resumes_stage_window_and_revision_with_curriculum(make_manager, make_runner, tmp_path):
    source = make_manager()
    warm_up(source)
    runner = make_runner(source)
    callback = attach_stage(runner, source)
    callback._on_step_end(source.collect_metrics())
    path = runner.save(tmp_path / "stage.pt")

    target = make_manager()
    resumed_runner = make_runner(target)
    resumed = attach_stage(resumed_runner, target)
    resumed_runner.prepare_checkpoint_load(path)
    resumed_runner.load()
    resumed_runner.train()
    assert list(resumed._conditions[0].values) == [2.0]
    assert resumed._conditions[0].last_revision == 1
    resumed._on_step_end(target.collect_metrics())
    assert list(resumed._conditions[0].values) == [2.0]
    for _ in range(2):
        warm_up(target)
        resumed._on_step_end(target.collect_metrics())
    assert resumed.stop_training
    assert resumed._conditions[0].last_revision == 3


@pytest.mark.parametrize("resave", [False, True])
def test_runner_discards_stage_curriculum_history_without_environment_state(make_manager, make_runner, tmp_path, resave):
    source = make_manager()
    warm_up(source)
    runner = make_runner(source)
    callback = attach_stage(runner, source, window=1)
    callback._on_step_end(source.collect_metrics())
    state = runner.checkpoint_state_dict()
    state.pop("environment")
    path = tmp_path / "no_environment.pt"
    torch.save({"runner": state}, path)
    target = make_manager()
    resumed_runner = make_runner(target)
    resumed = attach_stage(resumed_runner, target, window=1)
    resumed_runner.prepare_checkpoint_load(path)
    resumed_runner.load()
    if resave:
        path = resumed_runner.save(tmp_path / "resaved.pt")
        assert torch.load(path, weights_only=False)["runner"]["environment"] is None
        resumed_runner.prepare_checkpoint_load(path)
        resumed_runner.load()
    resumed_runner.train()
    assert not resumed.stop_training
    assert not resumed._conditions[0].values
    assert resumed._conditions[0].last_revision is None


def test_runner_restored_completed_stage_does_not_collect_more_steps(make_manager, make_runner, tmp_path):
    source = make_manager()
    warm_up(source)
    runner = make_runner(source)
    callback = attach_stage(runner, source, window=1)
    callback._on_step_end(source.collect_metrics())
    path = runner.save(tmp_path / "completed.pt")
    target = make_manager()
    resumed_runner = make_runner(target)
    attach_stage(resumed_runner, target, window=1)
    resumed_runner.max_iterations = 6
    resumed_runner.prepare_checkpoint_load(path)
    resumed_runner.load()
    result = resumed_runner.train()
    assert result.reason is TrainStopReason.STAGE_COMPLETED
    assert result.current_iteration == 4
    assert_same_checkpoint(source.checkpoint_state_dict(), target.checkpoint_state_dict())


@pytest.mark.parametrize("mode", [RestoreMode.EVALUATE, RestoreMode.ADVANCE_STAGE])
def test_runner_does_not_restore_stage_window_outside_resume(make_manager, make_runner, tmp_path, mode):
    source = make_manager()
    warm_up(source)
    runner = make_runner(source)
    callback = attach_stage(runner, source)
    callback._on_step_end(source.collect_metrics())
    path = runner.save(tmp_path / "stage.pt")
    target = make_manager()
    resumed_runner = make_runner(target)
    resumed = attach_stage(resumed_runner, target)
    resumed_runner.prepare_checkpoint_load(path)
    resumed_runner.load(mode=mode)
    resumed_runner.train()
    assert not resumed._conditions[0].values
    assert resumed._conditions[0].last_revision is None


def make_scripted_training(make_runner, manager, *, normalize, invalidate_step=None):
    runner = make_runner(manager)
    runner.current_iteration = -1
    runner.max_iterations = 24
    name = next(iter(manager.terms))
    stage = StageCallback(condition=[
        {
            "metric": f"curriculum/{name}/motion/reward_min",
            "method": "min", "operator": ">=", "threshold": 0.0,
            "window": 3, "sample_on_update": True, "require_all_cells_valid": True,
        },
        {
            "metric": f"curriculum/{name}/posture/reward_min",
            "method": "min", "operator": ">=",
            "threshold": 1.25 if normalize else 2.5,
            "window": 2, "sample_on_update": True, "require_all_cells_valid": True,
        },
        {
            "metric": "rollout/test_score", "method": "mean",
            "operator": ">=", "threshold": 1.0, "window": 2,
        },
    ])
    stage.set_runner(runner)
    trace = []

    class TrainingTrace(BaseCallback):
        def __init__(self):
            self.runner = runner

        def _on_iteration_end(self, *args, **kwargs):
            term = get_term(manager)
            trace.append({
                "iteration": runner.current_iteration,
                "curriculum": manager.checkpoint_state_dict(),
                "stage": stage.checkpoint_state_dict(),
                "metrics": manager.collect_metrics(),
                "assignments": {
                    group: buffer.assigned_cell_ids.clone()
                    for group, buffer in term.buffers.items()
                },
                "episode_steps": term.episode_step_count.clone(),
            })
            return True

    runner.callbacks = [stage, TrainingTrace()]
    runner.algorithm.process_transition = lambda **kwargs: None
    runner.algorithm.compute_returns = lambda **kwargs: None
    runner.algorithm.update = lambda: {"rollout/test_score": 1.0}

    def step(action):
        index = runner.current_iteration
        term = get_term(manager)
        if index % 2 == 0:
            # Prescribed episode inputs isolate persistence from simulator/RNG state.
            term.buffers["motion"].assigned_cell_ids.copy_(torch.tensor([0, 1, 2]))
            term.buffers["posture"].assigned_cell_ids.copy_(torch.tensor([2, 1, 0]))
        strength = index // 4 + 1
        update(manager, [strength, 2 * strength, 3 * strength])
        done = index % 2 == 1
        if done:
            torch.manual_seed(1000 + index)
            manager.reset()
        if index == invalidate_step:
            # Evaluation batches are independent of LPAC's sampling windows.
            # Inject a failed evaluation without changing the sampling rewards.
            term._record_cell_statistics(
                "motion", torch.tensor([0, 0]), torch.tensor([float("nan"), float("nan")]),
            )
        obs = torch.full((3, 1), float(index + 1))
        return (
            obs, obs, torch.ones(3), torch.full((3,), done, dtype=torch.bool),
            torch.zeros(3, dtype=torch.bool), manager.collect_metrics(),
        )

    runner.environment.step = step
    return runner, stage, trace


def assert_same_training_trace(expected, actual):
    assert len(expected) == len(actual)
    for before, after in zip(expected, actual):
        assert before["iteration"] == after["iteration"]
        assert before["stage"] == after["stage"]
        assert_same_checkpoint(before["curriculum"], after["curriculum"])
        for key in ("metrics", "assignments"):
            assert before[key].keys() == after[key].keys()
            for name in before[key]:
                torch.testing.assert_close(before[key][name], after[key][name], equal_nan=True)
        torch.testing.assert_close(before["episode_steps"], after["episode_steps"])


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("checkpoint_step", [2, 6, 14])
def test_continuous_and_resumed_training_match_every_iteration(
    make_manager, make_runner, tmp_path, normalize, checkpoint_step,
):
    continuous_manager = make_manager(normalize_by_episode_steps=normalize)
    continuous, _, expected_trace = make_scripted_training(
        make_runner, continuous_manager, normalize=normalize,
    )
    expected_result = continuous.train()
    assert expected_result.reason is TrainStopReason.STAGE_COMPLETED
    assert expected_result.current_iteration == 15
    assert len(expected_trace) == 16

    interrupted_manager = make_manager(normalize_by_episode_steps=normalize)
    interrupted, _, prefix = make_scripted_training(
        make_runner, interrupted_manager, normalize=normalize,
    )
    interrupted.max_iterations = checkpoint_step
    assert interrupted.train().reason is TrainStopReason.MAX_ITERATIONS_REACHED
    term = get_term(interrupted_manager)
    assert not term.episode_step_count.any()
    assert all((stats.sample_count == 1).all() for stats in term.cell_statistics.values())
    path = interrupted.save(tmp_path / "interrupted.pt")

    resumed_manager = make_manager(normalize_by_episode_steps=normalize)
    resumed, _, suffix = make_scripted_training(
        make_runner, resumed_manager, normalize=normalize,
    )
    resumed.prepare_checkpoint_load(path)
    resumed.load()
    assert resumed.train() == expected_result
    assert_same_training_trace(expected_trace, prefix + suffix)
    assert_same_checkpoint(continuous_manager.checkpoint_state_dict(), resumed_manager.checkpoint_state_dict())


@pytest.mark.parametrize("checkpoint_step", [6, 8])
def test_invalid_cell_evaluation_recovers_identically_after_resume(
    make_manager, make_runner, tmp_path, checkpoint_step,
):
    continuous_manager = make_manager()
    continuous, _, expected_trace = make_scripted_training(
        make_runner, continuous_manager, normalize=False, invalidate_step=5,
    )
    expected_result = continuous.train()
    assert expected_result.reason is TrainStopReason.STAGE_COMPLETED

    interrupted_manager = make_manager()
    interrupted, stage, prefix = make_scripted_training(
        make_runner, interrupted_manager, normalize=False, invalidate_step=5,
    )
    interrupted.max_iterations = checkpoint_step
    interrupted.train()
    assert not get_term(interrupted_manager).cell_statistics["motion"].valid[0]
    assert not stage._conditions[0].values
    assert stage._conditions[1].values
    path = interrupted.save(tmp_path / "invalid_evaluation.pt")

    resumed_manager = make_manager()
    resumed, _, suffix = make_scripted_training(
        make_runner, resumed_manager, normalize=False, invalidate_step=5,
    )
    resumed.prepare_checkpoint_load(path)
    resumed.load()
    assert resumed.train() == expected_result
    assert_same_training_trace(expected_trace, prefix + suffix)
    assert get_term(resumed_manager).cell_statistics["motion"].valid.all()
