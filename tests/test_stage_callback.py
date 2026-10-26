import pytest
import torch
from pathlib import Path
from types import SimpleNamespace

import app.stage_manager as stage_manager_module
from app.stage_manager import StageManager, StageTrainResult
from runners.callbacks import BaseCallback, StageCallback
from runners.on_policy import OnPolicyRunner
from runners.types import TrainResult, TrainStopReason
from utils import Component, ComponentInfo, RuntimeContext


class OtherStoppingCallback(BaseCallback):
    def __init__(self) -> None:
        self.runner = None

    def _on_step_end(self, *args, **kwargs) -> bool:
        return False


def make_callback(
    method: str = "mean",
    metric: str = "reward/action_diff_l2",
    operator: str = ">=",
    threshold: float = 2.0,
    window: int = 3,
) -> StageCallback:
    return StageCallback(condition=[
        {
            "method": method,
            "metric": metric,
            "operator": operator,
            "threshold": threshold,
            "window": window,
        }
    ])


def test_stage_callback_uses_complete_sliding_window() -> None:
    callback = make_callback()
    callback._on_train_start()

    callback._on_step_end(
        info={"reward/action_diff_l2": torch.tensor(1.0)}
    )
    callback._on_step_end(
        info={"reward/action_diff_l2": torch.tensor(2.0)}
    )
    assert callback.stop_training is False

    callback._on_step_end(
        info={"reward/action_diff_l2": torch.tensor(3.0)}
    )
    assert callback.stop_training is True


def test_stage_callback_latches_satisfied_condition() -> None:
    callback = make_callback(window=2)
    callback._on_step_end(info={"reward/action_diff_l2": 3.0})
    callback._on_step_end(info={"reward/action_diff_l2": 3.0})
    assert callback.stop_training is True

    callback._on_step_end(info={"reward/action_diff_l2": 0.0})
    assert callback.stop_training is True


def test_stage_callback_reads_reward_term_from_info() -> None:
    callback = StageCallback(condition=[
        {
            "method": "mean",
            "metric": "reward/action_diff_l2",
            "operator": "<=",
            "threshold": 0.5,
            "window": 2,
        }
    ])

    callback._on_step_end(info={"reward/action_diff_l2": 0.4})
    callback._on_step_end(info={"reward/action_diff_l2": 0.6})

    assert callback.stop_training is True


def test_stage_callback_reads_exact_metric_name() -> None:
    callback = make_callback(metric="reward", window=2)

    callback._on_step_end(info={"reward": 1.0})
    callback._on_step_end(info={"reward": 3.0})

    assert callback.stop_training is True


def test_stage_callback_reduces_vector_metric_by_mean() -> None:
    callback = make_callback(
        metric="reward/track_linear_velocity_xy_l2_exp_and_logcosh",
        threshold=6.0,
        window=2,
    )
    callback.set_runner(SimpleNamespace(
        environment=SimpleNamespace(num_envs=2)
    ))

    callback._on_step_end(info={
        "reward/track_linear_velocity_xy_l2_exp_and_logcosh": torch.tensor([
            4.0,
            8.0,
        ])
    })
    callback._on_step_end(info={
        "reward/track_linear_velocity_xy_l2_exp_and_logcosh": torch.tensor([
            6.0,
            6.0,
        ])
    })

    assert callback.stop_training is True


def test_stage_callback_rejects_empty_tensor_metric() -> None:
    callback = make_callback(window=1)

    with pytest.raises(ValueError, match="must not be empty"):
        callback._on_step_end(info={
            "reward/action_diff_l2": torch.tensor([]),
        })


def test_stage_callback_rejects_wrong_vector_metric_size() -> None:
    callback = make_callback(window=1)
    callback.set_runner(SimpleNamespace(
        environment=SimpleNamespace(num_envs=2)
    ))

    with pytest.raises(ValueError, match=r"num_envs \(2\).*got 3"):
        callback._on_step_end(info={
            "reward/action_diff_l2": torch.tensor([1.0, 2.0, 3.0]),
        })


def test_stage_callback_reads_iteration_end_metric() -> None:
    callback = make_callback(metric="rollout/mean_len", window=2)

    callback._on_step_end(info={"reward": 100.0})
    assert callback.stop_training is False

    callback._on_iteration_end(info={"rollout/mean_len": 1.0})
    callback._on_iteration_end(info={"rollout/mean_len": 3.0})
    assert callback.stop_training is True


@pytest.mark.parametrize(
    ("method", "threshold", "expected"),
    [
        ("mean", 2.0, True),
        ("max", 3.0, True),
        ("min", 2.0, False),
    ],
)
def test_stage_callback_supports_aggregation_methods(
    method: str,
    threshold: float,
    expected: bool,
) -> None:
    callback = make_callback(
        method=method,
        threshold=threshold,
        window=3,
    )

    for value in (1.0, 2.0, 3.0):
        callback._on_step_end(info={"reward/action_diff_l2": value})

    assert callback.stop_training is expected


def test_stage_callback_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        make_callback(metric="")

    with pytest.raises(ValueError, match="Unsupported stage method"):
        make_callback(method="median")

    with pytest.raises(ValueError, match="positive integer"):
        make_callback(window=0)


def test_stage_callback_tracks_multiple_explicit_conditions() -> None:
    callback = make_callback(metric="rollout/loss", window=1)
    callback._conditions.extend(make_callback(metric="policy/loss", window=1)._conditions)
    callback._on_iteration_end({"rollout/loss": 3.0, "policy/loss": 1.0})
    assert not callback.stop_training
    assert list(callback._conditions[0].values) == [3.0]
    assert list(callback._conditions[1].values) == [1.0]
    callback._on_iteration_end({"policy/loss": 3.0})
    assert callback.stop_training


CELL_PREFIX = "curriculum/test/motion/"


def make_cell_callback(window=2, **overrides):
    config = {
        "metric": CELL_PREFIX + "reward_min",
        "sample_on_update": True,
        "require_all_cells_valid": True,
        "method": "min", "operator": ">=", "threshold": 2.0,
        "window": window,
    }
    config.update(overrides)
    callback = StageCallback(condition=[config])
    manager = SimpleNamespace(get_metric_names=lambda: {
        CELL_PREFIX + name for name in (
            "reward_min", "reward_max", "reward_mean", "reward_std",
            "statistics_revision", "all_cells_valid", "coverage",
        )
    })
    callback.set_runner(SimpleNamespace(
        environment=SimpleNamespace(task=SimpleNamespace(curriculum_manager=manager)),
    ))
    return callback


def cell_info(revision=1, value=3.0, valid=1):
    return {
        CELL_PREFIX + "statistics_revision": torch.tensor(revision),
        CELL_PREFIX + "reward_min": torch.tensor(value),
        CELL_PREFIX + "all_cells_valid": torch.tensor(valid),
    }


def test_stage_samples_each_revision_once_across_both_hooks():
    callback = make_cell_callback()
    for _ in range(5):
        callback._on_step_end(cell_info())
        callback._on_iteration_end(cell_info())
    assert list(callback._conditions[0].values) == [3.0]
    assert not callback.stop_training
    assert not callback._on_step_end(cell_info(revision=2))
    assert callback.stop_training


def test_stage_validity_loss_clears_window_even_without_primary_metric():
    callback = make_cell_callback()
    callback._on_step_end(cell_info())
    invalid = cell_info(revision=2, valid=0)
    del invalid[CELL_PREFIX + "reward_min"]
    callback._on_step_end(invalid)
    assert not callback._conditions[0].values
    # An invalid revision cannot later count as a fresh evaluation.
    callback._on_step_end(cell_info(revision=2))
    assert not callback._conditions[0].values
    callback._on_step_end(cell_info(revision=3))
    assert not callback.stop_training
    callback._on_step_end(cell_info(revision=4))
    assert callback.stop_training


def test_stage_invalid_gate_prevents_stale_condition_combining_with_other_metric():
    first = make_cell_callback(window=1)
    other = make_callback(metric="rollout/mean_len", window=1)
    first._conditions.extend(other._conditions)
    first._on_step_end(cell_info())
    assert not first.stop_training
    first._on_step_end(cell_info(valid=0))
    first._on_iteration_end({"rollout/mean_len": 100.0})
    assert not first.stop_training
    first._on_step_end(cell_info(revision=2))
    assert first.stop_training


def test_stage_ignores_unrelated_hooks_and_resets_on_revision_rollback():
    callback = make_cell_callback()
    callback._on_step_end(cell_info(revision=10))
    callback._on_iteration_end({"rollout/mean_len": 100.0})
    assert list(callback._conditions[0].values) == [3.0]
    callback._on_step_end(cell_info(revision=1))
    assert not callback.stop_training
    assert list(callback._conditions[0].values) == [3.0]
    callback._on_step_end(cell_info(revision=2))
    assert callback.stop_training


@pytest.mark.parametrize("key", ["statistics_revision", "all_cells_valid"])
def test_stage_missing_required_control_metric_raises(key):
    callback = make_cell_callback()
    callback._on_step_end(cell_info())
    incomplete = cell_info(revision=2)
    del incomplete[CELL_PREFIX + key]
    with pytest.raises(KeyError, match=CELL_PREFIX + key):
        callback._on_step_end(incomplete)
    assert list(callback._conditions[0].values) == [3.0]
    assert not callback.stop_training


@pytest.mark.parametrize("require_valid", [False, True])
def test_stage_missing_primary_metric_raises_only_after_validity_check(require_valid):
    callback = make_cell_callback(require_all_cells_valid=require_valid)
    callback._on_step_end(cell_info())
    incomplete = cell_info(revision=2)
    del incomplete[CELL_PREFIX + "reward_min"]
    if require_valid:
        with pytest.raises(KeyError, match="reward_min.*all_cells_valid being 1"):
            callback._on_step_end(incomplete)
    else:
        callback._on_step_end(incomplete)
        assert not callback._conditions[0].values
    assert not callback.stop_training


def test_stage_waits_for_first_cell_evaluation_without_validity_gate():
    callback = make_cell_callback(require_all_cells_valid=False)
    initial = cell_info(revision=0, valid=0)
    del initial[CELL_PREFIX + "reward_min"]
    callback._on_step_end(initial)
    assert not callback._conditions[0].values
    callback._on_step_end(cell_info(revision=1))
    assert list(callback._conditions[0].values) == [3.0]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_stage_nonfinite_value_clears_window(value):
    callback = make_cell_callback()
    callback._on_step_end(cell_info())
    callback._on_step_end(cell_info(revision=2, value=value))
    assert not callback._conditions[0].values
    assert not callback.stop_training


@pytest.mark.parametrize("revision", [-1.0, 1.5, float("nan"), float("inf"), -float("inf")])
def test_stage_invalid_revision_raises_without_changing_window(revision):
    callback = make_cell_callback()
    callback._on_step_end(cell_info())
    with pytest.raises(ValueError, match="statistics_revision.*finite non-negative integer"):
        callback._on_step_end(cell_info(revision=revision))
    assert list(callback._conditions[0].values) == [3.0]
    assert callback._conditions[0].last_revision == 1
    assert not callback.stop_training


@pytest.mark.parametrize("key", ["statistics_revision", "all_cells_valid"])
def test_stage_control_metrics_must_be_scalar(key):
    callback = make_cell_callback()
    callback.set_runner(SimpleNamespace(environment=SimpleNamespace(num_envs=2)))
    info = cell_info()
    info[CELL_PREFIX + key] = torch.tensor([1, 1])
    with pytest.raises(ValueError, match="control metric.*scalar"):
        callback._on_step_end(info)


@pytest.mark.parametrize("key", ["sample_on_update", "require_all_cells_valid"])
@pytest.mark.parametrize("value", [None, "", 1, "true"])
def test_stage_curriculum_options_must_be_boolean(key, value):
    with pytest.raises(TypeError, match=key + ".*bool"):
        make_cell_callback(**{key: value})


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_stage_threshold_must_be_finite(value):
    with pytest.raises(ValueError, match="threshold.*finite"):
        make_cell_callback(threshold=value)


def test_stage_new_train_run_resets_revision_and_window():
    callback = make_cell_callback()
    callback._on_step_end(cell_info())
    callback._on_train_start()
    callback._on_step_end(cell_info())
    assert list(callback._conditions[0].values) == [3.0]
    assert not callback.stop_training


def test_stage_restore_discards_gated_windows_but_keeps_ordinary_windows():
    callback = make_cell_callback(window=1)
    callback._conditions.extend(make_callback(window=1)._conditions)
    callback._on_step_end(cell_info() | {"reward/action_diff_l2": 3.0})
    assert callback.stop_training
    restored = make_cell_callback(window=1)
    restored._conditions.extend(make_callback(window=1)._conditions)
    restored.load_checkpoint_state_dict(callback.checkpoint_state_dict())
    assert not restored.stop_training
    assert not restored._conditions[0].values
    assert restored._conditions[0].last_revision is None
    assert list(restored._conditions[1].values) == [3.0]
    restored._on_step_end(cell_info())
    assert restored.stop_training


@pytest.mark.parametrize("sample,validity", [(True, True), (True, False), (False, True)])
def test_stage_restores_curriculum_window_with_environment(sample, validity):
    options = {"sample_on_update": sample, "require_all_cells_valid": validity}
    callback = make_cell_callback(window=3, **options)
    callback._on_step_end(cell_info(revision=7))
    restored = make_cell_callback(window=3, **options)
    restored.load_checkpoint_state_dict(
        callback.checkpoint_state_dict(), environment_restored=True,
    )
    condition = restored._conditions[0]
    assert list(condition.values) == [3.0]
    assert condition.last_revision == (7 if sample else None)
    restored._on_step_end(cell_info(revision=7))
    assert list(condition.values) == ([3.0] if sample else [3.0, 3.0])
    restored._on_step_end(cell_info(revision=8))
    assert restored.stop_training is (not sample)


def test_stage_restores_revision_even_when_invalidity_cleared_window():
    callback = make_cell_callback()
    callback._on_step_end(cell_info(revision=4))
    callback._on_step_end(cell_info(revision=5, valid=0))
    restored = make_cell_callback()
    restored.load_checkpoint_state_dict(
        callback.checkpoint_state_dict(), environment_restored=True,
    )
    assert not restored._conditions[0].values
    assert restored._conditions[0].last_revision == 5
    restored._on_step_end(cell_info(revision=5))
    assert not restored._conditions[0].values
    restored._on_step_end(cell_info(revision=6))
    assert list(restored._conditions[0].values) == [3.0]


@pytest.mark.parametrize("event", ["duplicate_invalid", "rollback", "nan"])
def test_stage_restored_window_still_checks_invalid_events(event):
    callback = make_cell_callback(window=3)
    callback._on_step_end(cell_info(revision=4))
    restored = make_cell_callback(window=3)
    restored.load_checkpoint_state_dict(
        callback.checkpoint_state_dict(), environment_restored=True,
    )
    if event == "duplicate_invalid":
        restored._on_step_end(cell_info(revision=4, valid=0))
        assert not restored._conditions[0].values
    elif event == "nan":
        restored._on_step_end(cell_info(revision=5, value=float("nan")))
        assert not restored._conditions[0].values
    else:
        restored._on_step_end(cell_info(revision=1, value=4.0))
        assert list(restored._conditions[0].values) == [4.0]


@pytest.mark.parametrize("revision", [-1, True, 1.5, float("nan"), "1", None])
def test_stage_rejects_invalid_checkpoint_revision(revision):
    callback = make_cell_callback()
    callback._on_step_end(cell_info())
    state = callback.checkpoint_state_dict()
    state["condition_revisions"] = [revision]
    with pytest.raises(ValueError, match="revision"):
        make_cell_callback().load_checkpoint_state_dict(state, environment_restored=True)


def test_stage_restores_completed_curriculum_condition():
    callback = make_cell_callback(window=1)
    callback._on_step_end(cell_info(revision=4))
    restored = make_cell_callback(window=1)
    restored.load_checkpoint_state_dict(
        callback.checkpoint_state_dict(), environment_restored=True,
    )
    assert restored.stop_training
    assert restored._conditions[0].last_revision == 4


@pytest.mark.parametrize("optional", ["sample_on_update", "require_all_cells_valid"])
def test_stage_control_options_can_be_used_independently(optional):
    config = {
        "metric": CELL_PREFIX + "reward_min", "method": "min",
        "operator": ">=", "threshold": 2.0, "window": 2,
        optional: True,
    }
    callback = StageCallback(condition=[config])
    info = cell_info()
    callback._on_step_end(info)
    callback._on_step_end(info)
    assert callback.stop_training is (optional == "require_all_cells_valid")
    if optional == "sample_on_update":
        callback._on_step_end(cell_info(revision=2))
        assert callback.stop_training


@pytest.mark.parametrize("metric", [
    "reward", "reward/action_diff_l2", "curriculum/test/reward_min",
    "curriculum//motion/reward_min", "curriculum/test//reward_min",
])
def test_stage_enabled_curriculum_options_require_explicit_group(metric):
    with pytest.raises(ValueError, match="curriculum options require"):
        make_cell_callback(metric=metric)


def test_stage_disabled_curriculum_options_allow_ordinary_metrics():
    callback = make_cell_callback(
        metric="reward", sample_on_update=False, require_all_cells_valid=False,
    )
    callback._on_step_end({"reward": 3.0})
    callback._on_step_end({"reward": 3.0})
    assert callback.stop_training


def test_stage_binds_controls_only_to_the_metric_group():
    callback = make_cell_callback()
    callback._on_step_end(cell_info())
    foreign = {
        "curriculum/test/other/statistics_revision": torch.tensor(2),
        "curriculum/test/other/all_cells_valid": torch.tensor(1),
    }
    with pytest.raises(KeyError, match=CELL_PREFIX + "statistics_revision"):
        callback._on_step_end({CELL_PREFIX + "reward_min": torch.tensor(3.0)} | foreign)
    assert list(callback._conditions[0].values) == [3.0]
    assert not callback.stop_training
    foreign["curriculum/test/other/all_cells_valid"] = torch.tensor(0)
    # Start a fresh window so this test can verify the other group's false gate
    # does not block the correctly bound, valid control metrics.
    callback._on_train_start()
    callback._on_step_end(cell_info(revision=2) | foreign)
    assert list(callback._conditions[0].values) == [3.0]
    callback._on_step_end(cell_info(revision=3) | foreign)
    assert callback.stop_training


@pytest.mark.parametrize("metric", [
    "*/loss", "reward/*", "reward/a?", "reward/[abc]",
    CELL_PREFIX + "reward_*", "curriculum/*/motion/reward_min",
])
def test_stage_rejects_wildcard_metric_names(metric):
    with pytest.raises(ValueError, match="exact metric name.*wildcards"):
        make_cell_callback(metric=metric)


def test_stage_keeps_separate_windows_for_explicit_curriculum_metrics():
    callback = make_cell_callback(window=2)
    callback._conditions.extend(make_cell_callback(metric=CELL_PREFIX + "reward_mean", window=2)._conditions)
    callback._on_train_start()
    first = cell_info() | {CELL_PREFIX + "reward_mean": torch.tensor(1.0)}
    callback._on_step_end(first)
    callback._on_step_end(first)
    assert all(len(condition.values) == 1 for condition in callback._conditions)
    second = cell_info(revision=2) | {CELL_PREFIX + "reward_mean": torch.tensor(3.0)}
    callback._on_step_end(second)
    assert not callback.stop_training
    callback._on_step_end({**second, CELL_PREFIX + "statistics_revision": torch.tensor(3)})
    assert callback.stop_training


@pytest.mark.parametrize("metric", [
    "curriculum/typo/motion/reward_min", "curriculum/test/typo/reward_min",
    CELL_PREFIX + "reward_mni", CELL_PREFIX + "missing_metric",
])
@pytest.mark.parametrize("enabled", [False, True])
def test_stage_rejects_unknown_curriculum_metrics_before_training(metric, enabled):
    callback = make_cell_callback(
        metric=metric, sample_on_update=enabled, require_all_cells_valid=enabled,
    )
    with pytest.raises(KeyError, match="not a supported metric"):
        callback._on_train_start()


def test_stage_requires_curriculum_manager_before_training():
    callback = make_cell_callback()
    callback.set_runner(SimpleNamespace(environment=SimpleNamespace(task=SimpleNamespace())))
    with pytest.raises(ValueError, match="curriculum manager"):
        callback._on_train_start()


def test_stage_restores_explicit_condition_windows_separately():
    def make_conditions():
        callback = make_callback(metric="policy/loss", window=2)
        callback._conditions.extend(make_callback(metric="rollout/loss", window=2)._conditions)
        return callback

    callback = make_conditions()
    callback._on_iteration_end({"policy/loss": 1.0, "rollout/loss": 3.0})
    restored = make_conditions()
    restored.load_checkpoint_state_dict(callback.checkpoint_state_dict())
    assert list(restored._conditions[0].values) == [1.0]
    assert list(restored._conditions[1].values) == [3.0]
    restored._on_iteration_end({"policy/loss": 3.0, "rollout/loss": 3.0})
    assert restored.stop_training


def test_stage_checkpoint_values_require_metric_names():
    callback = make_callback(window=3)
    with pytest.raises(TypeError, match="must map metric names"):
        callback.load_checkpoint_state_dict({"condition_values": [[1.0, 2.0]], "stop_training": False})


def test_runner_records_callback_that_requests_stop(
    runtime_context: RuntimeContext,
) -> None:
    runner = OnPolicyRunner(context=runtime_context)
    callback = make_callback(window=1)
    runner.callbacks = [callback]
    runner.stop_callback.clear()

    should_continue = runner._run_callbacks(
        "_on_step_end",
        info={"reward/action_diff_l2": 2.0},
    )

    assert should_continue is False
    assert runner.stop_callback == [callback]


def test_stage_manager_distinguishes_stop_callback(
    runtime_context: RuntimeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = StageManager.__new__(StageManager)
    manager.current_stage = 0
    manager.stage_detail = [{
        "test": {
            "max_iterations": 10,
            "transition": [{
                "method": "mean",
                "metric": "reward/action_diff_l2",
                "operator": ">=",
                "threshold": 2.0,
                "window": 1,
            }],
        }
    }]
    manager.runner = OnPolicyRunner(context=runtime_context)
    stage_callback = make_callback(window=1)
    component = Component(None, None, None, None, None, None)

    monkeypatch.setattr(
        manager,
        "_get_current_component",
        lambda: component,
    )
    monkeypatch.setattr(
        manager,
        "_build_runner",
        lambda **kwargs: None,
    )
    for reason, expected in (
        (TrainStopReason.STAGE_COMPLETED, StageTrainResult.STAGE_COMPLETED),
        (TrainStopReason.CALLBACK_STOPPED, StageTrainResult.STOPPED_BY_CALLBACK),
        (TrainStopReason.MAX_ITERATIONS_REACHED, StageTrainResult.MAX_ITERATIONS_REACHED),
    ):
        monkeypatch.setattr(manager.runner, "train", lambda reason=reason: TrainResult(reason, 0))
        assert manager._train_current() is expected


def test_stage_manager_applies_stage_max_iterations(
    runtime_context: RuntimeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = StageManager.__new__(StageManager)
    manager.current_stage = 0
    manager.stage_detail = [{
        "stand": {
            "max_iterations": 123,
            "transition": [{
                "method": "mean",
                "metric": "reward/action_diff_l2",
                "operator": ">=",
                "threshold": 2.0,
                "window": 1,
            }],
        }
    }]
    manager.runner = OnPolicyRunner(context=runtime_context)
    manager.runner.stop_callback = []
    component = Component(None, None, None, None, None, None)
    captured: dict[str, int] = {}

    monkeypatch.setattr(manager, "_get_current_component", lambda: component)
    monkeypatch.setattr(manager.runner, "train", lambda: TrainResult(TrainStopReason.MAX_ITERATIONS_REACHED, 0))

    def build_runner(**kwargs) -> None:
        captured["max_iterations"] = kwargs["max_iterations"]

    monkeypatch.setattr(manager, "_build_runner", build_runner)

    manager._train_current()

    assert captured["max_iterations"] == 123


def test_stage_manager_injects_stage_max_iterations(
    runtime_context: RuntimeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, context: object) -> None:
            captured["context"] = context

        def config_update(self, **kwargs) -> None:
            captured.update(kwargs)

    manager = StageManager.__new__(StageManager)
    manager.context = runtime_context
    component = Component(
        runner=ComponentInfo(
            type="fake",
            config=Path("runner.yaml"),
        ),
        algorithm=None,
        policy=None,
        environment=None,
        simulator=None,
        task=None,
    )

    monkeypatch.setitem(
        stage_manager_module.RUNNER_TYPE_MAP,
        "fake",
        FakeRunner,
    )
    monkeypatch.setattr(
        "app.stage_manager.load_yaml",
        lambda _: {"rollout_length": 64},
    )

    manager._build_runner(component=component, max_iterations=123)

    assert captured["max_iterations"] == 123
    assert captured["rollout_length"] == 64


def test_stage_manager_uses_last_stage_after_completion() -> None:
    manager = StageManager.__new__(StageManager)
    manager.stage_detail = [{"stage1": {}}, {"stage2": {}}]

    manager.current_stage = 1
    assert manager.continue_training is True
    assert manager._get_effective_stage() == 1

    manager.current_stage = 2
    assert manager.continue_training is False
    assert manager._get_effective_stage() == 1


def test_evaluation_without_resume_selection_returns_none(
    runtime_context: RuntimeContext,
    tmp_path: Path,
) -> None:
    manager = StageManager.__new__(StageManager)
    manager.context = runtime_context
    manager.context = type(runtime_context)(
        device=runtime_context.device,
        dtype=runtime_context.dtype,
        num_threads=runtime_context.num_threads,
        seed=runtime_context.seed,
        deterministic_ops=runtime_context.deterministic_ops,
        load_dir=tmp_path,
        save_dir=tmp_path,
    )
    manager.load_dir = tmp_path
    manager.resume_selection = None
    manager.stage_detail = [{"stage": {}}]

    assert manager._prepare_evaluation_stage() is None

@pytest.mark.parametrize("current_stage", [-1, 3])
def test_stage_manager_rejects_invalid_stage_index(
    current_stage: int,
) -> None:
    manager = StageManager.__new__(StageManager)
    manager.stage_detail = [{"stage1": {}}, {"stage2": {}}]
    manager.current_stage = current_stage

    with pytest.raises(RuntimeError, match="Invalid current stage index"):
        manager._get_effective_stage()


def test_stage_manager_cannot_train_after_completion() -> None:
    manager = StageManager.__new__(StageManager)
    manager.stage_detail = [{"stage1": {}}]
    manager.current_stage = 1

    with pytest.raises(RuntimeError, match="already been completed"):
        manager._train_current()


def test_runner_removes_stage_callback() -> None:
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    other_callback = OtherStoppingCallback()
    runner.callbacks = [other_callback, make_callback()]

    runner._set_stage_transition(None)

    assert runner.callbacks == [other_callback]
