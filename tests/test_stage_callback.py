from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import app.stage_manager as stage_manager_module
from app.utils.context import RuntimeContext
from app.stage_manager import StageManager, StageTrainResult
from runners.callbacks.base import BaseCallback
from runners.callbacks.stage import StageCallback
from runners.on_policy import OnPolicyRunner
from utils.component import Component, ComponentInfo


class OtherStoppingCallback(BaseCallback):
    def __init__(self) -> None:
        self.runner = None

    def _on_step_end(self, *args, **kwargs) -> bool:
        return False


def make_callback(
    method: str = "mean",
    metric: str = "*/action_diff_l2",
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


def test_stage_callback_maps_reward_term_metric_to_info_key() -> None:
    callback = StageCallback(condition=[
        {
            "method": "mean",
            "metric": "*/action_diff_l2",
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
    callback = make_callback(metric="*/mean_len", window=2)

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


def test_stage_callback_rejects_ambiguous_metric_pattern() -> None:
    callback = make_callback(metric="*/loss", window=1)

    with pytest.raises(ValueError, match="is ambiguous"):
        callback._on_iteration_end(info={
            "rollout/loss": 1.0,
            "policy/loss": 1.0,
        })


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
                "metric": "*/action_diff_l2",
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
        "_build_current_stage_callback",
        lambda: stage_callback,
    )
    monkeypatch.setattr(
        manager,
        "_build_runner",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(manager.runner, "train", lambda: None)

    stage_callback.stop_training = True
    manager.runner.stop_callback = [stage_callback]
    assert manager._train_current() is (
        StageTrainResult.STAGE_COMPLETED
    )

    stage_callback.stop_training = False
    manager.runner.stop_callback = [OtherStoppingCallback()]
    assert manager._train_current() is (
        StageTrainResult.STOPPED_BY_CALLBACK
    )

    manager.runner.stop_callback = []
    assert manager._train_current() is (
        StageTrainResult.MAX_ITERATIONS_REACHED
    )


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
                "metric": "*/action_diff_l2",
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
    monkeypatch.setattr(manager.runner, "train", lambda: None)

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

        def stage_update(self, stage_callback) -> None:
            captured["stage_callback"] = stage_callback

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

    runner.stage_update(None)

    assert runner.callbacks == [other_callback]
