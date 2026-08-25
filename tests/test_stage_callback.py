import pytest
import torch

from app.utils.context import RuntimeContext
from app.stage_manager import StageManager, StageTrainResult
from runners.callbacks.base import BaseCallback
from runners.callbacks.stage import StageCallback
from runners.on_policy import OnPolicyRunner
from utils.component import Component


class OtherStoppingCallback(BaseCallback):
    def __init__(self) -> None:
        self.runner = None

    def _on_step_end(self, *args, **kwargs) -> bool:
        return False


def make_callback(
    operator: str = ">=",
    threshold: float = 2.0,
    window: int = 3,
) -> StageCallback:
    return StageCallback(condition={
        "mean_reward": {
            "operator": operator,
            "threshold": threshold,
            "window": window,
        }
    })


def test_stage_callback_uses_complete_sliding_window() -> None:
    callback = make_callback()
    callback._on_train_start()

    callback._on_step_end(info={"reward": torch.tensor(1.0)})
    callback._on_step_end(info={"reward": torch.tensor(2.0)})
    assert callback.stop_training is False

    callback._on_step_end(info={"reward": torch.tensor(3.0)})
    assert callback.stop_training is True


def test_stage_callback_latches_satisfied_condition() -> None:
    callback = make_callback(window=2)
    callback._on_step_end(info={"reward": 3.0})
    callback._on_step_end(info={"reward": 3.0})
    assert callback.stop_training is True

    callback._on_step_end(info={"reward": 0.0})
    assert callback.stop_training is True


def test_stage_callback_maps_reward_term_metric_to_info_key() -> None:
    callback = StageCallback(condition={
        "mean_action_diff_l2": {
            "operator": "<=",
            "threshold": 0.5,
            "window": 2,
        }
    })

    callback._on_step_end(info={"reward/action_diff_l2": 0.4})
    callback._on_step_end(info={"reward/action_diff_l2": 0.6})

    assert callback.stop_training is True


def test_stage_callback_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="Invalid stage metric"):
        StageCallback(condition={"unknown": {}})

    with pytest.raises(ValueError, match="Unsupported stage aggregation"):
        StageCallback(condition={"median_reward": {}})

    with pytest.raises(ValueError, match="positive integer"):
        make_callback(window=0)


def test_runner_records_callback_that_requests_stop(
    runtime_context: RuntimeContext,
) -> None:
    runner = OnPolicyRunner(context=runtime_context)
    callback = make_callback(window=1)
    runner.callbacks = [callback]
    runner.stop_callback.clear()

    should_continue = runner._run_callbacks(
        "_on_step_end",
        info={"reward": 2.0},
    )

    assert should_continue is False
    assert runner.stop_callback == [callback]


def test_stage_manager_distinguishes_stop_callback(
    runtime_context: RuntimeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = StageManager.__new__(StageManager)
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


def test_runner_removes_stage_callback() -> None:
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    other_callback = OtherStoppingCallback()
    runner.callbacks = [other_callback, make_callback()]

    runner.stage_update(None)

    assert runner.callbacks == [other_callback]
