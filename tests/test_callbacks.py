from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from runners.callbacks.checkpoint import CheckpointCallback
from runners.callbacks.early_stopping import EarlystoppingCallback
from runners.callbacks.logging import LoggingCallback
from runners.callbacks.progress_bar import ProgressBarCallback
from runners.callbacks.tensorboard import TensorboardCallback
from runners.base import BaseRunner
from runners.on_policy import OnPolicyRunner
from app.utils.context import RuntimeContext


class DummyAlgorithm:
    def __init__(self) -> None:
        self.policy = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.Adam(self.policy.parameters())


def make_runner() -> BaseRunner:
    return cast(BaseRunner, SimpleNamespace(
        current_iteration=0,
        algorithm=DummyAlgorithm(),
    ))


def make_context(save_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(save_dir=save_dir)


def test_early_stopping_stops_at_no_improvement_limit() -> None:
    callback = EarlystoppingCallback(
        runner=make_runner(),
        monitor="score",
        mode="max",
        max_no_improve_iters=2,
        min_delta=0.0,
    )
    callback._on_train_start()

    assert callback._on_iteration_end({"score": 1.0}) is True
    assert callback._on_iteration_end({"score": 0.9}) is True
    assert callback._on_iteration_end({"score": 0.8}) is False


def test_early_stopping_rejects_missing_metric() -> None:
    callback = EarlystoppingCallback(
        runner=make_runner(),
        monitor="score",
        max_no_improve_iters=2,
        mode="min",
        min_delta=0.0,
    )

    with pytest.raises(KeyError, match="score"):
        callback._on_iteration_end({})


def test_early_stopping_default_monitor_matches_loss_namespace() -> None:
    callback = EarlystoppingCallback(
        runner=make_runner(),
        monitor="*/loss",
        max_no_improve_iters=1,
        mode="min",
        min_delta=0.0,
    )

    assert callback._on_iteration_end({"ppo/loss": 1.0}) is True
    assert callback._on_iteration_end({"ppo/loss": 1.1}) is False


def test_early_stopping_rejects_ambiguous_monitor_pattern() -> None:
    callback = EarlystoppingCallback(
        runner=make_runner(),
        monitor="*/loss",
        max_no_improve_iters=2,
        mode="min",
        min_delta=0.0,
    )

    with pytest.raises(ValueError, match="ambiguous"):
        callback._on_iteration_end({
            "actor/loss": 1.0,
            "critic/loss": 1.0,
        })


def test_runner_builds_early_stopping_from_yaml_style_config(
    runtime_context: RuntimeContext,
) -> None:
    runner = OnPolicyRunner(context=runtime_context)
    runner.max_iterations = 10
    runner.rollout_length = 4

    runner._build_callbacks(callbacks=[{
        "early_stopping": {
            "monitor": "actor/loss",
            "mode": "min",
            "max_no_improve_iters": 7,
            "min_delta": 0.01,
            "warmup_iters": 3,
        }
    }])

    assert len(runner.callbacks) == 1
    callback = runner.callbacks[0]
    assert isinstance(callback, EarlystoppingCallback)
    assert callback.monitor == "actor/loss"
    assert callback.max_no_improve_iters == 7


def test_runner_rejects_invalid_callback_mapping(
    runtime_context: RuntimeContext,
) -> None:
    runner = OnPolicyRunner(context=runtime_context)
    runner.max_iterations = 10
    runner.rollout_length = 4

    with pytest.raises(ValueError, match="exactly one"):
        runner._build_callbacks(callbacks=[{
            "logging": {},
            "progress_bar": {},
        }])

    with pytest.raises(TypeError, match="must be a mapping"):
        runner._build_callbacks(callbacks=[{"logging": True}])


def test_logging_callback_writes_scalar_metrics(tmp_path: Path) -> None:
    runner = make_runner()
    callback = LoggingCallback(
        runner=runner,
        context=make_context(tmp_path),
        console=False,
    )
    callback._on_train_start()
    callback._on_iteration_end({
        "loss": torch.tensor(1.25),
        "structured": torch.ones(2),
    })
    callback._on_train_end()
    callback._on_close()

    content = (tmp_path / "logs" / "training.log").read_text(
        encoding="utf-8"
    )
    assert "Iteration 1" in content
    assert "Training started.\n\n" in content
    assert "loss :         1.25" in content
    assert "structured" not in content
    assert "loss :         1.25\n\n" in content


def test_tensorboard_callback_writes_event_file(tmp_path: Path) -> None:
    callback = TensorboardCallback(
        runner=make_runner(),
        context=make_context(tmp_path),
    )
    callback._on_step_end({"reward": torch.tensor(2.0)})
    callback._on_iteration_end({"ppo/loss": 1.0})
    callback._on_close()

    event_files = list((tmp_path / "tensorboard").glob("events.out.*"))
    assert event_files
    assert event_files[0].stat().st_size > 0


def test_checkpoint_callback_saves_policy_and_optimizer(
    tmp_path: Path,
) -> None:
    runner = make_runner()
    callback = CheckpointCallback(
        runner=runner,
        context=make_context(tmp_path),
        save_interval=1,
    )
    callback._on_iteration_end({"ppo/loss": 1.0})

    checkpoint_path = (
        tmp_path / "checkpoints" / "checkpoint_00000001.pt"
    )
    payload: dict[str, Any] = torch.load(
        checkpoint_path,
        weights_only=False,
    )
    assert checkpoint_path.is_file()
    assert (tmp_path / "checkpoints" / "latest.pt").is_file()
    assert payload["global_iteration"] == 1
    assert "policy_state_dict" in payload
    assert "optimizer_state_dict" in payload
    assert not list((tmp_path / "checkpoints").glob("*.tmp"))


def test_progress_bar_reports_completed_iteration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = make_runner()
    callback = ProgressBarCallback(
        runner=runner,
        max_iterations=2,
        rollout_length=4,
        width=4,
        refresh_interval=0.0,
    )
    callback._on_train_start()
    assert capsys.readouterr().out == ""
    callback._on_step_end()
    progress_output = capsys.readouterr().out
    callback._on_iteration_end({"runner/current_iter": 0})
    cleared_output = capsys.readouterr().out

    assert ">" in progress_output
    assert "0/2" in progress_output
    assert "12.50%" in progress_output
    assert "step=" not in progress_output
    assert "elapsed=" not in progress_output
    assert "[" not in cleared_output
    assert callback._completed_iterations == 1


def test_progress_bar_cursor_wraps_and_can_be_hidden() -> None:
    callback = ProgressBarCallback(
        runner=make_runner(),
        max_iterations=7,
        rollout_length=4,
        width=14,
        refresh_interval=0.0,
    )

    callback._cursor_position = 7
    assert callback._build_bar(filled=2) == "##----->------"

    callback._advance_cursor()
    assert callback._build_bar(filled=2) == "##------>-----"

    callback._cursor_position = 13
    callback._advance_cursor()
    assert callback._build_bar(filled=2) == "##------------"
