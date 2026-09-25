import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from runners.callbacks.checkpoint import CheckpointCallback
from runners.callbacks.early_stopping import EarlystoppingCallback
from runners.callbacks.logging import LoggingCallback
from runners.callbacks.keyboard_interrupt import KeyboardInterruptCallback
from runners.callbacks.progress_bar import ProgressBarCallback
from runners.callbacks import tensorboard as tensorboard_module
from runners.callbacks.tensorboard import TensorboardCallback
from runners.callbacks.adaptive_learning_rate import (
    AdaptiveLearningRateCallback,
)
from runners.base import BaseRunner
from runners.on_policy import OnPolicyRunner
from app.utils.context import RuntimeContext
from utils.logging import configure_logging, get_logger


class DummyAlgorithm:
    def __init__(self) -> None:
        self.policy = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.Adam(self.policy.parameters())
        self.learning_rate = 0.001

    def set_learning_rate(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

    def checkpoint_state_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "optimizer": self.optimizer.state_dict(),
            "policy": {
                "type": type(self.policy).__name__,
                "state_dict": self.policy.state_dict(),
            },
        }

    def load_checkpoint_state_dict(
        self,
        state: dict[str, Any],
        load_optimizer: bool = False,
    ) -> None:
        self.policy.load_state_dict(state["policy"]["state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])

    def save_module_artifacts(self, directory: Path) -> list[Path]:
        return []


def make_runner() -> BaseRunner:
    return cast(BaseRunner, SimpleNamespace(
        current_iteration=0,
        rollout_length=1,
        algorithm=DummyAlgorithm(),
    ))


def test_keyboard_interrupt_stops_and_saves_after_update(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "latest.pt"
    saved: list[bool] = []
    runner = cast(BaseRunner, SimpleNamespace(
        save=lambda: (saved.append(True), checkpoint_path)[1],
    ))
    callback = KeyboardInterruptCallback(runner)
    callback._stop_requested = True

    assert callback._on_step_end() is False
    assert callback._on_iteration_end() is False
    assert callback._on_train_end() is True
    assert saved == [True]


def make_context(save_dir: Path) -> RuntimeContext:
    return cast(
        RuntimeContext,
        SimpleNamespace(
            save_dir=save_dir,
            device="cpu",
        ),
    )


def make_checkpoint_runner(
    save_dir: Path,
    stage_index: int | None = None,
) -> OnPolicyRunner:
    runner = OnPolicyRunner(context=make_context(save_dir))
    runner.current_iteration = 0
    runner.stage_index = stage_index
    runner.algorithm = cast(Any, DummyAlgorithm())
    return runner


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


def test_adaptive_learning_rate_tracks_metric_range() -> None:
    runner = make_runner()
    algorithm = cast(Any, runner.algorithm)
    algorithm.learning_rate = 0.001
    callback = AdaptiveLearningRateCallback(
        runner=runner,
        monitor="rollout/approx_kl",
        allowed_range=(0.01, 0.02),
        factor=0.5,
        buffer_len=2,
    )
    callback._on_train_start()

    low_info = {"rollout/approx_kl": 0.005}
    callback._on_iteration_end(low_info)
    assert algorithm.learning_rate == pytest.approx(0.001)
    callback._on_iteration_end(low_info)
    assert algorithm.learning_rate == pytest.approx(0.002)
    assert algorithm.optimizer.param_groups[0]["lr"] == pytest.approx(
        0.002
    )
    assert "rollout/learning_rate" not in low_info

    in_range_info = {"rollout/approx_kl": 0.015}
    callback._on_iteration_end(in_range_info)
    assert algorithm.learning_rate == pytest.approx(0.002)

    high_info = {"rollout/approx_kl": 0.03}
    callback._on_iteration_end(high_info)
    assert algorithm.learning_rate == pytest.approx(0.001)
    assert "rollout/learning_rate" not in high_info


def test_adaptive_learning_rate_rejects_invalid_range() -> None:
    with pytest.raises(ValueError, match="lower bound"):
        AdaptiveLearningRateCallback(
            runner=make_runner(),
            monitor="rollout/approx_kl",
            allowed_range=(0.02, 0.01),
            factor=0.5,
        )


def test_adaptive_learning_rate_supports_reverse_direction() -> None:
    runner = make_runner()
    algorithm = cast(Any, runner.algorithm)
    algorithm.learning_rate = 0.001
    callback = AdaptiveLearningRateCallback(
        runner=runner,
        monitor="rollout/entropy",
        allowed_range=(1.0, 2.0),
        factor=0.5,
        buffer_len=2,
    )
    callback._on_train_start()

    callback._on_iteration_end({"rollout/entropy": 0.5})
    assert algorithm.learning_rate == pytest.approx(0.001)
    callback._on_iteration_end({"rollout/entropy": 0.5})
    assert algorithm.learning_rate == pytest.approx(0.0005)

    callback._on_iteration_end({"rollout/entropy": 2.5})
    assert algorithm.learning_rate == pytest.approx(0.0005)
    callback._on_iteration_end({"rollout/entropy": 2.5})
    assert algorithm.learning_rate == pytest.approx(0.001)


def test_adaptive_learning_rate_ignores_nonfinite_metric() -> None:
    runner = make_runner()
    algorithm = cast(Any, runner.algorithm)
    algorithm.learning_rate = 0.001
    callback = AdaptiveLearningRateCallback(
        runner=runner,
        monitor="rollout/approx_kl",
        allowed_range=(0.01, 0.02),
        factor=0.5,
        buffer_len=2,
    )
    callback._on_train_start()

    callback._on_iteration_end({"rollout/approx_kl": float("nan")})
    callback._on_iteration_end({"rollout/approx_kl": 0.005})
    assert algorithm.learning_rate == pytest.approx(0.001)

    callback._on_iteration_end({"rollout/approx_kl": 0.005})
    assert algorithm.learning_rate == pytest.approx(0.002)


def test_adaptive_learning_rate_rejects_unsupported_monitor() -> None:
    with pytest.raises(KeyError, match="rollout/loss"):
        AdaptiveLearningRateCallback(
            runner=make_runner(),
            monitor="rollout/loss",
            allowed_range=(0.1, 1.0),
            factor=0.5,
        )


def test_runner_builds_adaptive_learning_rate_from_yaml_style_config(
    runtime_context: RuntimeContext,
) -> None:
    runner = OnPolicyRunner(context=runtime_context)
    runner.max_iterations = 10
    runner.rollout_length = 4

    runner._build_callbacks(callbacks=[{
        "adaptive_learning_rate": {
            "monitor": "rollout/approx_kl",
            "allowed_range": [0.005, 0.02],
            "factor": 0.5,
        }
    }])

    assert len(runner.callbacks) == 1
    callback = runner.callbacks[0]
    assert isinstance(callback, AdaptiveLearningRateCallback)
    assert callback.lower_bound == pytest.approx(0.005)
    assert callback.upper_bound == pytest.approx(0.02)
    assert callback.factor == pytest.approx(0.5)


def test_runner_checkpoint_restores_callback_runtime_state(
    tmp_path: Path,
) -> None:
    source = make_checkpoint_runner(tmp_path, stage_index=1)
    source.max_iterations = 20
    source.rollout_length = 4
    source._build_callbacks(callbacks=[{
        "adaptive_learning_rate": {
            "monitor": "rollout/approx_kl",
            "allowed_range": [0.01, 0.02],
            "factor": 0.5,
            "buffer_len": 3,
        },
    }])
    source_callback = cast(
        AdaptiveLearningRateCallback,
        source.callbacks[0],
    )
    source_callback._on_train_start()
    source_callback._on_iteration_end({"rollout/approx_kl": 0.012})
    source_callback._on_iteration_end({"rollout/approx_kl": 0.013})
    checkpoint_path = source.save(tmp_path / "resume.pt")

    restored = make_checkpoint_runner(tmp_path, stage_index=1)
    restored.max_iterations = 20
    restored.rollout_length = 4
    restored.prepare_checkpoint_load(checkpoint_path)
    restored._build_callbacks(callbacks=[{
        "adaptive_learning_rate": {
            "monitor": "rollout/approx_kl",
            "allowed_range": [0.01, 0.02],
            "factor": 0.5,
            "buffer_len": 3,
        },
    }])
    restored.load(load_optimizer=True)
    restored_callback = cast(
        AdaptiveLearningRateCallback,
        restored.callbacks[0],
    )
    restored_callback._on_train_start()
    restored._load_pending_callback_states()

    assert list(restored_callback._buffer) == pytest.approx([0.012, 0.013])
    assert restored_callback._hold_current_iters == 0


def test_progress_bar_resumes_from_runner_iteration() -> None:
    runner = make_runner()
    runner.current_iteration = 4
    callback = ProgressBarCallback(
        runner=runner,
        max_iterations=10,
        rollout_length=8,
    )

    callback._on_train_start()

    assert callback._completed_iterations == 5
    assert callback._completed_steps == 40


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


def test_runner_rebuilds_previous_callbacks_when_config_is_none(
    runtime_context: RuntimeContext,
) -> None:
    runner = OnPolicyRunner(context=runtime_context)
    runner.max_iterations = 10
    runner.rollout_length = 4

    runner._build_callbacks(callbacks=["progress_bar"])
    original_callback = runner.callbacks[0]
    assert isinstance(original_callback, ProgressBarCallback)
    original_callback._completed_steps = 7

    runner.max_iterations = 20
    runner._build_callbacks(callbacks=None)

    rebuilt_callback = runner.callbacks[0]
    assert isinstance(rebuilt_callback, ProgressBarCallback)
    assert rebuilt_callback is not original_callback
    assert rebuilt_callback.max_iterations == 20
    assert rebuilt_callback._completed_steps == 0


def test_logging_callback_writes_scalar_metrics(tmp_path: Path) -> None:
    session = configure_logging(
        tmp_path / "logs" / "training.log",
        console=False,
    )
    runner = make_runner()
    runner.stage_index = 0
    callback = LoggingCallback(runner=runner)
    callback._on_train_start()
    callback._on_iteration_end({
        "runner/current_iter": 0,
        "loss": torch.tensor(1.25),
        "structured": torch.ones(2),
    })
    callback._on_train_end()
    session.close()

    content = (tmp_path / "logs" / "training.log").read_text(
        encoding="utf-8"
    )
    assert "Stage 0, iteration 1" in content
    assert " | INFO | Training started for stage 0.\n" in content
    assert "\n    loss" in content
    assert ":         1.25" in content
    assert "runner/current_iter" in content
    assert "structured" not in content
    assert "runner/current_iter :            0\n" in content


def test_logging_callback_reports_test_and_play_lifecycle(
    tmp_path: Path,
) -> None:
    session = configure_logging(
        tmp_path / "logs" / "training.log",
        console=False,
    )
    runner = make_runner()
    runner.stage_index = 1
    callback = LoggingCallback(runner=runner)

    callback._on_test_start()
    callback._on_test_end({
        "num_episodes": 3,
        "mean_reward": 1.25,
        "mean_episode_length": 20.0,
    })
    callback._on_play_start()
    output_path = tmp_path / "videos" / "play.gif"
    callback._on_play_end({"output_paths": [output_path]})
    session.close()

    content = (tmp_path / "logs" / "training.log").read_text(
        encoding="utf-8"
    )
    assert "Testing started for stage 1." in content
    assert "Testing ended for stage 1: 3 episodes" in content
    assert "Playback started for stage 1." in content
    assert f"Playback ended. Saved output to: {output_path.as_posix()}" in content


def test_global_logger_uses_callback_handlers(
    tmp_path: Path,
) -> None:
    session = configure_logging(
        tmp_path / "logs" / "training.log",
        console=False,
    )
    get_logger("tests").info("Message from another component.")
    session.close()

    content = (tmp_path / "logs" / "training.log").read_text(
        encoding="utf-8"
    )
    assert " | INFO | Message from another component." in content


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


def test_tensorboard_validates_max_reload_threads(tmp_path: Path) -> None:
    callback = TensorboardCallback(
        runner=make_runner(),
        context=make_context(tmp_path),
        max_reload_threads=4,
    )
    assert callback.max_reload_threads == 4
    callback._on_close()

    with pytest.raises(ValueError, match="max_reload_threads"):
        TensorboardCallback(
            runner=make_runner(),
            context=make_context(tmp_path),
            max_reload_threads=0,
        )


def test_tensorboard_callback_writes_stage_event_file(
    tmp_path: Path,
) -> None:
    callback = TensorboardCallback(
        runner=make_runner(),
        context=make_context(tmp_path),
        stage_index=1,
    )
    callback._on_iteration_end({"ppo/loss": 1.0})
    callback._on_close()

    event_files = list(
        (
            tmp_path
            / "tensorboard"
            / "stage_001"
        ).glob("events.out.*")
    )
    assert event_files


def test_tensorboard_reduces_configured_vector_metrics(
    tmp_path: Path,
) -> None:
    class RecordingWriter:
        def __init__(self) -> None:
            self.default_bins = [-10.0, 0.0, 10.0]
            self.scalars: dict[str, float] = {}
            self.scalar_steps: dict[str, int] = {}
            self.histograms: list[str] = []
            self.histogram_values: list[dict[str, Any]] = []

        def add_scalar(
            self,
            name: str,
            value: float,
            step: int,
        ) -> None:
            self.scalars[name] = float(value)
            self.scalar_steps[name] = step

        def add_histogram_raw(
            self,
            name: str,
            **kwargs: Any,
        ) -> None:
            self.histograms.append(name)
            self.histogram_values.append(kwargs)

        def close(self) -> None:
            pass

    callback = TensorboardCallback(
        runner=make_runner(),
        context=make_context(tmp_path),
        histogram_step_interval=2,
        step_metrics={
            "reward": ["value"],
            "episode_length": ["value"],
            "termination/*": ["mean"],
            "action/action": ["mean", "std", "min", "max", "histogram"],
        },
    )
    writer = RecordingWriter()
    callback.writer = cast(Any, writer)
    info = {
        "reward": 1.5,
        "episode_length": torch.tensor(12.0),
        "ignored_scalar": 2.5,
        "termination/fall": torch.tensor([True, False]),
        "action/action": torch.tensor([[-1.0, 1.0], [3.0, 5.0]]),
        "observation/policy": torch.ones(2, 4),
    }

    callback._on_step_end(info)
    assert writer.scalars["reward"] == pytest.approx(1.5)
    assert writer.scalars["episode_length"] == pytest.approx(12.0)
    assert "ignored_scalar" not in writer.scalars
    assert writer.scalars["termination/fall/mean"] == pytest.approx(0.5)
    assert writer.scalars["action/action/mean"] == pytest.approx(2.0)
    assert writer.scalars["action/action/std"] == pytest.approx(5 ** 0.5)
    assert writer.scalars["action/action/min"] == pytest.approx(-1.0)
    assert writer.scalars["action/action/max"] == pytest.approx(5.0)
    assert not any(name.startswith("observation/") for name in writer.scalars)
    assert writer.histograms == []
    accumulator = callback._pending_histograms["action/action"]
    assert accumulator.counts.numel() == 2
    assert int(accumulator.num.item()) == 4

    callback._on_step_end(info)
    assert writer.histograms == ["action/action/distribution"]
    histogram = writer.histogram_values[0]
    assert histogram["min"] == pytest.approx(-1.0)
    assert histogram["max"] == pytest.approx(5.0)
    assert histogram["num"] == 8
    assert histogram["sum"] == pytest.approx(16.0)
    assert histogram["sum_squares"] == pytest.approx(72.0)
    assert histogram["bucket_limits"] == [-10.0, 0.0, 10.0]
    assert histogram["bucket_counts"] == [0, 2, 6]
    assert histogram["global_step"] == 2
    assert callback._pending_histograms == {}
    callback._on_iteration_end({"rollout/loss": 0.25})
    assert writer.scalar_steps["rollout/loss"] == callback.global_step
    callback._on_close()


def test_tensorboard_purges_events_after_resumed_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer_arguments: dict[str, Any] = {}

    class RecordingSummaryWriter:
        def __init__(self, **kwargs: Any) -> None:
            writer_arguments.update(kwargs)

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeTensorBoard:
        def configure(self, argv: tuple[str, ...]) -> None:
            pass

        def launch(self) -> str:
            logging.getLogger("tensorboard").info(
                "TensorBoard done reloading. Load took 0.000 secs"
            )
            return "http://127.0.0.1:43123/"

    monkeypatch.setattr(
        tensorboard_module,
        "SummaryWriter",
        RecordingSummaryWriter,
    )
    monkeypatch.setattr(
        tensorboard_module,
        "TensorBoard",
        FakeTensorBoard,
    )
    monkeypatch.setattr(TensorboardCallback, "_SERVER_URLS", {})
    runner = make_runner()
    runner.current_iteration = 4
    runner.rollout_length = 8
    callback = TensorboardCallback(
        runner=runner,
        context=make_context(tmp_path),
    )

    callback._on_train_start()

    assert callback.global_step == 40
    assert writer_arguments["purge_step"] == 41
    callback._on_close()


def test_tensorboard_starts_server_and_logs_returned_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTensorBoard:
        configured_argv: tuple[str, ...] = ()

        def configure(self, argv: tuple[str, ...]) -> None:
            FakeTensorBoard.configured_argv = argv

        def launch(self) -> str:
            logging.getLogger("tensorboard").info(
                "TensorBoard done reloading. Load took 0.000 secs"
            )
            return "http://127.0.0.1:43123/"

    monkeypatch.setattr(
        tensorboard_module,
        "TensorBoard",
        FakeTensorBoard,
    )
    runner = make_runner()
    tensorboard = TensorboardCallback(
        runner=runner,
        context=make_context(tmp_path),
    )
    logging_session = configure_logging(
        tmp_path / "logs" / "training.log",
        console=False,
    )
    runner.callbacks = [tensorboard]

    tensorboard._on_train_start()
    logging_session.close()
    tensorboard._on_close()

    content = (tmp_path / "logs" / "training.log").read_text(
        encoding="utf-8"
    )
    assert f"TensorBoard dir: {tensorboard.tensorboard_log_dir.as_posix()}" in content
    assert "TensorBoard url: http://127.0.0.1:43123/" in content
    assert FakeTensorBoard.configured_argv == (
        "tensorboard",
        "--logdir",
        str(tensorboard.tensorboard_log_dir),
        "--host",
        "0.0.0.0",
        "--load_fast",
        "false",
        "--max_reload_threads",
        "4",
    )


def test_tensorboard_reuses_server_across_stage_callbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTensorBoard:
        configured_argvs: list[tuple[str, ...]] = []
        launches = 0

        def configure(self, argv: tuple[str, ...]) -> None:
            FakeTensorBoard.configured_argvs.append(argv)

        def launch(self) -> str:
            FakeTensorBoard.launches += 1
            logging.getLogger("tensorboard").info(
                "TensorBoard done reloading. Load took 0.000 secs"
            )
            return "http://127.0.0.1:43123/"

    monkeypatch.setattr(
        tensorboard_module,
        "TensorBoard",
        FakeTensorBoard,
    )
    monkeypatch.setattr(TensorboardCallback, "_SERVER_URLS", {})
    runner = make_runner()
    context = make_context(tmp_path)

    first = TensorboardCallback(
        runner=runner,
        context=context,
        stage_index=0,
    )
    second = TensorboardCallback(
        runner=runner,
        context=context,
        stage_index=1,
    )

    first._on_train_start()
    second._on_train_start()
    first._on_close()
    second._on_close()

    assert FakeTensorBoard.launches == 1
    assert FakeTensorBoard.configured_argvs == [(
        "tensorboard",
        "--logdir",
        str((tmp_path / "tensorboard").resolve()),
        "--host",
        "0.0.0.0",
        "--load_fast",
        "false",
        "--max_reload_threads",
        "4",
    )]
    assert first.tensorboard_log_dir == (
        tmp_path / "tensorboard" / "stage_000"
    ).resolve()
    assert second.tensorboard_log_dir == (
        tmp_path / "tensorboard" / "stage_001"
    ).resolve()


def test_checkpoint_callback_saves_policy_and_optimizer(
    tmp_path: Path,
) -> None:
    runner = make_checkpoint_runner(tmp_path)
    callback = CheckpointCallback(
        runner=runner,
        context=make_context(tmp_path),
        save_iter_interval=1,
    )
    callback._on_iteration_end({"ppo/loss": 1.0})
    callback._on_iteration_finalize()

    checkpoint_path = (
        tmp_path / "checkpoints" / "checkpoint_00000001.pt"
    )
    payload: dict[str, Any] = torch.load(
        checkpoint_path,
        weights_only=False,
    )
    assert checkpoint_path.is_file()
    assert (tmp_path / "checkpoints" / "latest.pt").is_file()
    assert "format_version" not in payload
    runner_state = payload["runner"]
    assert runner_state["current_iteration"] == 0
    assert "global_iteration" not in runner_state
    assert "metrics" not in runner_state
    assert "stage_completed" not in runner_state
    assert "policy" in runner_state["algorithm"]
    assert "state_dict" in runner_state["algorithm"]["policy"]
    assert "optimizer" in runner_state["algorithm"]
    assert "policy_state_dict" not in payload
    assert "optimizer_state_dict" not in payload
    assert not list((tmp_path / "checkpoints").glob("*.tmp"))


def test_checkpoint_callback_uses_stage_directory(
    tmp_path: Path,
) -> None:
    runner = make_checkpoint_runner(tmp_path, stage_index=1)
    callback = CheckpointCallback(
        runner=runner,
        context=make_context(tmp_path),
        save_iter_interval=1,
        stage_index=1,
    )
    callback._on_iteration_end({"ppo/loss": 1.0})
    callback._on_iteration_finalize()

    checkpoint_path = (
        tmp_path
        / "checkpoints"
        / "stage_001"
        / "checkpoint_00000001.pt"
    )
    payload: dict[str, Any] = torch.load(
        checkpoint_path,
        weights_only=False,
    )
    assert checkpoint_path.is_file()
    assert payload["runner"]["stage_index"] == 1


def test_checkpoint_callback_does_not_save_on_train_end_by_default(
    tmp_path: Path,
) -> None:
    runner = make_checkpoint_runner(tmp_path)
    runner.current_iteration = -1
    callback = CheckpointCallback(
        runner=runner,
        context=make_context(tmp_path),
        save_iter_interval=100,
    )

    callback._on_train_start()
    callback._on_iteration_end({})
    callback._on_train_end()

    assert not list((tmp_path / "checkpoints").glob("*.pt"))


def test_checkpoint_callback_can_explicitly_save_on_train_end(
    tmp_path: Path,
) -> None:
    runner = make_checkpoint_runner(tmp_path)
    runner.current_iteration = -1
    callback = CheckpointCallback(
        runner=runner,
        context=make_context(tmp_path),
        save_iter_interval=100,
        directory_name="explicit_checkpoints",
        save_on_train_end=True,
    )

    callback._on_train_start()
    callback._on_iteration_end({})
    callback._on_train_end()

    checkpoint_dir = tmp_path / "explicit_checkpoints"
    assert (checkpoint_dir / "checkpoint_00000001.pt").is_file()
    assert (checkpoint_dir / "latest.pt").is_file()


def test_progress_bar_reports_completed_iteration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = make_runner()
    runner.current_iteration = -1
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
    assert "0/2 iterations" in progress_output
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


def test_progress_bar_tracks_play_steps(
    capsys: pytest.CaptureFixture[str],
) -> None:
    callback = ProgressBarCallback(
        runner=make_runner(),
        max_iterations=2,
        rollout_length=4,
        width=4,
        refresh_interval=0.0,
    )

    callback._on_play_start(num_steps=4)
    callback._on_step_end()
    output = capsys.readouterr().out
    callback._on_play_end()

    assert "[#>--]" in output
    assert "1/4 steps" in output
    assert "25.00%" in output


def test_progress_bar_tracks_test_steps_and_episodes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    callback = ProgressBarCallback(
        runner=make_runner(),
        max_iterations=2,
        rollout_length=4,
        width=4,
        refresh_interval=0.0,
    )

    callback._on_test_start(num_episodes=10)
    callback._on_step_end(completed_episodes=3, total_episodes=10)
    output = capsys.readouterr().out
    callback._on_test_end()

    assert "#>--]" in output
    assert "3/10 episodes" in output
    assert "30.00%" in output
