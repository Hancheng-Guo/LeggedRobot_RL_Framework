from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import app.application_entry as application_entry_module
from app.application_entry import ApplicationEntry
from app.stage_manager import StageManager
from app.utils.context import RuntimeContext
from runners.base import BaseRunner
from runners.callbacks.stage import StageCallback
from runners.utils.frames import VideoFormat, VideoFormats


def test_application_train_archives_configs_before_training(
    tmp_path: Path,
) -> None:
    load_dir = tmp_path / "source"
    save_dir = tmp_path / "run"
    config_dir = load_dir / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "app.yaml").write_text("runtime: {}", encoding="utf-8")
    trained: list[bool] = []

    application = ApplicationEntry.__new__(ApplicationEntry)
    application.load_dir = load_dir
    application.save_dir = save_dir
    application.stage_manager = cast(
        Any,
        SimpleNamespace(train=lambda: trained.append(True)),
    )
    application._closed = False

    application.train()

    assert trained == [True]
    assert (save_dir / "configs" / "app.yaml").read_text(
        encoding="utf-8",
    ) == "runtime: {}"


def test_application_rejects_incomplete_historical_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    application = ApplicationEntry.__new__(ApplicationEntry)
    application.app_name = "missing"
    application.train_time = datetime(2026, 1, 2, 3, 4, 5)

    with pytest.raises(FileNotFoundError, match="missing.*yaml"):
        application._get_runtime_dir(skip_check=False)


def test_application_device_override_is_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "runtime": {"device": "cuda", "dtype": "float32"},
        "component": {},
        "stage": [],
    }
    received_runtime: dict[str, Any] = {}
    context = cast(RuntimeContext, SimpleNamespace(device="cpu"))

    monkeypatch.setattr(
        ApplicationEntry,
        "_get_runtime_dir",
        lambda self, skip_check: (tmp_path, tmp_path),
    )
    monkeypatch.setattr(
        application_entry_module,
        "load_yaml",
        lambda path: config,
    )

    def create_context(**kwargs: Any) -> RuntimeContext:
        received_runtime.update(kwargs["runtime_config"])
        return context

    monkeypatch.setattr(
        application_entry_module,
        "create_runtime_context",
        create_context,
    )
    monkeypatch.setattr(
        application_entry_module,
        "StageManager",
        lambda **kwargs: SimpleNamespace(),
    )

    with pytest.warns(UserWarning, match="'cuda' -> 'cpu'"):
        application = ApplicationEntry("app", device="cpu")

    assert application.context is context
    assert received_runtime["device"] == "cpu"
    assert application.config["runtime"]["device"] == "cuda"


class WorkflowRunner(BaseRunner):

    def __init__(self, context: RuntimeContext) -> None:
        super().__init__(context)
        self.stage_callback: StageCallback | None = None
        self.max_iterations_history: list[int] = []
        self.callback_results: list[list[bool]] = []
        self.test_calls: list[int] = []
        self.play_calls: list[int] = []
        self.allow_transition = True

    def config_update(
        self,
        max_iterations: int | None = None,
        rollout_length: int | None = None,
        rollout_length_history_size: int | None = None,
        callbacks: Sequence[str | Mapping[str, Any]] | None = None,
        stage_index: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        assert max_iterations is not None
        self.max_iterations = max_iterations
        self.stage_index = stage_index
        self.max_iterations_history.append(max_iterations)

    def stage_update(
        self,
        stage_callback: StageCallback | None,
    ) -> None:
        self.stage_callback = stage_callback

    def train(self) -> None:
        callback = self.stage_callback
        assert callback is not None
        callback._on_train_start()
        self.stop_callback.clear()

        if self.max_iterations == 3:
            results = [callback._on_step_end(info={"reward": 1.0})]
            if self.allow_transition:
                results.append(
                    callback._on_step_end(info={"reward": 3.0})
                )
        else:
            results = [
                callback._on_iteration_end(
                    info={"rollout/mean_len": 5.0}
                ),
                callback._on_iteration_end(
                    info={"rollout/mean_len": 4.0}
                ),
            ]

        self.callback_results.append(results)
        if results[-1] is False:
            self.stop_callback.append(callback)

    def test(self, num_episodes: int = 1000) -> None:
        self.test_calls.append(num_episodes)

    def play(
        self,
        num_steps: int = 5000,
        formats: VideoFormat | VideoFormats = "gif",
    ) -> None:
        self.play_calls.append(num_steps)

    def close(self) -> None:
        pass

    def save(self, path: Path | None = None) -> Path:
        if path is None:
            path = (
                Path(self.context.save_dir)
                / "checkpoints"
                / f"stage_{self.stage_index:03d}"
                / "latest.pt"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def prepare_checkpoint_load(self, path: Path) -> dict[str, Any]:
        raise NotImplementedError

    def load(self, load_optimizer: bool = False) -> None:
        raise NotImplementedError


def make_stage_manager(
    context: RuntimeContext,
    save_dir: Path,
) -> StageManager:
    context = replace(
        context,
        load_dir=save_dir,
        save_dir=save_dir,
    )
    manager = StageManager(
        component={},
        context=context,
        stage_detail=[
            {
                "step_stage": {
                    "max_iterations": 3,
                    "transition": [{
                        "method": "mean",
                        "metric": "reward",
                        "operator": ">=",
                        "threshold": 2.0,
                        "window": 2,
                    }],
                }
            },
            {
                "iteration_stage": {
                    "max_iterations": 5,
                    "transition": [{
                        "method": "min",
                        "metric": "*/mean_len",
                        "operator": ">=",
                        "threshold": 4.0,
                        "window": 2,
                    }],
                }
            },
        ],
        load_dir=Path("."),
    )
    manager.runner = WorkflowRunner(context)
    return manager


def test_training_resume_selects_latest_stage_checkpoint(
    runtime_context: RuntimeContext,
    tmp_path: Path,
) -> None:
    runtime_context = replace(
        runtime_context,
        load_dir=tmp_path,
        save_dir=tmp_path,
    )
    manager = StageManager(
        component={},
        context=runtime_context,
        stage_detail=[
            {"stage_0": {"max_iterations": 1, "transition": [True]}},
            {"stage_1": {"max_iterations": 1, "transition": [True]}},
        ],
        load_dir=tmp_path,
    )
    first = tmp_path / "checkpoints" / "stage_000" / "latest.pt"
    second = tmp_path / "checkpoints" / "stage_001" / "latest.pt"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()

    resume_info = manager._prepare_training_stage()
    assert resume_info is not None
    assert resume_info.checkpoint_path == second
    assert resume_info.stage_completed is False
    assert manager.current_stage == 1

    (second.parent / "stage_completed").touch()
    resume_info = manager._prepare_training_stage()
    assert resume_info is not None
    assert resume_info.checkpoint_path == second
    assert resume_info.stage_completed is True


def test_stage_manager_continues_with_each_stage_transition(
    runtime_context: RuntimeContext,
    tmp_path: Path,
) -> None:
    manager = make_stage_manager(runtime_context, tmp_path)
    runner = manager.runner
    assert isinstance(runner, WorkflowRunner)

    manager.train()

    assert manager.current_stage == len(manager.stage_detail)
    assert manager.continue_training is False
    assert runner.max_iterations_history == [3, 5]
    assert runner.callback_results == [
        [True, False],
        [True, False],
    ]


def test_stage_manager_does_not_advance_when_condition_is_not_met(
    runtime_context: RuntimeContext,
    tmp_path: Path,
) -> None:
    manager = make_stage_manager(runtime_context, tmp_path)
    runner = manager.runner
    assert isinstance(runner, WorkflowRunner)
    runner.allow_transition = False

    manager.train()

    assert manager.current_stage == 0
    assert manager.continue_training is True
    assert runner.callback_results == [[True]]

    runner.allow_transition = True
    manager.train()

    assert manager.current_stage == len(manager.stage_detail)
    assert runner.callback_results == [
        [True],
        [True, False],
        [True, False],
    ]


def test_application_entry_can_test_and_play_after_training(
    runtime_context: RuntimeContext,
    tmp_path: Path,
) -> None:
    manager = make_stage_manager(runtime_context, tmp_path)
    runner = manager.runner
    assert isinstance(runner, WorkflowRunner)
    application = ApplicationEntry.__new__(ApplicationEntry)
    application.stage_manager = manager
    application.load_dir = tmp_path
    application.save_dir = tmp_path

    application.train()
    application.test(num_episodes=7)
    application.play(num_steps=11)

    assert manager.current_stage == len(manager.stage_detail)
    assert runner.stage_callback is None
    assert runner.max_iterations_history == [3, 5]
    assert runner.test_calls == [7]
    assert runner.play_calls == [11]
