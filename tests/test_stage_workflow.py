from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.application_entry import ApplicationEntry
from app.stage_manager import StageManager
from app.utils.context import RuntimeContext
from runners.base import BaseRunner
from runners.callbacks.stage import StageCallback


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
        *args: Any,
        **kwargs: Any,
    ) -> None:
        assert max_iterations is not None
        self.max_iterations = max_iterations
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

    def play(self, num_steps: int = 5000) -> None:
        self.play_calls.append(num_steps)

    def close(self) -> None:
        pass

    def save(self) -> None:
        pass


def make_stage_manager(context: RuntimeContext) -> StageManager:
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


def test_stage_manager_continues_with_each_stage_transition(
    runtime_context: RuntimeContext,
) -> None:
    manager = make_stage_manager(runtime_context)
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
) -> None:
    manager = make_stage_manager(runtime_context)
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
) -> None:
    manager = make_stage_manager(runtime_context)
    runner = manager.runner
    assert isinstance(runner, WorkflowRunner)
    application = ApplicationEntry.__new__(ApplicationEntry)
    application.stage_manager = manager

    application.train()
    application.test(num_episodes=7)
    application.play(num_steps=11)

    assert manager.current_stage == len(manager.stage_detail)
    assert runner.stage_callback is None
    assert runner.max_iterations_history == [3, 5, 5, 5]
    assert runner.test_calls == [7]
    assert runner.play_calls == [11]
