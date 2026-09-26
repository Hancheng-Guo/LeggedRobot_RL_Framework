from collections.abc import Mapping, Sequence
from datetime import datetime, tzinfo
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from app.utils.context import RuntimeContext
from envs.base import BaseEnv
from envs.registry import ENV_TYPE_MAP
from rl.algorithms.base import OnPolicyAlgorithm, PolicyOutput
from runners.callbacks.base import BaseCallback
from runners.on_policy import OnPolicyRunner
from runners import on_policy as on_policy_module
from utils.component import Component, ComponentInfo


class TrackingEnvironment(BaseEnv):
    instances: list["TrackingEnvironment"] = []
    SUPPORTS_CONCURRENT_INSTANCES: bool = True

    @property
    def render_mode(self) -> str:
        return "human"

    def __init__(self, context: RuntimeContext) -> None:
        super().__init__(context)
        self.closed = False
        self.instances.append(self)

    def config_update(
        self,
        component: Component,
        num_envs: int = 1,
        **kwargs: Any,
    ) -> None:
        self.num_envs = num_envs

    def reset(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 1)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        raise NotImplementedError

    def close(self) -> None:
        self.closed = True

    def render(self) -> np.ndarray | None:
        return None


    @property
    def render_fps(self) -> float:
        return 50.0


    @property
    def action_dim(self) -> int:
        return 1


    @property
    def observation_dim(self) -> int:
        return 1


    @property
    def observation_slices(self) -> dict[str, slice]:
        return {"observation": slice(0, 1)}


class MinimalAlgorithm(OnPolicyAlgorithm):
    def set_train_mode(self) -> None:
        return None

    def config_update(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> PolicyOutput:
        raise NotImplementedError

    def process_transition(
        self,
        obs: torch.Tensor,
        policy_output: PolicyOutput,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        next_obs: torch.Tensor,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    def compute_returns(self, last_obs: torch.Tensor) -> None:
        raise NotImplementedError

    def update(self) -> dict[str, float]:
        raise NotImplementedError

    def set_learning_rate(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def close(self) -> None:
        return None


class StopAtStepStartCallback(BaseCallback):
    def __init__(self) -> None:
        self.runner = None

    def _on_step_start(self, *args: Any, **kwargs: Any) -> bool:
        return False


class StopAtIterationStartCallback(BaseCallback):
    def __init__(self) -> None:
        self.iterations: list[int] = []

    def _on_iteration_start(self, *args: Any, **kwargs: Any) -> bool:
        self.iterations.append(self.runner.current_iteration)
        return False


def test_train_stops_before_empty_rollout_update(
    runtime_context: RuntimeContext,
) -> None:
    runner = OnPolicyRunner(context=runtime_context)
    runner.environment = TrackingEnvironment(runtime_context)
    runner.environment.num_envs = 1
    runner.algorithm = MinimalAlgorithm(runtime_context)
    runner.max_iterations = 1
    runner.rollout_length = 1
    callback = StopAtStepStartCallback()
    runner.callbacks = [callback]

    runner.train()

    assert runner.stop_callback == [callback]


def test_train_resumes_from_iteration_after_checkpoint(
    runtime_context: RuntimeContext,
) -> None:
    runner = OnPolicyRunner(context=runtime_context)
    runner.environment = TrackingEnvironment(runtime_context)
    runner.environment.num_envs = 1
    runner.algorithm = MinimalAlgorithm(runtime_context)
    runner.current_iteration = 2
    runner.max_iterations = 5
    runner.rollout_length = 1
    callback = StopAtIterationStartCallback()
    callback.runner = runner
    runner.callbacks = [callback]

    runner.train()

    assert callback.iterations == [3]


@pytest.mark.parametrize("current_iteration", [-2, True, 1.5])
def test_load_rejects_invalid_checkpoint_iteration(
    runtime_context: RuntimeContext,
    current_iteration: Any,
) -> None:
    runner = OnPolicyRunner(context=runtime_context)
    runner.algorithm = MinimalAlgorithm(runtime_context)
    runner._pending_checkpoint_payload = {
        "runner": {
            "stage_index": None,
            "current_iteration": current_iteration,
            "algorithm": {},
        },
    }

    with pytest.raises(ValueError, match="current_iteration"):
        runner.load()


def test_runner_reports_recent_rollout_length_mean(
    runtime_context: RuntimeContext,
) -> None:
    runner = OnPolicyRunner(context=runtime_context)
    runner.current_iteration = 3
    runner._update_rollout_length_history_size(3)
    runner._recent_rollout_lengths.extend((8, 6, 4, 2))

    info = runner._get_info()

    assert info["runner/current_iter"] == 3
    assert info["rollout/mean_len"] == pytest.approx(4.0)


def test_runner_initializes_rollout_length_history_with_zeros(
    runtime_context: RuntimeContext,
) -> None:
    runner = OnPolicyRunner(context=runtime_context)

    assert len(runner._recent_rollout_lengths) == 50
    assert set(runner._recent_rollout_lengths) == {0}


def test_runner_tracks_rollout_length_for_each_environment(
    runtime_context: RuntimeContext,
) -> None:
    runner = OnPolicyRunner(context=runtime_context)
    runner.current_iteration = 0
    runner._environment_rollout_lengths = torch.zeros(
        2,
        dtype=torch.long,
    )

    runner._record_environment_rollout_lengths(
        terminated=torch.tensor([False, False]),
        truncated=torch.tensor([False, False]),
    )
    assert set(runner._recent_rollout_lengths) == {0}

    runner._record_environment_rollout_lengths(
        terminated=torch.tensor([True, False]),
        truncated=torch.tensor([False, False]),
    )
    runner._record_environment_rollout_lengths(
        terminated=torch.tensor([False, False]),
        truncated=torch.tensor([False, True]),
    )

    assert [
        length
        for length in runner._recent_rollout_lengths
        if length > 0
    ] == [2, 3]
    assert runner._environment_rollout_lengths.tolist() == [1, 0]
    info = runner._get_info()
    assert info["rollout/mean_len"] == pytest.approx(0.1)


def test_runner_reports_zero_mean_before_any_environment_finishes(
    runtime_context: RuntimeContext,
) -> None:
    runner = OnPolicyRunner(context=runtime_context)
    runner.current_iteration = 0

    assert runner._get_info() == {
        "runner/current_iter": 0,
        "rollout/mean_len": 0.0,
    }


def test_runner_rejects_invalid_rollout_length_history_size(
    runtime_context: RuntimeContext,
) -> None:
    runner = OnPolicyRunner(context=runtime_context)

    with pytest.raises(ValueError, match="rollout_length_history_size"):
        runner._update_rollout_length_history_size(0)


def test_play_restores_original_environment_after_failure(
    runtime_context: RuntimeContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_config = tmp_path / "environment.yaml"
    environment_config.write_text(
        yaml.safe_dump({"num_envs": 64}),
        encoding="utf-8",
    )
    component = Component(
        runner=None,
        algorithm=None,
        policy=None,
        environment=ComponentInfo(
            type="tracking_test",
            config=environment_config,
        ),
        simulator=None,
        task=None,
    )
    monkeypatch.setitem(
        ENV_TYPE_MAP,
        "tracking_test",
        TrackingEnvironment,
    )
    TrackingEnvironment.instances.clear()

    runner = OnPolicyRunner(context=runtime_context)
    original_environment = TrackingEnvironment(runtime_context)
    original_environment.config_update(component=component, num_envs=64)
    runner.environment = original_environment
    runner.algorithm = MinimalAlgorithm(runtime_context)
    runner._merge_component(component)

    def fail_during_play(
        num_steps: int,
        formats: str | Sequence[str],
        frame_saver: Any,
    ) -> None:
        assert callable(frame_saver)
        assert runner.environment is not None
        assert runner.environment.num_envs == 1
        raise RuntimeError("play failed")

    monkeypatch.setattr(runner, "_play_steps", fail_during_play)

    with pytest.raises(RuntimeError, match="play failed"):
        runner.play(num_steps=1, formats="gif", num_plays=1)

    temporary_environment = TrackingEnvironment.instances[-1]
    assert temporary_environment is not original_environment
    assert temporary_environment.closed is True
    assert runner.environment is original_environment
    assert original_environment.closed is False
    assert original_environment.num_envs == 64


def test_play_reuses_environment_when_concurrent_instances_are_unsupported(
    runtime_context: RuntimeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    TrackingEnvironment.instances.clear()
    runner = OnPolicyRunner(context=runtime_context)
    environment = TrackingEnvironment(runtime_context)
    environment.num_envs = 64
    environment.SUPPORTS_CONCURRENT_INSTANCES = False
    runner.environment = environment
    runner.algorithm = MinimalAlgorithm(runtime_context)

    def fail_during_play(
        num_steps: int,
        formats: str | Sequence[str],
        frame_saver: Any,
    ) -> None:
        assert runner.environment is environment
        assert runner.environment.num_envs == 64
        raise RuntimeError("play failed")

    monkeypatch.setattr(runner, "_play_steps", fail_during_play)

    with pytest.raises(RuntimeError, match="play failed"):
        runner.play(num_steps=1, formats="gif", num_plays=1)

    assert TrackingEnvironment.instances == [environment]
    assert environment.closed is False
    assert runner.environment is environment


def test_playback_file_name_uses_stage_iteration_and_collision_suffix(
    runtime_context: RuntimeContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> "FixedDatetime":
            return cls(2026, 9, 25, 14, 51, 31, tzinfo=tz)

    monkeypatch.setattr(on_policy_module, "datetime", FixedDatetime)
    runner = OnPolicyRunner(context=runtime_context)
    runner.stage_index = 2
    runner.current_iteration = 124
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    first_name = "stage_002_iter_0125_20260925_145131_1"
    assert runner._next_playback_file_name(video_dir) == first_name
    (video_dir / f"{first_name}.gif").touch()
    assert runner._next_playback_file_name(video_dir) == (
        "stage_002_iter_0125_20260925_145131_2"
    )


def test_play_records_requested_number_of_plays(
    runtime_context: RuntimeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = OnPolicyRunner(context=runtime_context)
    environment = TrackingEnvironment(runtime_context)
    environment.SUPPORTS_CONCURRENT_INSTANCES = False
    runner.environment = environment
    runner.algorithm = MinimalAlgorithm(runtime_context)
    recordings: list[int] = []

    def record_playback(num_steps: int, formats: Any, frame_saver: Any) -> bool:
        recordings.append(num_steps)
        return True

    monkeypatch.setattr(runner, "_play_steps", record_playback)
    runner.play(num_steps=12, formats="gif", num_plays=3)

    assert recordings == [12, 12, 12]


@pytest.mark.parametrize("num_plays", [0, -1, True, 1.5])
def test_play_rejects_invalid_play_count(
    runtime_context: RuntimeContext,
    num_plays: Any,
) -> None:
    runner = OnPolicyRunner(context=runtime_context)
    runner.environment = TrackingEnvironment(runtime_context)
    runner.algorithm = MinimalAlgorithm(runtime_context)

    with pytest.raises(ValueError, match="num_plays"):
        runner.play(num_steps=1, formats="gif", num_plays=num_plays)


def test_play_stops_when_primary_environment_episode_ends(
    runtime_context: RuntimeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PlaybackEnvironment(TrackingEnvironment):
        def __init__(self, context: RuntimeContext) -> None:
            super().__init__(context)
            self.num_envs = 2
            self.steps = 0

        def step(self, action: torch.Tensor):
            self.steps += 1
            done = torch.tensor([False, self.steps == 2])
            obs = torch.zeros(2, 1)
            return obs, obs, torch.zeros(2), done, torch.zeros(2, dtype=torch.bool), {}

        def render(self) -> np.ndarray:
            return np.zeros((2, 2, 3), dtype=np.uint8)

        @property
        def playback_env_index(self) -> int:
            return 1

    environment = PlaybackEnvironment(runtime_context)
    runner = OnPolicyRunner(context=runtime_context)
    runner.environment = environment
    runner.callbacks = []
    runner.algorithm = MinimalAlgorithm(runtime_context)
    runner.algorithm.set_eval_mode = lambda: None
    monkeypatch.setattr(
        runner.algorithm,
        "act",
        lambda obs, deterministic=False: PolicyOutput(
            action=torch.zeros(2, 1),
            log_prob=torch.zeros(2),
            value=torch.zeros(2),
        ),
    )
    runner.algorithm.reset_policy_state = lambda env_ids=None: None
    saved_frame_counts: list[int] = []

    runner._play_steps(
        num_steps=10,
        formats="gif",
        frame_saver=lambda frames, **kwargs: (
            saved_frame_counts.append(len(frames)) or []
        ),
    )

    assert environment.steps == 2
    assert saved_frame_counts == [2]


def test_play_warms_up_renderer_before_progress_callbacks(
    runtime_context: RuntimeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle: list[str] = []

    class WarmupEnvironment(TrackingEnvironment):
        @property
        def render_mode(self) -> str:
            return "rgb_array"

        def __init__(self, context: RuntimeContext) -> None:
            super().__init__(context)
            self.num_envs = 1
            self.render_calls = 0

        def render(self) -> np.ndarray | None:
            self.render_calls += 1
            lifecycle.append(f"render:{self.render_calls}")
            if self.render_calls < 3:
                return None
            return np.zeros((2, 2, 3), dtype=np.uint8)

    environment = WarmupEnvironment(runtime_context)
    runner = OnPolicyRunner(context=runtime_context)
    runner.environment = environment
    callback = StopAtStepStartCallback()
    callback._on_play_start = lambda **kwargs: (
        lifecycle.append("play.start") or True
    )
    runner.callbacks = [callback]
    runner.algorithm = MinimalAlgorithm(runtime_context)
    runner.algorithm.set_eval_mode = lambda: None

    runner._play_steps(
        num_steps=10,
        formats="gif",
        frame_saver=lambda frames, **kwargs: [],
    )

    assert lifecycle[:4] == [
        "render:1",
        "render:2",
        "render:3",
        "play.start",
    ]
