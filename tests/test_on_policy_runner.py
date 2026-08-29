from collections.abc import Mapping
from pathlib import Path
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
from utils.component import Component, ComponentInfo


class TrackingEnvironment(BaseEnv):
    instances: list["TrackingEnvironment"] = []

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
    def action_dim(self) -> int:
        return 1


    @property
    def observation_dim(self) -> int:
        return 1


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

    def close(self) -> None:
        return None


class StopAtStepStartCallback(BaseCallback):
    def __init__(self) -> None:
        self.runner = None

    def _on_step_start(self, *args: Any, **kwargs: Any) -> bool:
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

    def fail_during_play(num_steps: int) -> None:
        assert runner.environment is not None
        assert runner.environment.num_envs == 1
        raise RuntimeError("play failed")

    monkeypatch.setattr(runner, "_play_steps", fail_during_play)

    with pytest.raises(RuntimeError, match="play failed"):
        runner.play(num_steps=1)

    temporary_environment = TrackingEnvironment.instances[-1]
    assert temporary_environment is not original_environment
    assert temporary_environment.closed is True
    assert runner.environment is original_environment
    assert original_environment.closed is False
    assert original_environment.num_envs == 64
